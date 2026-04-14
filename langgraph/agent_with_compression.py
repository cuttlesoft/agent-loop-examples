"""TODO Scanner Agent - LangGraph with Context Compression

Extended version of the basic LangGraph agent that handles large
directories (200+ files) without exhausting the context window.

Key additions:
- files_processed counter and findings accumulator in state
- compress_context node that extracts TODOs and trims message history
- Modified routing that compresses every 20 tool calls
- MemorySaver checkpointer for failure recovery

Usage:
    uv sync --group langgraph
    export ANTHROPIC_API_KEY=your-key
    uv run python langgraph/agent_with_compression.py

    # To test with a larger directory:
    uv run python langgraph/agent_with_compression.py /path/to/large/project
"""

import json
import re
import sys
from pathlib import Path
from typing import Annotated, Literal

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import AIMessage, BaseMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict

from shared_tools import list_files_in_directory, read_file_contents, write_todo_report

# Defaults
DEFAULT_SCAN_DIR = str(Path(__file__).parent.parent / "sample_project" / "src")
REPORT_PATH = str(Path(__file__).parent / "todo-report.md")

# How many files to process before compressing context
COMPRESSION_INTERVAL = 20


# --- State Definition ---
# Extended state with compression support. files_processed and findings
# persist outside the message history, so they survive context trimming.

class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    files_processed: int        # track how many files the agent has read
    findings: list[dict]        # accumulate TODOs outside the context window


# --- Tool Definitions ---

@tool
def list_files(directory: str) -> list[str]:
    """List all source files in the given directory, recursively.

    Args:
        directory: Path to the directory to scan.
    """
    return list_files_in_directory(directory)


@tool
def read_file(file_path: str) -> str:
    """Read and return the contents of a source file.

    Args:
        file_path: Path to the file, relative to the scan directory.
    """
    # Resolve scan_dir from the first message or use default
    return read_file_contents(file_path, base_directory=_get_scan_dir())


@tool
def write_report(todos: list[dict], output_path: str) -> str:
    """Write a categorized TODO report to the specified path.

    Args:
        todos: List of TODO items with file, line, text, and urgency fields.
        output_path: Where to write the Markdown report.
    """
    return write_todo_report(todos, output_path)


# --- Helpers ---

# Module-level scan dir, set in main
_scan_dir = DEFAULT_SCAN_DIR


def _get_scan_dir() -> str:
    return _scan_dir


def extract_todos_from_messages(messages: list[BaseMessage]) -> list[dict]:
    """Extract TODO items from recent tool results and AI analysis.

    Scans tool results for TODO patterns and AI messages for
    structured TODO references. Returns a list of TODO dicts.
    """
    todos = []
    todo_pattern = re.compile(
        r"TODO\(?(critical|important|minor)?\)?[:\s]*(.+)",
        re.IGNORECASE,
    )

    for msg in messages:
        if not isinstance(msg, ToolMessage):
            continue

        content = msg.content if isinstance(msg.content, str) else str(msg.content)
        lines = content.split("\n")

        for i, line in enumerate(lines, 1):
            match = todo_pattern.search(line)
            if match:
                urgency = match.group(1) or "unknown"
                text = match.group(2).strip()
                todos.append({
                    "file": "unknown",  # will be refined by the LLM
                    "line": i,
                    "text": text,
                    "urgency": urgency.lower(),
                })

    return todos


def keep_recent_messages(messages: list[BaseMessage], last_n: int = 6) -> list:
    """Keep only the system message (if any) and the last N messages.

    This discards old tool call/result pairs that have already been
    processed, freeing context window space for new file reads.
    """
    # Always keep the first message if it's a system or human message
    kept = []
    if messages and not isinstance(messages[0], (ToolMessage, AIMessage)):
        kept.append(messages[0])

    # Keep the last N messages
    recent = messages[-last_n:] if len(messages) > last_n else messages
    for msg in recent:
        if msg not in kept:
            kept.append(msg)

    return kept


# --- Model Setup ---

tools = [list_files, read_file, write_report]
tools_by_name = {t.name: t for t in tools}
model = ChatAnthropic(model="claude-sonnet-4-6").bind_tools(tools)


# --- Graph Nodes ---

def call_model(state: AgentState) -> dict:
    """Call the LLM with the current message history.

    If findings have been accumulated from previous compression cycles,
    inject a summary so the LLM knows what it has found so far.
    """
    messages = list(state["messages"])

    # If we have accumulated findings from previous batches,
    # remind the LLM about them
    if state.get("findings"):
        count = len(state["findings"])
        by_urgency = {}
        for f in state["findings"]:
            u = f.get("urgency", "unknown")
            by_urgency[u] = by_urgency.get(u, 0) + 1

        reminder = (
            f"Note: You have already found {count} TODOs in previous batches: "
            f"{json.dumps(by_urgency)}. "
            f"Continue scanning remaining files. When done, combine ALL findings "
            f"(previous + new) into the final report."
        )
        messages.append(SystemMessage(content=reminder))

    response = model.invoke(messages)
    return {"messages": [response]}


def tool_node(state: AgentState) -> dict:
    """Execute all tool calls from the last LLM response.

    Tracks the number of read_file calls to drive compression timing.
    """
    results = []
    files_read = 0

    for call in state["messages"][-1].tool_calls:
        tool_fn = tools_by_name[call["name"]]
        result = tool_fn.invoke(call["args"])
        results.append(
            ToolMessage(content=str(result), tool_call_id=call["id"])
        )
        if call["name"] == "read_file":
            files_read += 1

    return {
        "messages": results,
        "files_processed": state.get("files_processed", 0) + files_read,
    }


def compress_context(state: AgentState) -> dict:
    """Compress context by extracting findings and trimming message history.

    This is the key node that makes large directory scanning possible.
    It pulls TODO findings out of the message history into the persistent
    state, then trims old messages to free context window space.
    """
    # Extract TODOs from recent tool results
    recent_todos = extract_todos_from_messages(state["messages"])

    # Keep only recent messages, discard old file contents
    trimmed = keep_recent_messages(state["messages"], last_n=6)

    # Summary for the LLM
    total_found = len(state.get("findings", [])) + len(recent_todos)
    files_done = state.get("files_processed", 0)
    summary = (
        f"Context compressed. Progress: {files_done} files scanned, "
        f"{total_found} TODOs found so far. Continue scanning remaining files."
    )

    print(f"  [compress] {files_done} files processed, {total_found} TODOs accumulated")

    return {
        "messages": trimmed + [SystemMessage(content=summary)],
        "findings": state.get("findings", []) + recent_todos,
    }


# --- Routing ---

def should_continue(
    state: AgentState,
) -> Literal["tool_node", "compress_context", "__end__"]:
    """Route to tools, compression, or termination.

    Compression triggers every COMPRESSION_INTERVAL files to prevent
    the context window from filling up on large directories.
    """
    last_message = state["messages"][-1]

    # No tool calls = LLM is done
    if not last_message.tool_calls:
        return "__end__"

    # Check if we should compress before processing more files
    files_processed = state.get("files_processed", 0)
    if (
        files_processed > 0
        and files_processed % COMPRESSION_INTERVAL == 0
        # Only compress if the next batch includes read_file calls
        and any(
            call["name"] == "read_file"
            for call in last_message.tool_calls
        )
    ):
        return "compress_context"

    return "tool_node"


# --- Build the Graph ---
#
#   START -> call_model -> should_continue -> tool_node ---------> call_model
#                                         -> compress_context ---> call_model
#                                         -> __end__

graph = StateGraph(AgentState)
graph.add_node("call_model", call_model)
graph.add_node("tool_node", tool_node)
graph.add_node("compress_context", compress_context)
graph.add_edge(START, "call_model")
graph.add_conditional_edges("call_model", should_continue)
graph.add_edge("tool_node", "call_model")
graph.add_edge("compress_context", "call_model")

# MemorySaver enables checkpointing: if the agent fails at file 50,
# you can resume from that checkpoint instead of starting over.
agent = graph.compile(checkpointer=MemorySaver())


# --- Run ---

if __name__ == "__main__":
    # Allow overriding the scan directory via command line
    scan_dir = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_SCAN_DIR
    _scan_dir = scan_dir

    print(f"Scanning: {scan_dir}")
    print(f"Report:   {REPORT_PATH}")
    print(f"Compression interval: every {COMPRESSION_INTERVAL} files")
    print()

    result = agent.invoke(
        {
            "messages": [
                SystemMessage(
                    content="""You are a code reviewer that scans source files for TODO comments.

Your task:
1. List all source files in the given directory
2. Read each file and find TODO comments
3. Categorize each TODO by urgency: critical, important, minor, or unknown
   - Look for explicit markers like TODO(critical), TODO(important), TODO(minor)
   - If no marker, infer from context (security issues = critical, etc.)
4. Write a summary report using the write_report tool

Be thorough. Read every file. Do not skip any.
When writing the final report, include ALL TODOs found across all batches."""
                ),
                (
                    "user",
                    f"Scan {scan_dir} for TODO comments and write a report to {REPORT_PATH}",
                ),
            ],
            "files_processed": 0,
            "findings": [],
        },
        # Thread ID for checkpointing
        config={"configurable": {"thread_id": "todo-scan-1"}},
    )

    # Final message
    final_message = result["messages"][-1]
    print()
    print("--- Agent Response ---")
    print(final_message.content)
    print()
    print(f"Files processed: {result.get('files_processed', 'unknown')}")
    print(f"TODOs accumulated in state: {len(result.get('findings', []))}")
