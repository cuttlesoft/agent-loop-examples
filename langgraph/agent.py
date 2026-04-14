"""TODO Scanner Agent - LangGraph Implementation (Fully Explicit Loop)

The loop is a directed cyclic graph you define yourself. Nodes are
functions, edges are routing decisions, and the cycle IS the loop.
You own the control flow.

Usage:
    uv sync --group langgraph
    export ANTHROPIC_API_KEY=your-key
    uv run python langgraph/agent.py
"""

from pathlib import Path
from typing import Annotated, Literal

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import SystemMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.graph import START, StateGraph
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict

from shared_tools import list_files_in_directory, read_file_contents, write_todo_report

# The directory to scan (the sample project)
SCAN_DIR = str(Path(__file__).parent.parent / "sample_project" / "src")
REPORT_PATH = str(Path(__file__).parent / "todo-report.md")


# --- State Definition ---
# The state flows through every node in the graph.
# add_messages reducer means each node APPENDS to the list
# rather than replacing it.


class AgentState(TypedDict):
    messages: Annotated[list, add_messages]


# --- Tool Definitions ---
# LangGraph uses @tool from langchain_core.


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
    return read_file_contents(file_path, base_directory=SCAN_DIR)


@tool
def write_report(todos: list[dict], output_path: str) -> str:
    """Write a categorized TODO report to the specified path.

    Args:
        todos: List of TODO items with file, line, text, and urgency fields.
        output_path: Where to write the Markdown report.
    """
    return write_todo_report(todos, output_path)


# --- Model Setup ---

tools = [list_files, read_file, write_report]
tools_by_name = {t.name: t for t in tools}
model = ChatAnthropic(model="claude-sonnet-4-6").bind_tools(tools)


# --- Graph Nodes ---
# Each node is a function that receives state and returns state updates.


def call_model(state: AgentState) -> dict:
    """Call the LLM with the current message history."""
    response = model.invoke(state["messages"])
    return {"messages": [response]}


def tool_node(state: AgentState) -> dict:
    """Execute all tool calls from the last LLM response."""
    results = []
    for call in state["messages"][-1].tool_calls:
        tool_fn = tools_by_name[call["name"]]
        result = tool_fn.invoke(call["args"])
        results.append(ToolMessage(content=str(result), tool_call_id=call["id"]))
    return {"messages": results}


# --- Routing ---
# The conditional edge that creates the loop.
# Tool calls -> continue. No tool calls -> stop.


def should_continue(state: AgentState) -> Literal["tool_node", "__end__"]:
    """Decide whether to execute tools or stop."""
    last_message = state["messages"][-1]
    if last_message.tool_calls:
        return "tool_node"
    return "__end__"


# --- Build the Graph ---
# This is the loop, made explicit.
#
#   START -> call_model -> should_continue -> tool_node -> call_model -> ...
#                                         -> __end__

graph = StateGraph(AgentState)
graph.add_node("call_model", call_model)
graph.add_node("tool_node", tool_node)
graph.add_edge(START, "call_model")
graph.add_conditional_edges("call_model", should_continue)
graph.add_edge("tool_node", "call_model")  # The loop: tools -> back to LLM

agent = graph.compile()


# --- Run ---

if __name__ == "__main__":
    print(f"Scanning: {SCAN_DIR}")
    print(f"Report:   {REPORT_PATH}")
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

Be thorough. Read every file. Do not skip any."""
                ),
                (
                    "user",
                    f"Scan {SCAN_DIR} for TODO comments and write a report to {REPORT_PATH}",
                ),
            ]
        }
    )

    # The final message in state is the LLM's text response
    final_message = result["messages"][-1]
    print("--- Agent Response ---")
    print(final_message.content)
