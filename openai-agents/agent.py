"""TODO Scanner Agent - OpenAI Agents SDK Implementation (Semi-Explicit Loop)

The SDK manages the loop but exposes a structured decision model.
Each turn produces one of four outcomes: final_output, run_again,
handoff, or interruption. You see the decisions without owning the loop.

Usage:
    uv sync --group openai-agents
    export OPENAI_API_KEY=your-key
    uv run python openai-agents/agent.py
"""

from pathlib import Path

from agents import Agent, Runner, function_tool

from shared_tools import list_files_in_directory, read_file_contents, write_todo_report

# The directory to scan (the sample project)
SCAN_DIR = str(Path(__file__).parent.parent / "sample_project" / "src")
REPORT_PATH = str(Path(__file__).parent / "todo-report.md")


# --- Tool Registration ---
# @function_tool inspects the function signature and docstring
# to generate the tool schema, similar to Pydantic AI.


@function_tool
def list_files(directory: str) -> list[str]:
    """List all source files in the given directory, recursively.

    Args:
        directory: Path to the directory to scan.
    """
    return list_files_in_directory(directory)


@function_tool
def read_file(file_path: str) -> str:
    """Read and return the contents of a source file.

    Args:
        file_path: Path to the file, relative to the scan directory.
    """
    return read_file_contents(file_path, base_directory=SCAN_DIR)


@function_tool
def write_report(todos: list[dict], output_path: str) -> str:
    """Write a categorized TODO report to the specified path.

    Args:
        todos: List of TODO items with file, line, text, and urgency fields.
        output_path: Where to write the Markdown report.
    """
    return write_todo_report(todos, output_path)


# --- Agent Definition ---

agent = Agent(
    name="todo_researcher",
    instructions="""You are a code reviewer that scans source files for TODO comments.

Your task:
1. List all source files in the given directory
2. Read each file and find TODO comments
3. Categorize each TODO by urgency: critical, important, minor, or unknown
   - Look for explicit markers like TODO(critical), TODO(important), TODO(minor)
   - If no marker, infer from context (security issues = critical, etc.)
4. Write a summary report using the write_report tool

Be thorough. Read every file. Do not skip any.""",
    tools=[list_files, read_file, write_report],
)


# --- Run ---

if __name__ == "__main__":
    print(f"Scanning: {SCAN_DIR}")
    print(f"Report:   {REPORT_PATH}")
    print()

    # Runner.run_sync manages the loop. Each turn internally classifies
    # its outcome as one of:
    #   - final_output: LLM produced text, no tool calls -> stop
    #   - run_again: tool calls present -> execute tools, continue
    #   - handoff: delegate to another agent (not used here)
    #   - interruption: tool needs human approval -> pause
    #
    # max_turns limits LLM invocations (tool execution doesn't count).
    result = Runner.run_sync(
        agent,
        f"Scan {SCAN_DIR} for TODO comments and write a report to {REPORT_PATH}",
        max_turns=15,
    )

    print("--- Agent Response ---")
    print(result.final_output)
