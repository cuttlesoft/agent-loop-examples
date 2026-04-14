"""TODO Scanner Agent - Pydantic AI Implementation (Implicit Loop)

The loop is entirely hidden. You register tools, call run_sync,
and the framework handles iteration, tool execution, and termination.

Usage:
    uv sync --group pydantic-ai
    export ANTHROPIC_API_KEY=your-key
    uv run python pydanticai/agent.py
"""

from pathlib import Path

from pydantic_ai import Agent
from pydantic_ai.usage import UsageLimits

from shared_tools import list_files_in_directory, read_file_contents, write_todo_report

# The directory to scan (the sample project)
SCAN_DIR = str(Path(__file__).parent.parent / "sample_project" / "src")
REPORT_PATH = str(Path(__file__).parent / "todo-report.md")

# --- Agent Definition ---

agent = Agent(
    "anthropic:claude-sonnet-4-6",
    instructions="""You are a code reviewer that scans source files for TODO comments.

Your task:
1. List all source files in the given directory
2. Read each file and find TODO comments
3. Categorize each TODO by urgency: critical, important, minor, or unknown
   - Look for explicit markers like TODO(critical), TODO(important), TODO(minor)
   - If no marker, infer from context (security issues = critical, etc.)
4. Write a summary report using the write_report tool

Be thorough. Read every file. Do not skip any.""",
)


# --- Tool Registration ---
# Pydantic AI extracts the function signature, type annotations,
# and docstring to build the tool schema automatically.


@agent.tool_plain
def list_files(directory: str) -> list[str]:
    """List all source files in the given directory, recursively.

    Args:
        directory: Path to the directory to scan.
    """
    return list_files_in_directory(directory)


@agent.tool_plain
def read_file(file_path: str) -> str:
    """Read and return the contents of a source file.

    Args:
        file_path: Path to the file, relative to the scan directory.
    """
    return read_file_contents(file_path, base_directory=SCAN_DIR)


@agent.tool_plain
def write_report(todos: list[dict], output_path: str) -> str:
    """Write a categorized TODO report to the specified path.

    Args:
        todos: List of TODO items with file, line, text, and urgency fields.
        output_path: Where to write the Markdown report.
    """
    return write_todo_report(todos, output_path)


# --- Run ---

if __name__ == "__main__":
    print(f"Scanning: {SCAN_DIR}")
    print(f"Report:   {REPORT_PATH}")
    print()

    # run_sync handles the entire loop:
    # - Sends the prompt and tool definitions to the LLM
    # - Receives tool calls, executes them, sends results back
    # - Repeats until the LLM responds with text (no tool calls)
    # - Returns the final result
    #
    # UsageLimits prevents infinite loops. request_limit caps LLM calls
    # (not tool calls). Without this, a confused LLM could loop forever.
    result = agent.run_sync(
        f"Scan {SCAN_DIR} for TODO comments and write a report to {REPORT_PATH}",
        usage_limits=UsageLimits(request_limit=25),
    )

    print("--- Agent Response ---")
    print(result.output)
    print()
    print(f"Requests used: {result.usage().requests}")
    print(f"Tokens used:   {result.usage().total_tokens}")
