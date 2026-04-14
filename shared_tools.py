"""Shared tool implementations for the TODO scanner agent.

These functions are the actual operations the agent can perform.
Each framework wraps them differently, but the logic is identical.
"""

import json
from datetime import datetime
from pathlib import Path

# File extensions to scan
SOURCE_EXTENSIONS = {
    ".py",
    ".js",
    ".ts",
    ".tsx",
    ".jsx",
    ".rb",
    ".go",
    ".rs",
    ".java",
    ".kt",
    ".swift",
    ".c",
    ".cpp",
    ".h",
}


def list_files_in_directory(directory: str) -> list[str]:
    """Recursively list all source files in the given directory.

    Args:
        directory: Path to the directory to scan.

    Returns:
        List of file paths relative to the directory.
    """
    results = []
    root_path = Path(directory).resolve()

    if not root_path.exists():
        return [f"Error: Directory '{directory}' does not exist."]

    for path in sorted(root_path.rglob("*")):
        if path.is_file() and path.suffix in SOURCE_EXTENSIONS:
            results.append(str(path.relative_to(root_path)))

    return results


def read_file_contents(file_path: str, base_directory: str = ".") -> str:
    """Read and return the contents of a file.

    Args:
        file_path: Path to the file to read (relative to base_directory).
        base_directory: Base directory for resolving relative paths.

    Returns:
        The file contents as a string, or an error message.
    """
    full_path = Path(base_directory).resolve() / file_path

    if not full_path.exists():
        return f"Error: File '{file_path}' does not exist."

    if not full_path.is_file():
        return f"Error: '{file_path}' is not a file."

    try:
        return full_path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return f"Error: '{file_path}' is not a text file."


def write_todo_report(
    todos: list[dict],
    output_path: str,
) -> str:
    """Write a categorized TODO report to the specified path.

    Args:
        todos: List of TODO items. Each item should have:
            - file: str (file path)
            - line: int (line number)
            - text: str (the TODO comment text)
            - urgency: str (critical, important, minor, or unknown)
        output_path: Where to write the Markdown report.

    Returns:
        Confirmation message with the report path and summary counts.
    """
    # Group by urgency
    by_urgency: dict[str, list[dict]] = {
        "critical": [],
        "important": [],
        "minor": [],
        "unknown": [],
    }

    for todo in todos:
        urgency = todo.get("urgency", "unknown").lower()
        if urgency not in by_urgency:
            urgency = "unknown"
        by_urgency[urgency].append(todo)

    # Build report
    lines = [
        "# TODO Report",
        "",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        f"Total TODOs found: {len(todos)}",
        "",
        "## Summary",
        "",
        f"- Critical: {len(by_urgency['critical'])}",
        f"- Important: {len(by_urgency['important'])}",
        f"- Minor: {len(by_urgency['minor'])}",
        f"- Unknown: {len(by_urgency['unknown'])}",
        "",
    ]

    for urgency in ["critical", "important", "minor", "unknown"]:
        items = by_urgency[urgency]
        if not items:
            continue

        lines.append(f"## {urgency.title()}")
        lines.append("")

        for item in items:
            file_path = item.get("file", "unknown")
            line_num = item.get("line", "?")
            text = item.get("text", "")
            lines.append(f"- **{file_path}:{line_num}** - {text}")

        lines.append("")

    report = "\n".join(lines)

    # Write to file
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(report, encoding="utf-8")

    counts = {k: len(v) for k, v in by_urgency.items() if v}
    return f"Report written to {output_path}. Found {len(todos)} TODOs: {json.dumps(counts)}"
