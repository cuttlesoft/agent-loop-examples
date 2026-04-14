# Agent Loop Examples

[![Ship AI agents that actually work in production.](https://cuttlesoft.com/api/img/promo/?title=Ship+AI+Agents+That+Work+in+Production&subtitle=&cta=Accelerate+Your+AI+Roadmap&size=banner&color=purple)](https://cuttlesoft.com/services/ai-ml-development-and-consulting/)

Three implementations of the same agent task across three frameworks, demonstrating the spectrum from implicit to fully explicit agent loops.

**Blog post:** [The Agent Loop: One Pattern, Three Frameworks](https://cuttlesoft.com/blog/2026/04/14/the-agent-loop-one-pattern-three-frameworks/)

## The Task

A file system researcher that scans a directory for TODO comments in source files, categorizes each one by urgency (critical, important, minor, unknown), and writes a Markdown summary report.

The `sample_project/` directory contains Python source files with 13 TODO comments at various urgency levels. Each agent implementation scans this directory using the same three tools.

## Structure

```bash
.
├── pyproject.toml           # Project config and framework dependencies (uv)
├── sample_project/          # Source files with TODO comments to scan
│   └── src/
│       ├── api/routes.py    # 5 TODOs (2 critical, 2 important, 1 minor)
│       ├── utils/cache.py   # 3 TODOs (1 important, 2 minor)
│       └── models/user.py   # 5 TODOs (1 critical, 2 important, 2 minor)
├── shared_tools.py          # Tool implementations shared by all three agents
├── pydanticai/              # Implicit loop (framework manages everything)
├── openai-agents/           # Semi-explicit loop (structured turn classification)
└── langgraph/               # Fully explicit loop (graph-based control flow)
```

## Shared Tools

All three implementations use the same underlying functions from `shared_tools.py`:

- **list_files** - Recursively list source files in a directory
- **read_file** - Read file contents
- **write_report** - Write a categorized Markdown report

Each framework wraps these functions differently (decorators, tool objects, etc.), but the logic is identical.

## Setup

Requires [uv](https://docs.astral.sh/uv/) and Python 3.11+.

```bash
uv sync --all-groups
```

Or install only one framework's dependencies:

```bash
uv sync --group pydantic-ai
uv sync --group openai-agents
uv sync --group langgraph
```

Set your API keys:

```bash
export ANTHROPIC_API_KEY=your-key   # Pydantic AI and LangGraph
export OPENAI_API_KEY=your-key      # OpenAI Agents SDK
```

### Pydantic AI (Implicit Loop)

```bash
uv run python pydanticai/agent.py
```

The loop is entirely hidden. You register tools, call `run_sync`, and the framework handles iteration, tool execution, and termination internally.

### OpenAI Agents SDK (Semi-Explicit Loop)

```bash
uv run python openai-agents/agent.py
```

The SDK manages the loop but exposes a structured decision model. Each turn is classified as `final_output`, `run_again`, `handoff`, or `interruption`.

### LangGraph (Fully Explicit Loop)

```bash
uv run python langgraph/agent.py
```

The loop is a directed cyclic graph you define yourself. Nodes are functions, edges are routing decisions, and the cycle is the loop.

## The Abstraction Spectrum

| Aspect                | Pydantic AI              | OpenAI Agents SDK          | LangGraph                   |
| --------------------- | ------------------------ | -------------------------- | --------------------------- |
| Loop visibility       | Hidden                   | Classified per-turn        | Fully visible               |
| Control flow          | Framework-owned          | Framework-owned with hooks | Developer-owned             |
| Stop condition        | `UsageLimits`            | `max_turns`                | `should_continue` edge      |
| Mid-loop intervention | Limited (`agent.iter()`) | Guardrails, streaming      | Any node insertion          |
| Checkpointing         | No                       | Session-based              | Built-in (every node)       |
| Best for              | Simple tool-using agents | Multi-agent handoffs       | Durable/resumable workflows |

## Expected Output

Each agent produces a `todo-report.md` in its own directory with contents like:

```markdown
# TODO Report

Generated: 2026-04-13 14:30
Total TODOs found: 13

## Summary

- Critical: 3
- Important: 5
- Minor: 5
- Unknown: 0

## Critical

- api/routes.py:7 - Rate limiting is not implemented...
- api/routes.py:22 - This currently hard-deletes...
- models/user.py:15 - Password is stored as plain text...

...
```

## Scaling: Context Compression

The blog post includes a fourth variation showing how to add context compression to the LangGraph agent for scanning 200+ files. That example adds a compression node to the graph, demonstrating mid-loop intervention — one of the key advantages of explicit control flow. See [the blog post](https://cuttlesoft.com/blog/2026/04/14/the-agent-loop-one-pattern-three-frameworks/) for the annotated walkthrough.

## ⚖️ License

MIT

## 🤝 Need Help with Your AI/ML Project?

Cuttlesoft specializes in [AI/ML development and consulting](https://cuttlesoft.com/services/ai-ml-development-and-consulting/), including LLM integration, AI product development, and custom AI solutions. Whether you need help building RAG pipelines, fine-tuning models, or shipping AI-powered features to production, our team of expert developers is here to help.

[Contact us](https://cuttlesoft.com/contact/) to discuss how we can bring AI to your project!
