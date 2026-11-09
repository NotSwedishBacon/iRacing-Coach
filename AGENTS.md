# Agent Instructions (`AGENTS.md`)

## Mandatory Git Branching & Commit Workflow

1. **Always Work on a Feature/Working Branch (Never Push Directly to `main`)**:
   - Do not commit or push directly to `main` during iterative development.
   - If currently on `main`, create and switch to a descriptive working branch (e.g., `feat/<topic>`, `fix/<topic>`, or `chore/<topic>`) before committing.

2. **Commit & Push After Every Interaction**:
   - At the end of **every** interaction that modifies code, documentation, data, or configuration, stage all relevant changes, run the test suite (`.venv/bin/pytest`) to verify everything is green, commit with a clear conventional commit message, and push the branch to `origin`:
     ```bash
     git checkout -b <branch-name>   # if not already on a non-main working branch
     git add -A
     git commit -m "<type>: <concise summary of interaction changes>"
     git push -u origin <branch-name>
     ```
   - This allows the user to review, test, and fine-tune changes on the branch before merging into `main`.

## Project Verification Standards

- **Test Execution**: Always run `.venv/bin/pytest -v` before committing code changes.
- **Static Corpus & Vector Store**: The scraped GitBook documentation (`data/gitbook_docs/`), chunk metadata (`data/gitbook_chunks.json`), and ChromaDB store (`data/chroma_db/`) are version-controlled directly in Git. Never re-introduce live web scraping dependencies unless explicitly requested.
- **Deterministic Telemetry Math**: All telemetry physics calculations in `telemetry/signal_math.py` must remain strictly deterministic NumPy/SciPy computations without LLM estimation.
