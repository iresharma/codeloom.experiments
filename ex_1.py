#!/usr/bin/env python3
"""
Minimal CLI coding agent.

This is a deliberately small, single-file example of how tools like
Devin / Claude Code / Cursor's agent work under the hood:

    1. You give the model a task + a set of tools (read_file, write_file, run_shell).
    2. The model responds with either plain text (done) or a "tool_use" request.
    3. You execute that tool yourself and feed the result back.
    4. Repeat until the model stops asking for tools, or you hit a turn limit.

Nothing here is magic. There's no framework, no hidden state. Just a while
loop, a dict of Python functions, and the Claude API's tool-use feature.

Usage:
    export ANTHROPIC_API_KEY=sk-ant-...
    python agent.py --workdir ./my_project "Fix the bug in utils.py where divide() crashes on zero"

Requires:
    pip install anthropic
"""

import argparse
import os
import subprocess
import sys

from anthropic import Anthropic

MODEL = "claude-sonnet-4-6"
MAX_TURNS = 25          # hard cap so a buggy loop can't run forever
SHELL_TIMEOUT = 60      # seconds


# ---------------------------------------------------------------------------
# 1. TOOL SCHEMAS
#
# This is the ONLY thing the model actually "sees". It never sees your
# Python code below -- it only sees these names, descriptions, and
# parameter schemas, and decides which one to call based on them.
# ---------------------------------------------------------------------------

TOOLS = [
    {
        "name": "read_file",
        "description": (
            "Read and return the full text contents of a file at the given "
            "path, relative to the working directory. Returns an error "
            "message string if the file does not exist or can't be read."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Relative path to the file, e.g. 'src/utils.py'",
                }
            },
            "required": ["path"],
        },
    },
    {
        "name": "write_file",
        "description": (
            "Write text content to a file at the given path, overwriting it "
            "if it already exists (creating parent directories if needed). "
            "Always write the FULL new file contents, not a diff/patch."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Relative path to the file"},
                "content": {"type": "string", "description": "Full new contents of the file"},
            },
            "required": ["path", "content"],
        },
    },
    {
        "name": "list_dir",
        "description": "List files and directories at the given relative path (non-recursive).",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Relative directory path, e.g. '.' or 'src'"}
            },
            "required": ["path"],
        },
    },
    {
        "name": "run_shell",
        "description": (
            "Run a shell command inside the working directory (e.g. run tests, "
            "grep for text, install a package). Returns stdout, stderr, and the "
            f"exit code. Times out after {SHELL_TIMEOUT} seconds."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "command": {"type": "string", "description": "The shell command to run"}
            },
            "required": ["command"],
        },
    },
]


# ---------------------------------------------------------------------------
# 2. TOOL IMPLEMENTATIONS ("the dispatcher")
#
# The model can only ever produce JSON like {"name": "run_shell", "input":
# {"command": "..."}}. It is THIS code -- yours -- that decides what that
# JSON is actually allowed to do. This is your security boundary: restrict
# paths, block dangerous commands, run in Docker, whatever you need.
# ---------------------------------------------------------------------------

class Sandbox:
    """Confines every tool call to a single working directory."""

    def __init__(self, workdir: str):
        self.workdir = os.path.abspath(workdir)
        os.makedirs(self.workdir, exist_ok=True)

    def _resolve(self, rel_path: str) -> str:
        """Resolve a relative path and refuse to escape the working directory."""
        full = os.path.abspath(os.path.join(self.workdir, rel_path))
        if not full.startswith(self.workdir):
            raise ValueError(f"Path '{rel_path}' escapes the working directory")
        return full

    def read_file(self, path: str) -> str:
        try:
            full = self._resolve(path)
            with open(full, "r", errors="replace") as f:
                return f.read()
        except Exception as e:
            return f"Error reading '{path}': {e}"

    def write_file(self, path: str, content: str) -> str:
        try:
            full = self._resolve(path)
            os.makedirs(os.path.dirname(full) or ".", exist_ok=True)
            with open(full, "w") as f:
                f.write(content)
            return f"Wrote {len(content)} chars to '{path}'"
        except Exception as e:
            return f"Error writing '{path}': {e}"

    def list_dir(self, path: str) -> str:
        try:
            full = self._resolve(path)
            entries = sorted(os.listdir(full))
            if not entries:
                return "(empty directory)"
            return "\n".join(entries)
        except Exception as e:
            return f"Error listing '{path}': {e}"

    def run_shell(self, command: str) -> str:
        # NOTE: this is intentionally simple for learning purposes.
        # In anything beyond a toy, run this inside a Docker container
        # instead of directly on your machine -- an LLM-issued shell
        # command can do anything a real shell command can do.
        try:
            result = subprocess.run(
                command,
                shell=True,
                cwd=self.workdir,
                capture_output=True,
                text=True,
                timeout=SHELL_TIMEOUT,
            )
            return (
                f"exit_code: {result.returncode}\n"
                f"stdout:\n{result.stdout}\n"
                f"stderr:\n{result.stderr}"
            )
        except subprocess.TimeoutExpired:
            return f"Error: command timed out after {SHELL_TIMEOUT}s"
        except Exception as e:
            return f"Error running command: {e}"


def execute_tool(sandbox: Sandbox, name: str, tool_input: dict) -> str:
    """Name -> function dispatch. This is the whole 'engine'."""
    if name == "read_file":
        return sandbox.read_file(tool_input["path"])
    elif name == "write_file":
        return sandbox.write_file(tool_input["path"], tool_input["content"])
    elif name == "list_dir":
        return sandbox.list_dir(tool_input["path"])
    elif name == "run_shell":
        return sandbox.run_shell(tool_input["command"])
    else:
        return f"Error: unknown tool '{name}'"


# ---------------------------------------------------------------------------
# 3. THE AGENT LOOP
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are a careful coding agent working inside a sandboxed
project directory. You have tools to read files, write files, list
directories, and run shell commands (e.g. tests).

Approach every task like this:
1. Explore first: list_dir / read_file to understand the relevant code
   before changing anything. Don't guess at file contents.
2. Make the smallest change that correctly solves the task.
3. After editing, verify your work: run tests or otherwise execute the
   code to check it actually works. Don't assume -- check.
4. If something fails, read the error output carefully and fix it, then
   re-verify. Keep iterating until it works or you're truly stuck.
5. When you're done, reply with plain text (no tool call) summarizing what
   you changed and why. That ends the task.
"""


def run_agent(task: str, workdir: str, max_turns: int = MAX_TURNS, verbose: bool = True):
    client = Anthropic()  # reads ANTHROPIC_API_KEY from env
    sandbox = Sandbox(workdir)

    messages = [{"role": "user", "content": task}]

    for turn in range(1, max_turns + 1):
        if verbose:
            print(f"\n=== Turn {turn} ===")

        response = client.messages.create(
            model=MODEL,
            max_tokens=4096,
            system=SYSTEM_PROMPT,
            messages=messages,
            tools=TOOLS,
        )

        # Print any text the model produced this turn (its reasoning / summary)
        for block in response.content:
            if block.type == "text" and block.text.strip():
                print(f"\n[assistant]: {block.text}")

        messages.append({"role": "assistant", "content": response.content})

        # If the model didn't ask for a tool, it's done.
        if response.stop_reason != "tool_use":
            print("\n=== Agent finished ===")
            return

        # Otherwise, execute every requested tool call and feed results back.
        tool_results = []
        for block in response.content:
            if block.type != "tool_use":
                continue

            if verbose:
                print(f"[tool_call] {block.name}({block.input})")

            result = execute_tool(sandbox, block.name, block.input)

            if verbose:
                preview = result if len(result) < 500 else result[:500] + "... [truncated]"
                print(f"[tool_result] {preview}")

            tool_results.append(
                {
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": result,
                }
            )

        messages.append({"role": "user", "content": tool_results})

    print(f"\n=== Stopped after {max_turns} turns without finishing ===")


# ---------------------------------------------------------------------------
# 4. CLI ENTRY POINT
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Minimal CLI coding agent")
    parser.add_argument("task", help="Natural-language description of what the agent should do")
    parser.add_argument("--workdir", default="./workspace", help="Directory the agent is confined to")
    parser.add_argument("--max-turns", type=int, default=MAX_TURNS)
    args = parser.parse_args()

    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("Error: set ANTHROPIC_API_KEY in your environment first.", file=sys.stderr)
        sys.exit(1)

    run_agent(args.task, args.workdir, args.max_turns)


if __name__ == "__main__":
    main()