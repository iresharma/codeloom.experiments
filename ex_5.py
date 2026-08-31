#!/usr/bin/env python3
"""
Minimal CLI coding agent.

This is a deliberately small, single-file example of how tools like
Devin / Claude Code / Cursor's agent work under the hood:

    1. You give the model a task + a set of tools (read_file, write_file, ...).
    2. The model responds with either plain text (done) or a "tool_use" request.
    3. You execute that tool yourself and feed the result back.
    4. Repeat until the model stops asking for tools, or you hit a turn limit.

Nothing here is magic. There's no framework, no hidden state. Just a while
loop, a dict of Python functions, and the Claude API's tool-use feature.

Usage:
    export ANTHROPIC_API_KEY=sk-ant-...
    python ex_5.py --repo https://github.com/pallets/flask.git
    # then keep asking questions at the [you]: prompt; type exit to quit

Requires:
    pip install anthropic
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys

from anthropic import Anthropic

MODEL = "claude-sonnet-4-6"
MAX_TURNS = 50          # hard cap so a buggy loop can't run forever
SHELL_TIMEOUT = 60      # seconds

# Caps on a single read_file result, so one huge file can't blow the context
# window. The model is told when truncation happened and which line to resume
# from, so it can page through deliberately — and then patch that same window.
MAX_READ_LINES = 2000
MAX_READ_BYTES = 250_000
BINARY_SNIFF_BYTES = 8192

HUNK_HEADER_RE = re.compile(
    r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@"
)

# Terminal colors: agent=blue, user=green, tool=red
BLUE = "\033[94m"
GREEN = "\033[92m"
RED = "\033[91m"
RESET = "\033[0m"


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
            "Read a file at the given path, relative to the working directory. "
            "Returns contents with a 1-based line-number gutter. Optional "
            "start_line/end_line read a slice — use that for large files, then "
            "edit the same window with replace_lines. Long files are truncated."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Relative path to the file, e.g. 'src/utils.py'",
                },
                "start_line": {
                    "type": "integer",
                    "description": "1-based line to start reading from (inclusive)",
                },
                "end_line": {
                    "type": "integer",
                    "description": "1-based line to stop reading at (inclusive)",
                },
            },
            "required": ["path"],
        },
    },
    {
        "name": "write_file",
        "description": (
            "Create a new file, or overwrite an existing one with the FULL "
            "new contents. Use this only for new files or complete rewrites "
            "of small files. For surgical edits prefer str_replace; for a "
            "window you just read prefer replace_lines; for multi-hunk "
            "structural edits prefer apply_diff."
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
        "name": "str_replace",
        "description": (
            "Replace exactly one occurrence of old_string with new_string. "
            "old_string is matched as an exact literal (whitespace and "
            "indent must match). It MUST be unique in the file — if it "
            "matches zero times or more than once the call fails and "
            "nothing is written. That uniqueness check is the main "
            "guardrail against silent edit corruption. Include enough "
            "surrounding lines to make the match unique. Set replace_all "
            "only when you intentionally want every occurrence changed."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Relative path to the file",
                },
                "old_string": {
                    "type": "string",
                    "description": "Exact text to find. Must be unique unless replace_all is true.",
                },
                "new_string": {
                    "type": "string",
                    "description": "Replacement text",
                },
                "replace_all": {
                    "type": "boolean",
                    "description": "If true, replace every occurrence. Default false.",
                },
            },
            "required": ["path", "old_string", "new_string"],
        },
    },
    {
        "name": "replace_lines",
        "description": (
            "Replace an inclusive 1-based line range with content. The "
            "range should be a window you already read (the gutter line "
            "numbers from read_file). Fails if the range is out of bounds "
            "— it will not clip, wrap, or guess. "
            "To insert before line N, pass start_line=N and end_line=N-1. "
            "To append, pass start_line=total_lines+1 and end_line=total_lines. "
            "content is spliced in as-is; include trailing newlines as they "
            "should appear in the file."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Relative path to the file",
                },
                "start_line": {
                    "type": "integer",
                    "description": "First line to replace (1-based, inclusive)",
                },
                "end_line": {
                    "type": "integer",
                    "description": (
                        "Last line to replace (1-based, inclusive). "
                        "end_line = start_line - 1 inserts before start_line."
                    ),
                },
                "content": {
                    "type": "string",
                    "description": "Replacement text for that range",
                },
            },
            "required": ["path", "start_line", "end_line", "content"],
        },
    },
    {
        "name": "apply_diff",
        "description": (
            "Apply a unified diff (---/+++ and @@ hunks, or just the hunks) "
            "to an existing file. Each hunk's old lines must match the file "
            "exactly and uniquely — no fuzzy matching, no guessing the "
            "nearest neighbor. Use this for larger structural edits that "
            "span several places in one file. Compact, and a format models "
            "are heavily trained on."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Relative path to the file to patch",
                },
                "diff": {
                    "type": "string",
                    "description": "Unified diff against the current file contents",
                },
            },
            "required": ["path", "diff"],
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
        "name": "clone_repo",
        "description": (
            "Clone a git repository into the working directory. Use this "
            "first if the workspace is empty. Fails (and does not overwrite) "
            "if the working directory already has files."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "repo_url": {
                    "type": "string",
                    "description": "Remote URL to clone, e.g. 'https://github.com/pallets/flask.git'",
                }
            },
            "required": ["repo_url"],
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

    @staticmethod
    def _looks_binary(full: str) -> bool:
        try:
            with open(full, "rb") as f:
                return b"\x00" in f.read(BINARY_SNIFF_BYTES)
        except OSError:
            return False

    @staticmethod
    def _number_lines(lines: list[str], first_lineno: int) -> str:
        out = []
        for offset, text in enumerate(lines):
            out.append(f"{first_lineno + offset:>6}\t{text.rstrip(chr(10))}")
        return "\n".join(out)

    @staticmethod
    def _split_keepends(text: str) -> list[str]:
        if text == "":
            return []
        return text.splitlines(keepends=True)

    @staticmethod
    def _detect_newline(text: str) -> str:
        if "\r\n" in text:
            return "\r\n"
        if "\r" in text and "\n" not in text:
            return "\r"
        return "\n"

    def _read_text(self, path: str) -> tuple[str, str]:
        """Return (absolute path, file text). Raises on binary / missing / dir."""
        full = self._resolve(path)
        if os.path.isdir(full):
            raise ValueError(f"'{path}' is a directory")
        if not os.path.isfile(full):
            raise FileNotFoundError(f"'{path}' does not exist")
        if self._looks_binary(full):
            size = os.path.getsize(full)
            raise ValueError(f"'{path}' looks like a binary file ({size} bytes)")
        with open(full, "r", errors="replace") as f:
            return full, f.read()

    def _write_text(self, full: str, content: str) -> None:
        os.makedirs(os.path.dirname(full) or ".", exist_ok=True)
        with open(full, "w") as f:
            f.write(content)

    def read_file(
        self,
        path: str,
        start_line: int | None = None,
        end_line: int | None = None,
    ) -> str:
        try:
            full = self._resolve(path)

            if os.path.isdir(full):
                return f"Error reading '{path}': it is a directory — use list_dir instead"
            if self._looks_binary(full):
                size = os.path.getsize(full)
                return f"'{path}' looks like a binary file ({size} bytes) — not shown."

            with open(full, "r", errors="replace") as f:
                lines = f.readlines()

            total = len(lines)
            if total == 0:
                return f"'{path}' is empty (0 lines)"

            start = max(0, (start_line or 1) - 1)
            end = total if end_line is None else min(end_line, total)
            if start >= total:
                return (
                    f"Error reading '{path}': start_line {start_line} is past "
                    f"the end of the file ({total} lines)"
                )
            if end <= start:
                return (
                    f"Error reading '{path}': end_line {end_line} is not after "
                    f"start_line {start_line}"
                )

            window = lines[start:end]
            truncated_at = None
            if len(window) > MAX_READ_LINES:
                window = window[:MAX_READ_LINES]
                truncated_at = start + MAX_READ_LINES
            running = 0
            for i, text in enumerate(window):
                running += len(text.encode("utf-8", errors="replace"))
                if running > MAX_READ_BYTES:
                    window = window[: i or 1]
                    truncated_at = start + len(window)
                    break

            body = self._number_lines(window, start + 1)
            shown_last = start + len(window)
            header = f"{path} — lines {start + 1}-{shown_last} of {total}"
            footer = ""
            if truncated_at is not None:
                footer = (
                    f"\n\n[truncated — {total - truncated_at} more line(s); "
                    f"continue with start_line={truncated_at + 1}]"
                )
            return f"{header}\n\n{body}{footer}"
        except Exception as e:
            return f"Error reading '{path}': {e}"

    def write_file(self, path: str, content: str) -> str:
        try:
            full = self._resolve(path)
            self._write_text(full, content)
            return f"Wrote {len(content)} chars to '{path}'"
        except Exception as e:
            return f"Error writing '{path}': {e}"

    def str_replace(
        self,
        path: str,
        old_string: str,
        new_string: str,
        replace_all: bool = False,
    ) -> str:
        """Exact-string edit. Unique match, or fail — never pick a candidate."""
        try:
            if old_string == "":
                return (
                    f"Error in str_replace '{path}': old_string is empty. "
                    "Refusing to match every empty position in the file."
                )
            if old_string == new_string:
                return (
                    f"Error in str_replace '{path}': old_string and new_string "
                    "are identical — nothing would change."
                )

            full, text = self._read_text(path)
            matches = _find_literal_matches(text, old_string)
            n = len(matches)

            if n == 0:
                return (
                    f"Error in str_replace '{path}': old_string was not found. "
                    "Read the file again — contents may have changed, or "
                    "whitespace/indent may differ. Nothing was written."
                )
            if n > 1 and not replace_all:
                lines = ", ".join(str(m) for m in matches)
                return (
                    f"Error in str_replace '{path}': old_string matched {n} "
                    f"times (lines {lines}). It must be unique. Add more "
                    "surrounding context to disambiguate, or set "
                    "replace_all=true if you really want every occurrence. "
                    "Nothing was written."
                )

            updated = text.replace(
                old_string, new_string, -1 if replace_all else 1
            )
            self._write_text(full, updated)
            if replace_all:
                return (
                    f"Replaced {n} occurrence(s) in '{path}' "
                    f"({len(text)} → {len(updated)} chars)"
                )
            return (
                f"Replaced 1 occurrence in '{path}' at line {matches[0]} "
                f"({len(text)} → {len(updated)} chars)"
            )
        except Exception as e:
            return f"Error in str_replace '{path}': {e}"

    def replace_lines(
        self,
        path: str,
        start_line: int,
        end_line: int,
        content: str,
    ) -> str:
        """Splice content into a 1-based inclusive line window already in view."""
        try:
            start_line = int(start_line)
            end_line = int(end_line)
            full, text = self._read_text(path)
            lines = self._split_keepends(text)
            total = len(lines)

            # Insert-before-N: end_line == start_line - 1 (empty range).
            # Append: start_line == total + 1, end_line == total.
            if start_line < 1:
                return (
                    f"Error in replace_lines '{path}': start_line must be >= 1 "
                    f"(got {start_line}). Nothing was written."
                )
            if end_line < start_line - 1:
                return (
                    f"Error in replace_lines '{path}': end_line {end_line} is "
                    f"before start_line-1 ({start_line - 1}). Nothing was written."
                )
            if start_line > total + 1:
                return (
                    f"Error in replace_lines '{path}': start_line {start_line} "
                    f"is past the end of the file ({total} lines). "
                    f"To append, use start_line={total + 1}, end_line={total}. "
                    "Nothing was written."
                )
            if end_line > total:
                return (
                    f"Error in replace_lines '{path}': end_line {end_line} is "
                    f"past the end of the file ({total} lines). "
                    "Nothing was written."
                )

            head = lines[: start_line - 1]
            tail = lines[end_line:]
            new_lines = self._split_keepends(content)
            updated_lines = head + new_lines + tail
            updated = "".join(updated_lines)
            self._write_text(full, updated)

            replaced = end_line - start_line + 1  # 0 when inserting
            kind = "inserted" if replaced == 0 else "replaced"
            return (
                f"{kind.capitalize()} {replaced} line(s) at "
                f"{start_line}-{end_line} in '{path}' with {len(new_lines)} "
                f"line(s); file is now {len(updated_lines)} lines "
                f"({len(text)} → {len(updated)} chars)"
            )
        except Exception as e:
            return f"Error in replace_lines '{path}': {e}"

    def apply_diff(self, path: str, diff: str) -> str:
        """Apply a unified diff. Hunks must match uniquely; no fuzz."""
        try:
            full, text = self._read_text(path)
            hunks = _parse_unified_diff(diff)
            if not hunks:
                return (
                    f"Error in apply_diff '{path}': no @@ hunks found in the "
                    "diff. Nothing was written."
                )

            new_text, applied = _apply_hunks(text, hunks)
            self._write_text(full, new_text)
            return (
                f"Applied {applied} hunk(s) to '{path}' "
                f"({len(text)} → {len(new_text)} chars)"
            )
        except _DiffError as e:
            return f"Error in apply_diff '{path}': {e} Nothing was written."
        except Exception as e:
            return f"Error in apply_diff '{path}': {e}"

    def clone_repo(self, repo_url: str) -> str:
        try:
            if os.listdir(self.workdir):
                return (
                    f"Workspace already has files at {self.workdir}; "
                    "skipping clone"
                )
            result = subprocess.run(
                ["git", "clone", repo_url, "."],
                cwd=self.workdir,
                capture_output=True,
                text=True,
                check=False,
            )
            if result.returncode != 0:
                return f"Clone failed: {result.stderr}"
            return "Cloned successfully"
        except Exception as e:
            return f"Error cloning '{repo_url}': {e}"

    def list_dir(self, path: str) -> str:
        try:
            full = self._resolve(path)
            entries = sorted(os.listdir(full))
            if not entries:
                return "(empty directory)"
            return "\n".join(entries)
        except Exception as e:
            return f"Error listing '{path}': {e}"


def _find_literal_matches(text: str, needle: str) -> list[int]:
    """1-based line numbers of every non-overlapping occurrence of needle."""
    lines: list[int] = []
    start = 0
    step = max(len(needle), 1)
    while True:
        idx = text.find(needle, start)
        if idx < 0:
            break
        lines.append(text.count("\n", 0, idx) + 1)
        start = idx + step
    return lines


class _DiffError(ValueError):
    """Loud, non-guessing failure while applying a unified diff."""


def _parse_unified_diff(diff_text: str) -> list[dict]:
    """Parse ---/+++ headers (ignored) and @@ hunks. Fails on malformed input."""
    raw_lines = diff_text.splitlines()
    hunks: list[dict] = []
    i = 0

    def skippable(line: str) -> bool:
        prefixes = (
            "diff --git",
            "index ",
            "new file mode",
            "deleted file mode",
            "old mode",
            "new mode",
            "similarity index",
            "rename from",
            "rename to",
            "--- ",
            "+++ ",
        )
        return line.startswith(prefixes) or line in ("---", "+++")

    while i < len(raw_lines):
        line = raw_lines[i]
        if skippable(line) or line.strip() == "":
            i += 1
            continue
        if not line.startswith("@@"):
            raise _DiffError(
                f"expected a hunk header (@@), got: {line!r}."
            )

        m = HUNK_HEADER_RE.match(line)
        if not m:
            raise _DiffError(f"malformed hunk header: {line!r}.")

        old_start = int(m.group(1))
        old_count = int(m.group(2) if m.group(2) is not None else "1")
        new_count = int(m.group(4) if m.group(4) is not None else "1")

        i += 1
        old_lines: list[str] = []
        new_lines: list[str] = []
        while i < len(raw_lines):
            body = raw_lines[i]
            if body.startswith("@@") or skippable(body):
                break
            if body.startswith("\\"):
                # "\ No newline at end of file" — we ignore it and join
                # with the original file's newline convention later.
                i += 1
                continue
            if body == "":
                # A blank hunk line is an empty context line whose leading
                # space got stripped. Treat it as context, not end-of-hunk.
                prefix, rest = " ", ""
            else:
                prefix, rest = body[0], body[1:]
                if prefix not in (" ", "+", "-", "\\"):
                    raise _DiffError(
                        f"hunk line must start with ' ', '+', or '-', got: {body!r}."
                    )
            if prefix in (" ", "-"):
                old_lines.append(rest)
            if prefix in (" ", "+"):
                new_lines.append(rest)
            i += 1

        if len(old_lines) != old_count:
            raise _DiffError(
                f"hunk at old line {old_start} declared {old_count} old "
                f"line(s) but the body has {len(old_lines)}."
            )
        if len(new_lines) != new_count:
            raise _DiffError(
                f"hunk at old line {old_start} declared {new_count} new "
                f"line(s) but the body has {len(new_lines)}."
            )
        hunks.append(
            {
                "old_start": old_start,
                "old_count": old_count,
                "old_lines": old_lines,
                "new_lines": new_lines,
                "header": line,
            }
        )

    return hunks


def _find_subseq(haystack: list[str], needle: list[str]) -> list[int]:
    if not needle:
        return []
    n = len(needle)
    hits = []
    for i in range(len(haystack) - n + 1):
        if haystack[i : i + n] == needle:
            hits.append(i)
    return hits


def _apply_hunks(text: str, hunks: list[dict]) -> tuple[str, int]:
    """Apply hunks bottom-to-top so earlier line numbers stay valid."""
    nl = Sandbox._detect_newline(text)
    ended_with_nl = text.endswith(("\n", "\r"))
    lines = text.splitlines()
    # splitlines() drops a trailing empty line that a final newline implies;
    # we restore that via ended_with_nl when rejoining.

    # Bottom-to-top: a hunk at line 50 doesn't shift a hunk at line 10.
    for hunk in sorted(hunks, key=lambda h: h["old_start"], reverse=True):
        old_lines = hunk["old_lines"]
        new_lines = hunk["new_lines"]
        old_start = hunk["old_start"]
        old_count = hunk["old_count"]

        if old_count == 0:
            # Insertion: old_start is the line AFTER which to insert
            # (0 means the start of the file).
            at = old_start
            if at < 0 or at > len(lines):
                raise _DiffError(
                    f"{hunk['header']} wants to insert after line {old_start} "
                    f"but the file has {len(lines)} line(s)."
                )
            lines[at:at] = new_lines
            continue

        # 1-based old_start → 0-based index.
        at = old_start - 1
        window = lines[at : at + old_count] if 0 <= at <= len(lines) else None
        if window == old_lines:
            lines[at : at + old_count] = new_lines
            continue

        # Specified line didn't match. Search the whole file — but only
        # accept a *unique* hit. Multiple hits = refuse, same as str_replace.
        hits = _find_subseq(lines, old_lines)
        if not hits:
            preview = "\n".join(old_lines[:4])
            extra = " …" if len(old_lines) > 4 else ""
            raise _DiffError(
                f"{hunk['header']} does not match the file. Expected to find:\n"
                f"{preview}{extra}"
            )
        if len(hits) > 1:
            where = ", ".join(str(h + 1) for h in hits)
            raise _DiffError(
                f"{hunk['header']} matches {len(hits)} places (lines {where}). "
                "Add more context lines so the hunk is unique."
            )
        at = hits[0]
        lines[at : at + old_count] = new_lines

    out = nl.join(lines)
    # splitlines() drops the final newline; put it back if the original
    # file had one. Always append — out may already end in nl because of
    # a trailing blank line in `lines`, which is not the same thing.
    if ended_with_nl:
        out += nl
    return out, len(hunks)


def execute_tool(sandbox: Sandbox, name: str, tool_input: dict) -> str:
    """Name -> function dispatch. This is the whole 'engine'."""
    if name == "read_file":
        return sandbox.read_file(
            tool_input["path"],
            tool_input.get("start_line"),
            tool_input.get("end_line"),
        )
    elif name == "write_file":
        return sandbox.write_file(tool_input["path"], tool_input["content"])
    elif name == "str_replace":
        return sandbox.str_replace(
            tool_input["path"],
            tool_input["old_string"],
            tool_input["new_string"],
            bool(tool_input.get("replace_all", False)),
        )
    elif name == "replace_lines":
        return sandbox.replace_lines(
            tool_input["path"],
            tool_input["start_line"],
            tool_input["end_line"],
            tool_input["content"],
        )
    elif name == "apply_diff":
        return sandbox.apply_diff(tool_input["path"], tool_input["diff"])
    elif name == "list_dir":
        return sandbox.list_dir(tool_input["path"])
    elif name == "clone_repo":
        return sandbox.clone_repo(tool_input["repo_url"])
    else:
        return f"Error: unknown tool '{name}'"


# ---------------------------------------------------------------------------
# 3. THE AGENT LOOP
#
# Same shape as ex_3: one tool-use inner loop per user message, then control
# returns to a [you]: prompt. Conversation history stays on `self.messages`.
# ---------------------------------------------------------------------------

class AgentLoop:
    SYSTEM_PROMPT = """You are a careful coding agent working inside a sandboxed
project directory. You have tools to clone a repo, read files, write files,
and list directories. You are in a multi-turn conversation: after you answer,
control returns to the user for the next question. Do not try to end a session.

How to start:
- clone_repo(repo_url): if the working directory is empty (or the task
  names a repository), clone it first, then list_dir / read_file. Do not
  clone again if files are already there.

How to edit (pick the tightest tool that fits):
- str_replace: default for small, surgical edits. old_string must match
  EXACTLY once. If the tool says it matched N times, add more surrounding
  context and retry — do not guess which occurrence to change.
- replace_lines: when you already have a line window open from read_file
  (especially large files). Address that same start_line/end_line range;
  do not rewrite the whole file.
- apply_diff: larger structural changes, several hunks in one file.
  Unified diffs only; hunks must match uniquely. No fuzzy patching.
- write_file: new files, or a full rewrite of a small file. Never dump a
  whole large file just to change a few lines.

Approach every task like this:
1. Clone if needed, then explore: list_dir / read_file to understand the
   relevant code before changing anything. Don't guess at file contents.
2. Make the smallest change that correctly solves the task.
3. After editing, re-read the changed region to confirm it looks right.
4. If an edit tool fails, read the error (zero matches, multiple matches,
   bad line range, hunk mismatch) and retry with a tighter anchor. Do not
   fall back to overwriting the whole file unless the file is small.
5. When you're done with the current question, reply with plain text (no
   tool call) summarizing what you found or changed. The user will ask
   the next thing.
"""

    def __init__(self, repo: str, workdir: str, max_turns: int = MAX_TURNS):
        self.repo = repo
        self.max_turns = max_turns
        self.client = Anthropic()  # reads ANTHROPIC_API_KEY from env
        self.sandbox = Sandbox(workdir)
        self.messages = []

    def _run_agent_turn(self):
        """Tool-use turns until the model gives a final text answer."""
        for _ in range(1, self.max_turns + 1):
            response = self.client.messages.create(
                model=MODEL,
                max_tokens=4096,
                system=self.SYSTEM_PROMPT,
                messages=self.messages,
                tools=TOOLS,
            )

            for block in response.content:
                if block.type == "text" and block.text.strip():
                    print(f"\n{BLUE}[assistant]: {block.text}{RESET}")

            self.messages.append({"role": "assistant", "content": response.content})

            if response.stop_reason != "tool_use":
                return

            tool_results = []
            for block in response.content:
                if block.type != "tool_use":
                    continue
                print(f"{RED}[tool_call]: {block.name}({block.input}){RESET}")
                result = execute_tool(self.sandbox, block.name, block.input)
                preview = result if len(result) < 500 else result[:500] + "... [truncated]"
                print(f"{RED}[tool_result]: {preview}{RESET}")
                tool_results.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": result,
                    }
                )
            self.messages.append({"role": "user", "content": tool_results})

        print("\n[warning] hit max_turns without a final response")

    def start_agent(self):
        print(f"=== Workspace: {self.sandbox.workdir} ===")
        print("=== Bootstrapping: cloning and exploring repo ===")
        bootstrap = (
            f"Clone the repository at {self.repo}, then run an initial exploration "
            f"pass with list_dir (recurse into a few key subdirectories if the "
            f"top level looks like a monorepo or has an obvious src/ layout). "
            f"Briefly summarize what kind of project this looks like once done."
        )
        self.messages.append({"role": "user", "content": bootstrap})
        self._run_agent_turn()

        print("\n=== Repo ready. Ask questions about it (type 'exit' to quit) ===")
        while True:
            try:
                user_input = input(f"\n{GREEN}[you]: ").strip()
                print(RESET, end="")
            except (EOFError, KeyboardInterrupt):
                print(RESET)
                break
            if user_input.lower() in ("exit", "quit"):
                break
            if not user_input:
                continue

            self.messages.append({"role": "user", "content": user_input})
            self._run_agent_turn()


# ---------------------------------------------------------------------------
# 4. CLI ENTRY POINT
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Minimal CLI coding agent")
    parser.add_argument("--repo", type=str, required=True, help="Git URL to clone into the workspace")
    parser.add_argument(
        "--workdir",
        default="./ex_5_workspace",
        help="Directory the agent is confined to (default: ./ex_5_workspace)",
    )
    parser.add_argument("--max-turns", type=int, default=MAX_TURNS)
    args = parser.parse_args()

    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("Error: set ANTHROPIC_API_KEY in your environment first.", file=sys.stderr)
        sys.exit(1)

    AgentLoop(args.repo, args.workdir, args.max_turns).start_agent()


if __name__ == "__main__":
    main()
