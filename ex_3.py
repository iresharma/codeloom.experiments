# An experiment to see if we can use tree-sitter and LSP to find the exact definition of a function, class, or method by name in a specific file, returning just that symbol's source code.

from __future__ import annotations

import argparse
import fnmatch
import json
import os
import queue
import shutil
import subprocess
import threading
import time
import uuid
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import ClassVar
from urllib.parse import unquote, urlparse

import tree_sitter_go as tsgo
import tree_sitter_javascript as tsjs
import tree_sitter_python as tspython
import tree_sitter_typescript as tsts
from anthropic import Anthropic
from tree_sitter import Language, Parser

MODEL = "claude-sonnet-4-6"
MAX_TURNS = 25          # hard cap so a buggy loop can't run forever
SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_PROFILE_LOG = SCRIPT_DIR / "ex_3_profile.jsonl"


def resolve_log_path(path: Path) -> Path:
    """Relative --profile-log paths go next to this script, not cwd."""
    path = Path(path)
    if not path.is_absolute():
        path = SCRIPT_DIR / path
    path.parent.mkdir(parents=True, exist_ok=True)
    return path.resolve()


# Terminal colors: agent=blue, user=green, tool=red
BLUE = "\033[94m"
GREEN = "\033[92m"
RED = "\033[91m"
RESET = "\033[0m"


class ProfileLog:
    """Append-only JSONL profiler. Console stays clean; graphs read this file."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self.session_id = (
            datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            + "-"
            + uuid.uuid4().hex[:8]
        )
        self._seq = 0
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.event("session_start", model=MODEL)

    def event(self, event: str, **fields):
        self._seq += 1
        record = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "session_id": self.session_id,
            "seq": self._seq,
            "event": event,
            **fields,
        }
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, default=str) + "\n")


PY_LANGUAGE = Language(tspython.language())
python_parser = Parser(PY_LANGUAGE)

JS_LANGUAGE = Language(tsjs.language())
js_parser = Parser(JS_LANGUAGE)

TS_LANGUAGE = Language(tsts.language_typescript())
ts_parser = Parser(TS_LANGUAGE)

TSX_LANGUAGE = Language(tsts.language_tsx())
tsx_parser = Parser(TSX_LANGUAGE)

GO_LANGUAGE = Language(tsgo.language())
go_parser = Parser(GO_LANGUAGE)

LANGUAGE_PARSERS = {
    "python": python_parser,
    "javascript": js_parser,
    "typescript": ts_parser,
    "typescript_jsx": tsx_parser,
    "go": go_parser,
}

LANGUAGE_SYMBOL_TYPES = {
    "python": {"function_definition", "class_definition"},
    "javascript": {"function_declaration", "class_declaration", "method_definition", "variable_declarator"},
    "typescript": {"function_declaration", "class_declaration", "method_definition", "variable_declarator", "interface_declaration"},
    "typescript_jsx": {"function_declaration", "class_declaration", "method_definition", "variable_declarator", "interface_declaration"},
    "go": {"function_declaration", "method_declaration", "type_declaration"},
}

# Manifest files are a strong, cheap signal — checked first.
MANIFEST_MARKERS = {
    "pyproject.toml": "python",
    "setup.py": "python",
    "requirements.txt": "python",
    "Pipfile": "python",
    "go.mod": "go",
    "go.sum": "go",
    "package.json": "javascript",  # refined to typescript below if tsconfig exists
    "tsconfig.json": "typescript",
    "tsconfig.app.json": "typescript",
}

# Extension fallback — used when no manifest is found, or to detect
# secondary languages in a monorepo.
EXTENSION_MAP = {
    ".py": "python",
    ".go": "go",
    ".js": "javascript",
    ".jsx": "javascript",
    ".ts": "typescript",
    ".tsx": "typescript",
}

# Directories that would skew extension counts if walked into.
IGNORE_DIRS = {
    ".git", ".venv", "venv", "node_modules", "__pycache__",
    "dist", "build", ".mypy_cache", ".pytest_cache", "target",
    ".next", ".turbo", "vendor",
}

# Cap background didOpen so a huge monorepo can't OOM the language server.
INDEX_FILE_CAP = 500

# Caps on a single read_file result, so one vendored 8k-line file can't
# blow out the context window. The model is told when truncation happened
# and which line to resume from, so it can page through deliberately.
MAX_READ_LINES = 2000
MAX_READ_BYTES = 250_000
# Files with a NUL byte in the first block are binary; reading them yields
# replacement-character garbage that costs tokens and teaches nothing.
BINARY_SNIFF_BYTES = 8192

# Search caps. A broad pattern on a big repo can match thousands of lines;
# the model only needs enough to pick where to look next, and is told the
# true total so it knows to narrow rather than assuming it saw everything.
GREP_MAX_MATCHES = 60
GREP_MAX_LINE_CHARS = 300
FIND_FILES_MAX_RESULTS = 100
SEARCH_TIMEOUT_S = 30

TOOLS = [
    {
        "name": "read_file",
        "description": (
            "Read a file at the given path, relative to the working "
            "directory. Returns the contents with a 1-based line-number "
            "gutter ('   42\\tsource'), preceded by a header giving the line "
            "range shown and the file's total line count. Line numbers are "
            "true file positions even when reading a slice, so they can be "
            "passed straight to the LSP tools. Long files are truncated at a "
            "cap, with a note giving the start_line to resume from. Returns "
            "an error message string if the file does not exist or can't be "
            "read."
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
                    "description": "The line number to start reading from",
                },
                "end_line": {
                    "type": "integer",
                    "description": "The line number to end reading at",
                },
            },
            "required": ["path"],
        },
    },
    {
        "name": "grep_search",
        "description": (
            "Search file *contents* across the repository for a regular "
            "expression, like ripgrep. Returns matches grouped by file as "
            "'line: content', so results can be fed straight into read_file "
            "or the LSP tools. This is the right first move when you don't "
            "yet know which file something lives in — much cheaper than "
            "sweeping with list_dir and read_file. Automatically skips "
            ".gitignore'd paths, binary files, and vendor/build directories. "
            "Results are capped; the reply says how many total matches there "
            "were, so narrow the pattern or path if you're near the cap."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "pattern": {
                    "type": "string",
                    "description": (
                        "Regular expression to search for, e.g. "
                        "'def retry_policy' or 'class \\\\w+Client'. Set "
                        "fixed_string if you want it treated as literal text."
                    ),
                },
                "path": {
                    "type": "string",
                    "description": (
                        "Optional relative directory or file to search under, "
                        "e.g. 'src' or 'src/api/client.py'. Defaults to the "
                        "whole workspace."
                    ),
                },
                "glob": {
                    "type": "string",
                    "description": (
                        "Optional filename filter, e.g. '*.py' or "
                        "'**/test_*.go'. Prefix with '!' to exclude."
                    ),
                },
                "case_insensitive": {
                    "type": "boolean",
                    "description": "Match without regard to case. Default false.",
                },
                "fixed_string": {
                    "type": "boolean",
                    "description": (
                        "Treat the pattern as a literal string rather than a "
                        "regex. Use this for things containing regex "
                        "metacharacters, e.g. 'config.get('. Default false."
                    ),
                },
                "context_lines": {
                    "type": "integer",
                    "description": (
                        "Lines of surrounding context to show around each "
                        "match, 0-5. Default 0."
                    ),
                },
            },
            "required": ["pattern"],
        },
    },
    {
        "name": "find_files",
        "description": (
            "Find files by *name or path pattern* across the repository, e.g. "
            "'**/*_test.py' or 'config*'. Use this when you know roughly what "
            "a file is called but not where it lives — it replaces walking "
            "the tree with repeated list_dir calls. Use grep_search instead "
            "when you're looking for something inside file contents."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "glob": {
                    "type": "string",
                    "description": (
                        "Glob to match against the relative path, e.g. "
                        "'**/*.go', 'test_*.py', 'src/**/index.ts'."
                    ),
                },
                "path": {
                    "type": "string",
                    "description": "Optional relative directory to search under.",
                },
            },
            "required": ["glob"],
        },
    },
    {
        "name": "list_dir",
        "description": "List files and directories at the given relative path (non-recursive).",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Relative directory path, e.g. '.' or 'src'",
                }
            },
            "required": ["path"],
        },
    },
    {
        "name": "clone_repo",
        "description": ("Clone a repository into the working directory."),
        "input_schema": {
            "type": "object",
            "properties": {
                "repo_url": {
                    "type": "string",
                    "description": "the remote from where the git repo needs to be cloned",
                }
            },
            "required": ["repo_url"],
        },
    },
    {
        "name": "find_symbol",
        "description": (
            "uses tree-sitter to find exact symbols like functions and "
            "variables in the code. Returns the symbol's source code "
            "together with its 1-based line/character position, so the "
            "result can be chained directly into goto_definition, "
            "find_references, or hover without a separate lookup."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "the path to the file to find the symbol in",
                },
                "symbol": {
                    "type": "string",
                    "description": "the symbol to find in the code",
                },
                "language": {
                    "type": "string",
                    "enum": ["python", "go", "javascript", "typescript"],
                    "description": "the language of the file, used to select the correct tree-sitter grammar",
                },
            },
            "required": ["path", "symbol", "language"],
        },
    },
    {
        "name": "goto_definition",
        "description": (
            "Uses the language server (LSP) to jump to the definition of the "
            "symbol at a specific position in a file. More precise than "
            "find_symbol for resolving imports, usages, or symbols defined in "
            "a different file — it understands scoping and type information, "
            "not just name matching within a single file. Returns the file, "
            "line, and column of each definition location found, plus a "
            "preview of that source line. Only works for files whose language "
            "has a configured server (python, go, javascript, typescript)."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Relative path to the file, e.g. 'src/utils.py'",
                },
                "line": {
                    "type": "integer",
                    "description": (
                        "1-based line number containing the symbol, "
                        "as you'd see it in an editor"
                    ),
                },
                "character": {
                    "type": "integer",
                    "description": (
                        "1-based column number of, or within, the symbol's "
                        "name on that line"
                    ),
                },
            },
            "required": ["path", "line", "character"],
        },
    },
    {
        "name": "find_references",
        "description": (
            "Uses the language server (LSP) to find every usage of the "
            "symbol at a specific position in a file, across the whole "
            "indexed workspace, including the declaration itself. Use this "
            "before renaming or changing a function/class/variable to see "
            "everywhere it's used. Only works for files whose language has a "
            "configured server (python, go, javascript, typescript)."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Relative path to the file, e.g. 'src/utils.py'",
                },
                "line": {
                    "type": "integer",
                    "description": (
                        "1-based line number containing the symbol, "
                        "as you'd see it in an editor"
                    ),
                },
                "character": {
                    "type": "integer",
                    "description": (
                        "1-based column number of, or within, the symbol's "
                        "name on that line"
                    ),
                },
            },
            "required": ["path", "line", "character"],
        },
    },
    {
        "name": "hover",
        "description": (
            "Uses the language server (LSP) to get type information and "
            "documentation for the symbol at a specific position in a file, "
            "similar to hovering over it in an editor. Useful for checking a "
            "function's signature, a variable's inferred type, or a "
            "docstring without reading the full definition. Only works for "
            "files whose language has a configured server (python, go, "
            "javascript, typescript)."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Relative path to the file, e.g. 'src/utils.py'",
                },
                "line": {
                    "type": "integer",
                    "description": (
                        "1-based line number containing the symbol, "
                        "as you'd see it in an editor"
                    ),
                },
                "character": {
                    "type": "integer",
                    "description": (
                        "1-based column number of, or within, the symbol's "
                        "name on that line"
                    ),
                },
            },
            "required": ["path", "line", "character"],
        },
    },
    {
        "name": "get_diagnostics",
        "description": (
            "Uses the language server (LSP) to return compiler/type-checker "
            "diagnostics (errors, warnings, hints) for a file — the same "
            "information shown as red/yellow squiggles in an editor. Opens "
            "the file with the language server if it isn't already open. "
            "Useful for checking whether a file has type errors or lint "
            "issues without running a separate build. Only works for files "
            "whose language has a configured server (python, go, javascript, "
            "typescript)."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Relative path to the file, e.g. 'src/utils.py'",
                },
            },
            "required": ["path"],
        },
    },
]


# ============================================================================
# LSP client — JSON-RPC over stdio, per-language server configs, and a
# manager that spawns/queries the right server based on file extension.
# ============================================================================

class LSPTimeoutError(RuntimeError):
    pass


class LSPClient:
    def __init__(self, cmd: list[str], cwd: str, env: dict | None = None):
        self.proc = subprocess.Popen(
            cmd,
            cwd=cwd,
            env=env,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        self._pending: dict[int, queue.Queue] = {}
        self._notifications: queue.Queue[dict] = queue.Queue()
        self._next_id = 1
        self._id_lock = threading.Lock()
        self._write_lock = threading.Lock()
        self._alive = True

        threading.Thread(
            target=self._reader_loop, daemon=True, name=f"lsp-reader-{cmd[0]}"
        ).start()
        threading.Thread(
            target=self._stderr_drain, daemon=True, name=f"lsp-stderr-{cmd[0]}"
        ).start()

    # ---------- wire framing ----------

    @staticmethod
    def _read_message(stream) -> dict | None:
        headers = {}
        while True:
            line = stream.readline()
            if not line:
                return None  # stream closed
            if line in (b"\r\n", b"\n", b""):
                break
            if b":" in line:
                key, _, value = line.decode("ascii", "replace").partition(":")
                headers[key.strip().lower()] = value.strip()
        length = int(headers.get("content-length", 0))
        body = b""
        while len(body) < length:
            chunk = stream.read(length - len(body))
            if not chunk:
                return None
            body += chunk
        return json.loads(body.decode("utf-8"))

    def _write_message(self, payload: dict):
        body = json.dumps(payload).encode("utf-8")
        header = f"Content-Length: {len(body)}\r\n\r\n".encode("ascii")
        try:
            with self._write_lock:
                self.proc.stdin.write(header + body)
                self.proc.stdin.flush()
        except (BrokenPipeError, OSError) as e:
            raise RuntimeError(f"LSP server process died: {e}")

    # ---------- background loops ----------

    def _reader_loop(self):
        while self._alive:
            try:
                msg = self._read_message(self.proc.stdout)
            except (OSError, ValueError, json.JSONDecodeError):
                break
            if msg is None:
                break
            if "id" in msg and ("result" in msg or "error" in msg):
                q = self._pending.pop(msg["id"], None)
                if q is not None:
                    q.put(msg)
            elif "method" in msg:
                # server->client request or notification (e.g. publishDiagnostics,
                # or window/workDoneProgress/create which expects a response)
                self._notifications.put(msg)
        self._alive = False

    def _stderr_drain(self):
        # must drain or the pipe fills and blocks the server
        for _ in iter(self.proc.stderr.readline, b""):
            pass

    # ---------- public API ----------

    def request(self, method: str, params: dict, timeout: float = 15.0) -> dict:
        if not self._alive:
            raise RuntimeError("LSP server is not running")
        with self._id_lock:
            msg_id = self._next_id
            self._next_id += 1
        q: queue.Queue[dict] = queue.Queue()
        self._pending[msg_id] = q
        self._write_message(
            {"jsonrpc": "2.0", "id": msg_id, "method": method, "params": params}
        )
        try:
            response = q.get(timeout=timeout)
        except queue.Empty:
            self._pending.pop(msg_id, None)
            raise LSPTimeoutError(f"{method} timed out after {timeout}s")
        if "error" in response:
            raise RuntimeError(f"{method} error: {response['error']}")
        return response.get("result")

    def notify(self, method: str, params: dict):
        self._write_message({"jsonrpc": "2.0", "method": method, "params": params})

    def reply(self, msg_id, result=None):
        """Respond to a server->client request (workspace/configuration, etc.)."""
        self._write_message({"jsonrpc": "2.0", "id": msg_id, "result": result})

    def get_notification(self, timeout: float = 0.0) -> dict | None:
        """Non-blocking-by-default pop from the notification queue."""
        try:
            return self._notifications.get(timeout=timeout)
        except queue.Empty:
            return None

    def shutdown(self):
        if not self._alive:
            return
        try:
            self.request("shutdown", {}, timeout=5.0)
            self.notify("exit", {})
        except (LSPTimeoutError, RuntimeError, OSError, BrokenPipeError):
            self._alive = False
        else:
            self._alive = False
        try:
            self.proc.terminate()
            self.proc.wait(timeout=5.0)
        except (OSError, subprocess.TimeoutExpired):
            self.proc.kill()


class LSPManager:
    @dataclass
    class ServerConfig:
        language: str                 # descriptive only, matches EXTENSION_MAP values
        cmd: list[str]                 # subprocess argv
        language_id: str               # LSP "languageId" for didOpen
        extensions: tuple[str, ...]    # which files this server should be asked about
        init_options: dict = field(default_factory=dict)   # merged into initialize params
        # Client settings tree. Pyright reads diagnosticMode from
        # workspace/configuration ("python.analysis") or from
        # workspace/didChangeConfiguration — NOT from init_options.
        settings: dict = field(default_factory=dict)

    # Shared across every LSPManager instance; read by _config_for_extension.
    SERVER_CONFIGS: ClassVar[dict[str, ServerConfig]] = {
        "python": ServerConfig(
            language="python",
            # npx avoids requiring a global npm install as a setup step
            cmd=["npx", "-y", "-p", "pyright", "pyright-langserver", "--stdio"],
            language_id="python",
            extensions=(".py", ".pyi"),
            settings={
                "python": {
                    "analysis": {
                        "diagnosticMode": "workspace",
                        "autoSearchPaths": True,
                        "useLibraryCodeForTypes": True,
                    }
                }
            },
        ),
        "typescript": ServerConfig(
            language="typescript",
            cmd=["npx", "-y", "typescript-language-server", "--stdio"],
            language_id="typescript",
            extensions=(".ts", ".tsx"),
            # typescript-language-server has no equivalent "diagnose the
            # whole workspace without didOpen" switch — it only ever
            # diagnoses files it's been told are open. Warm-starting still
            # gets the process/tsserver spun up and the project graph
            # loaded ahead of time, which is most of the latency win.
        ),
        "javascript": ServerConfig(
            language="javascript",
            # same binary handles JS; languageId differs per file
            cmd=["npx", "-y", "typescript-language-server", "--stdio"],
            language_id="javascript",
            extensions=(".js", ".jsx"),
        ),
        "go": ServerConfig(
            language="go",
            # gopls has no npx equivalent — must be installed separately:
            # go install golang.org/x/tools/gopls@latest
            # gopls analyzes the whole workspace as part of loading it, so
            # a warm-started server naturally gives workspace-wide coverage
            # without needing an explicit diagnosticMode toggle.
            cmd=["gopls", "serve"],
            language_id="go",
            extensions=(".go",),
        ),
    }

    VALID_ACTIONS: ClassVar[dict[str, str]] = {
        "references": "textDocument/references",
        "definition": "textDocument/definition",
        "hover": "textDocument/hover",
    }

    def __init__(self, workspace_root: str):
        self.root = os.path.abspath(workspace_root)
        self.root_uri = self._path_to_uri(self.root)
        self._clients: dict[tuple, LSPClient] = {}   # keyed by tuple(cmd)
        self._opened_files: set[str] = set()          # absolute paths already didOpen'd
        self._diagnostics: dict[str, list] = {}        # uri -> diagnostics list (cache)
        self._lock = threading.Lock()

    @staticmethod
    def _path_to_uri(path: str) -> str:
        return Path(path).resolve().as_uri()

    @classmethod
    def _config_for_extension(cls, ext: str) -> ServerConfig | None:
        for cfg in cls.SERVER_CONFIGS.values():
            if ext in cfg.extensions:
                return cfg
        return None

    # ---------- lifecycle ----------

    def _client_for(self, cfg: ServerConfig) -> LSPClient:
        key = tuple(cfg.cmd)
        with self._lock:
            client = self._clients.get(key)
            if client is not None:
                return client

        client = LSPClient(cfg.cmd, cwd=self.root)
        client.request(
            "initialize",
            {
                "processId": os.getpid(),
                "rootUri": self.root_uri,
                "rootPath": self.root,
                "workspaceFolders": [
                    {"uri": self.root_uri, "name": Path(self.root).name}
                ],
                "capabilities": {
                    "workspace": {
                        "workspaceFolders": True,
                        "configuration": True,
                        "didChangeConfiguration": {"dynamicRegistration": False},
                    },
                    "window": {"workDoneProgress": True},
                    "textDocument": {
                        "synchronization": {"didSave": True},
                        "publishDiagnostics": {"relatedInformation": True},
                        "definition": {},
                        "references": {},
                        "hover": {},
                    },
                },
                "initializationOptions": cfg.init_options,
            },
            timeout=30.0,  # some servers are slow to boot the first time
        )

        # Listener must be running before `initialized` — pyright immediately
        # sends workspace/configuration and waits for the reply before it
        # applies diagnosticMode=workspace.
        threading.Thread(
            target=self._notification_listener,
            args=(client, cfg),
            daemon=True,
            name=f"lsp-notify-{cfg.language}",
        ).start()

        client.notify("initialized", {})
        if cfg.settings:
            client.notify("workspace/didChangeConfiguration", {"settings": cfg.settings})

        with self._lock:
            existing = self._clients.get(key)
            if existing is not None:
                client.shutdown()
                return existing
            self._clients[key] = client

        return client

    def _notification_listener(self, client: LSPClient, cfg: ServerConfig):
        """Handle server->client requests and cache publishDiagnostics."""
        while client._alive:
            note = client.get_notification(timeout=0.5)
            if note is None:
                continue
            if "id" in note and "method" in note:
                self._handle_server_request(client, cfg, note)
                continue
            if note.get("method") == "textDocument/publishDiagnostics":
                params = note.get("params", {})
                uri = params.get("uri", "<unknown>")
                with self._lock:
                    self._diagnostics[uri] = params.get("diagnostics", [])

    @staticmethod
    def _settings_section(settings: dict, section: str | None):
        if not section:
            return settings
        node = settings
        for part in section.split("."):
            if not isinstance(node, dict) or part not in node:
                return None
            node = node[part]
        return node

    def _handle_server_request(self, client: LSPClient, cfg: ServerConfig, note: dict):
        method = note.get("method")
        msg_id = note["id"]
        params = note.get("params") or {}
        if method == "workspace/configuration":
            items = params.get("items") or [{}]
            result = [
                self._settings_section(cfg.settings, item.get("section"))
                for item in items
            ]
            client.reply(msg_id, result)
        elif method == "workspace/workspaceFolders":
            client.reply(
                msg_id,
                [{"uri": self.root_uri, "name": Path(self.root).name}],
            )
        else:
            client.reply(msg_id, None)

    def shutdown_all(self):
        for client in self._clients.values():
            client.shutdown()
        self._clients.clear()
        self._opened_files.clear()

    def warm_start(self, language: str) -> int | None:
        """
        Spawn + initialize the server, push workspace settings, then didOpen
        source files so analysis/indexing overlaps the agent's explore turns.
        Returns the number of files opened, or None if no server is configured.
        """
        cfg = self.SERVER_CONFIGS.get(language)
        if cfg is None:
            return None
        self._client_for(cfg)
        n = self.index_workspace(language)
        self._wait_for_diagnostics(n)
        return n

    def _wait_for_diagnostics(self, opened: int, timeout: float = 20.0):
        """Block only the warm-start thread until diagnostics arrive or timeout."""
        if opened == 0:
            return
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            with self._lock:
                n = len(self._diagnostics)
            if n >= opened:
                break
            time.sleep(0.2)

    def _iter_source_files(self, extensions: tuple[str, ...], max_files: int):
        count = 0
        for root, dirs, files in os.walk(self.root):
            dirs[:] = [d for d in dirs if d not in IGNORE_DIRS and not d.startswith(".")]
            for fname in files:
                if Path(fname).suffix not in extensions:
                    continue
                full = os.path.join(root, fname)
                rel = os.path.relpath(full, self.root)
                if os.sep != "/":
                    rel = rel.replace(os.sep, "/")
                yield rel
                count += 1
                if count >= max_files:
                    return

    def _did_open(self, rel_path: str, cfg: ServerConfig | None = None) -> str | None:
        """Send textDocument/didOpen without waiting for diagnostics. Returns uri."""
        full_path = str(Path(self.root, rel_path).resolve())
        uri = self._path_to_uri(full_path)
        with self._lock:
            if full_path in self._opened_files:
                return uri
        ext = Path(full_path).suffix
        cfg = cfg or self._config_for_extension(ext)
        if cfg is None:
            return None
        client = self._client_for(cfg)
        text = Path(full_path).read_text(errors="replace")
        client.notify(
            "textDocument/didOpen",
            {
                "textDocument": {
                    "uri": uri,
                    "languageId": cfg.language_id,
                    "version": 1,
                    "text": text,
                }
            },
        )
        with self._lock:
            self._opened_files.add(full_path)
        return uri

    def index_workspace(self, language: str, max_files: int = INDEX_FILE_CAP) -> int:
        cfg = self.SERVER_CONFIGS.get(language)
        if cfg is None:
            return 0
        opened = 0
        for rel in self._iter_source_files(cfg.extensions, max_files):
            try:
                if self._did_open(rel, cfg) is not None:
                    opened += 1
            except OSError:
                continue
        return opened

    # ---------- file open + diagnostics ----------

    def open_file_and_get_diagnostics(
        self, rel_path: str, wait_timeout: float = 5.0
    ) -> list:
        """
        didOpen has no response — diagnostics arrive later as an unprompted
        publishDiagnostics notification, picked up by the background
        _notification_listener thread (started in _client_for) and cached
        in self._diagnostics. We send didOpen here, then just poll that
        cache until this file's uri shows up or wait_timeout elapses
        (clean files may never get an explicit notification at all).
        """
        full_path = str(Path(self.root, rel_path).resolve())
        uri = self._path_to_uri(full_path)
        with self._lock:
            already_open = full_path in self._opened_files
        if already_open:
            return self._diagnostics.get(uri, [])

        ext = Path(full_path).suffix
        cfg = self._config_for_extension(ext)
        if cfg is None:
            raise ValueError(f"No LSP server configured for extension '{ext}'")

        self._did_open(rel_path, cfg)

        deadline = time.monotonic() + wait_timeout
        while time.monotonic() < deadline:
            if uri in self._diagnostics:
                return self._diagnostics[uri]
            time.sleep(0.1)

        # no diagnostics arrived in time; treat as clean (common for valid files)
        self._diagnostics.setdefault(uri, [])
        return self._diagnostics[uri]

    # ---------- queries ----------

    def ask_lsp(self, rel_path: str, line: int, character: int, action: str):
        if action not in self.VALID_ACTIONS:
            raise ValueError(f"action must be one of {list(self.VALID_ACTIONS)}")

        full_path = str(Path(self.root, rel_path).resolve())
        if full_path not in self._opened_files:
            self.open_file_and_get_diagnostics(rel_path)

        ext = Path(full_path).suffix
        cfg = self._config_for_extension(ext)
        client = self._client_for(cfg)
        uri = self._path_to_uri(full_path)

        params = {
            "textDocument": {"uri": uri},
            "position": {"line": line, "character": character},
        }
        if action == "references":
            params["context"] = {"includeDeclaration": True}

        return client.request(self.VALID_ACTIONS[action], params)


class Sandbox:
    def __init__(self):
        self.workdir = os.path.abspath("ex_3_workspace")
        self._create_workspace()

        # Workspace is usually empty until clone_repo runs, so language
        # detection here is best-effort against leftovers from a previous run.
        self.language = self._detect_project_language()
        self.lsp = LSPManager(self.workdir)

        if self.language is not None:
            self._start_lsp_warm_start()

    def _create_workspace(self):
        os.makedirs(self.workdir, exist_ok=True)

    def _start_lsp_warm_start(self):
        # Index in the background so the agent can explore while the server
        # starts. Failures here (missing gopls, etc.) stay non-fatal; ask_lsp
        # will surface an error if a tool later tries to use it.
        print(f"{RED}[lsp] warm-starting '{self.language}'{RESET}")
        threading.Thread(
            target=self._warm_lsp, daemon=True, name="lsp-warm-start"
        ).start()

    def _warm_lsp(self):
        try:
            n = self.lsp.warm_start(self.language)
            if n is None:
                print(
                    f"{RED}[lsp] no server configured for language "
                    f"'{self.language}' — skipping warm start{RESET}"
                )
            else:
                print(
                    f"{RED}[lsp] '{self.language}' indexed {n} file(s){RESET}"
                )
        except Exception as e:
            print(f"{RED}[lsp] warm start for '{self.language}' failed: {e}{RESET}")

    def _detect_project_language(self, max_files: int = 5000):
        path = self.workdir
        manifest_hits = []
        manifest_langs = set()
        ext_counter = Counter()
        file_count = 0

        for root, dirs, files in os.walk(path):
            # prune ignored dirs in-place so os.walk doesn't descend into them
            dirs[:] = [d for d in dirs if d not in IGNORE_DIRS and not d.startswith(".")]

            for fname in files:
                if fname in MANIFEST_MARKERS:
                    manifest_hits.append(fname)
                    manifest_langs.add(MANIFEST_MARKERS[fname])

                ext = os.path.splitext(fname)[1]
                if ext in EXTENSION_MAP:
                    ext_counter[EXTENSION_MAP[ext]] += 1
                    file_count += 1

            if file_count > max_files:
                break  # cap the walk on huge repos; we already have enough signal

        # package.json without tsconfig.json is javascript, not typescript —
        # only refine if we haven't already got a tsconfig hit
        if "package.json" in manifest_hits and "typescript" not in manifest_langs:
            manifest_langs.add("javascript")

        if manifest_langs:
            # rank manifest langs by extension count so primary reflects
            # which one actually dominates the codebase
            primary = max(manifest_langs, key=lambda l: ext_counter.get(l, 0)) \
                if any(ext_counter.get(l, 0) for l in manifest_langs) \
                else next(iter(manifest_langs))
        elif ext_counter:
            primary = ext_counter.most_common(1)[0][0]
        else:
            primary = None

        return primary

    def _resolve(self, rel_path: str) -> str:
        full = os.path.abspath(os.path.join(self.workdir, rel_path))
        try:
            if os.path.commonpath([full, self.workdir]) != self.workdir:
                raise ValueError(f"Path '{rel_path}' escapes the working directory")
        except ValueError:
            raise ValueError(f"Path '{rel_path}' escapes the working directory")
        return full

    def clone_repo(self, repo_url: str):
        result = subprocess.run(
            ["git", "clone", repo_url, "."],
            cwd=self.workdir, capture_output=True, text=True, check=False,
        )
        if result.returncode != 0:
            return f"Clone failed: {result.stderr}"

        # First point the workspace has content, so re-detect language and
        # start the LSP if we couldn't earlier. Non-fatal if detection fails.
        if self.language is None:
            self.language = self._detect_project_language()
            if self.language is not None:
                self._start_lsp_warm_start()

        return "Cloned successfully"

    def list_dir(self, path="."):
        try:
            full = self._resolve(path)
            entries = sorted(os.listdir(full))
            if not entries:
                return "(empty directory)"
            return "\n".join(entries)
        except (OSError, ValueError) as e:
            return f"Error listing '{path}': {e}"

    @staticmethod
    def _looks_binary(full: str) -> bool:
        try:
            with open(full, "rb") as f:
                return b"\x00" in f.read(BINARY_SNIFF_BYTES)
        except OSError:
            return False

    @staticmethod
    def _number_lines(lines: list[str], first_lineno: int) -> str:
        """Render lines as '   123\tsource', numbered from first_lineno.

        The number is a display gutter, NOT part of the file content: a
        `character` position counts columns in the source text after the
        tab, which keeps read_file coordinates consistent with the ones
        find_symbol returns and the LSP tools expect.
        """
        out = []
        for offset, text in enumerate(lines):
            out.append(f"{first_lineno + offset:>6}\t{text.rstrip(chr(10))}")
        return "\n".join(out)

    def read_file(self, path: str, start_line: int | None = None, end_line: int | None = None):
        try:
            full = self._resolve(path)

            if os.path.isdir(full):
                return f"Error reading '{path}': it is a directory — use list_dir instead"
            if self._looks_binary(full):
                size = os.path.getsize(full)
                return (
                    f"'{path}' looks like a binary file ({size} bytes) — not shown. "
                    f"Reading it would return unusable bytes."
                )

            with open(full, "r", errors="replace") as f:
                lines = f.readlines()

            total = len(lines)
            if total == 0:
                return f"'{path}' is empty (0 lines)"

            # Clamp the requested window to the file, and remember where it
            # actually starts so the gutter shows true file line numbers
            # rather than restarting at 1 on every slice.
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
            requested = len(window)

            # Truncate on whichever cap bites first.
            truncated_at = None
            if requested > MAX_READ_LINES:
                window = window[:MAX_READ_LINES]
                truncated_at = start + MAX_READ_LINES
            running = 0
            for i, text in enumerate(window):
                running += len(text.encode("utf-8", errors="replace"))
                if running > MAX_READ_BYTES:
                    window = window[:i or 1]
                    truncated_at = start + len(window)
                    break

            body = self._number_lines(window, start + 1)
            shown_last = start + len(window)
            header = (
                f"{path} — lines {start + 1}-{shown_last} of {total} "
                f"(gutter shows 1-based file line numbers; `character` counts "
                f"columns in the source after the tab)"
            )
            footer = ""
            if truncated_at is not None:
                footer = (
                    f"\n\n[truncated at the read_file cap — {total - truncated_at} "
                    f"more line(s) in this file; continue with "
                    f"start_line={truncated_at + 1}]"
                )
            return f"{header}\n\n{body}{footer}"
        except (OSError, ValueError) as e:
            return f"Error reading '{path}': {e}"

    # ---------- content & filename search ----------
    # ripgrep is the fast path: it honours .gitignore, skips binaries, and
    # emits --json we can parse without guessing at ':' separators in paths.
    # grep/os.walk is a correctness-equivalent fallback so the tool still
    # works on a machine without rg installed, just slower and without
    # .gitignore awareness (IGNORE_DIRS covers the worst of that).

    _rg_path: ClassVar[str | None] = None
    _rg_checked: ClassVar[bool] = False

    @classmethod
    def _ripgrep(cls) -> str | None:
        if not cls._rg_checked:
            cls._rg_path = shutil.which("rg")
            cls._rg_checked = True
        return cls._rg_path

    @staticmethod
    def _rg_text(field, placeholder: str = "") -> str:
        """rg --json gives {'text': ...} normally, {'bytes': b64} for
        non-UTF8 content. For line content, say so rather than emitting a
        blank line that would read as an empty match; for paths, the
        placeholder stays empty so the entry is skipped instead."""
        if not isinstance(field, dict):
            return ""
        if "text" in field:
            return field["text"]
        if "bytes" in field:
            return placeholder
        return ""

    @staticmethod
    def _clip(line: str) -> str:
        line = line.rstrip("\n").rstrip("\r")
        if len(line) > GREP_MAX_LINE_CHARS:
            return line[:GREP_MAX_LINE_CHARS] + f" ... [+{len(line) - GREP_MAX_LINE_CHARS} chars]"
        return line

    def _format_matches(self, hits: list[tuple[str, int, str, bool]], total: int,
                        total_files: int, header: str) -> str:
        """hits: (relpath, lineno, text, is_match). Grouped by file so the
        path isn't repeated on every line."""
        if not hits:
            return f"No matches for {header}"

        out = []
        current = None
        for path, lineno, text, is_match in hits:
            if path != current:
                out.append(f"\n{path}")
                current = path
            sep = ":" if is_match else "-"
            out.append(f"{lineno:>6}{sep} {self._clip(text)}")

        shown = sum(1 for h in hits if h[3])
        summary = f"{total} match(es) in {total_files} file(s) for {header}"
        if total > shown:
            shown_files = len({h[0] for h in hits})
            summary += (
                f" — showing the first {shown} match(es) across "
                f"{shown_files} file(s); narrow the pattern, add a glob, or "
                f"scope with path to see the rest"
            )
        return summary + "\n" + "\n".join(out).lstrip("\n")

    def _grep_via_rg(self, rg, pattern, search_root, glob, case_insensitive,
                     fixed_string, context_lines):
        cmd = [rg, "--json", "--sort", "path"]
        if case_insensitive:
            cmd.append("-i")
        if fixed_string:
            cmd.append("-F")
        if context_lines:
            cmd += ["-C", str(context_lines)]
        if glob:
            cmd += ["-g", glob]
        cmd += ["--", pattern, search_root]

        proc = subprocess.run(
            cmd, cwd=self.workdir, capture_output=True, text=True,
            check=False, timeout=SEARCH_TIMEOUT_S,
        )
        # rg exits 1 for "no matches" (not an error) and 2 for a real
        # failure, e.g. an invalid regex — which the model needs to see.
        if proc.returncode >= 2:
            return None, (proc.stderr.strip() or "ripgrep failed")

        hits, total, files = [], 0, set()
        for raw in proc.stdout.splitlines():
            try:
                evt = json.loads(raw)
            except json.JSONDecodeError:
                continue
            kind = evt.get("type")
            if kind not in ("match", "context"):
                continue
            data = evt["data"]
            path = self._rg_text(data.get("path", {}))
            text = self._rg_text(
                data.get("lines", {}),
                placeholder="[non-UTF8 line — read the file directly to inspect]",
            )
            lineno = data.get("line_number")
            if lineno is None or not path:
                continue
            if kind == "match":
                total += 1
                files.add(path)
            if len([h for h in hits if h[3]]) < GREP_MAX_MATCHES:
                hits.append((path, lineno, text, kind == "match"))
        return (hits, total, len(files)), None

    def _grep_via_grep(self, pattern, search_root, glob, case_insensitive,
                       fixed_string, context_lines):
        cmd = ["grep", "-rn", "--binary-files=without-match"]
        if case_insensitive:
            cmd.append("-i")
        cmd.append("-F" if fixed_string else "-E")
        if context_lines:
            cmd += ["-C", str(context_lines)]
        for d in sorted(IGNORE_DIRS):
            cmd.append(f"--exclude-dir={d}")
        if glob:
            # grep's --include matches the basename only, so a '**/' prefix
            # would never match; strip it and let the directory walk do that
            # part of the job.
            cmd.append(f"--include={glob.lstrip('!').replace('**/', '')}")
        cmd += ["-e", pattern, search_root]

        proc = subprocess.run(
            cmd, cwd=self.workdir, capture_output=True, text=True,
            check=False, timeout=SEARCH_TIMEOUT_S,
        )
        if proc.returncode >= 2 and not proc.stdout:
            return None, (proc.stderr.strip() or "grep failed")

        hits, total, files = [], 0, set()
        for raw in proc.stdout.splitlines():
            # 'path:lineno:text' for matches, 'path-lineno-text' for context.
            # Split on the first separator that yields an integer line number
            # so paths containing '-' don't get mangled.
            parsed = None
            for sep, is_match in ((":", True), ("-", False)):
                head, _, rest = raw.partition(sep)
                num, _, text = rest.partition(sep)
                if num.isdigit():
                    parsed = (head, int(num), text, is_match)
                    break
            if parsed is None:
                continue
            path = os.path.relpath(
                os.path.join(self.workdir, parsed[0]), self.workdir
            )
            if parsed[3]:
                total += 1
                files.add(path)
            if len([h for h in hits if h[3]]) < GREP_MAX_MATCHES:
                hits.append((path, parsed[1], parsed[2], parsed[3]))
        return (hits, total, len(files)), None

    def grep_search(self, pattern: str, path: str = ".", glob: str | None = None,
                    case_insensitive: bool = False, fixed_string: bool = False,
                    context_lines: int = 0):
        try:
            if not pattern:
                return "Error: pattern is required"
            self._resolve(path or ".")   # reject escapes before shelling out
            search_root = path or "."
            context_lines = max(0, min(int(context_lines or 0), 5))

            header = f"pattern '{pattern}'"
            if path not in (None, "", "."):
                header += f" under '{path}'"
            if glob:
                header += f" matching '{glob}'"

            rg = self._ripgrep()
            if rg:
                result, err = self._grep_via_rg(
                    rg, pattern, search_root, glob, case_insensitive,
                    fixed_string, context_lines,
                )
            else:
                result, err = self._grep_via_grep(
                    pattern, search_root, glob, case_insensitive,
                    fixed_string, context_lines,
                )
            if err:
                return f"Error searching for {header}: {err}"
            hits, total, total_files = result
            return self._format_matches(hits, total, total_files, header)
        except subprocess.TimeoutExpired:
            return (
                f"Search for '{pattern}' timed out after {SEARCH_TIMEOUT_S}s — "
                f"narrow it with a path or glob."
            )
        except (OSError, ValueError) as e:
            return f"Error searching for '{pattern}': {e}"

    def find_files(self, glob: str, path: str = "."):
        try:
            if not glob:
                return "Error: glob is required"
            root_full = self._resolve(path or ".")
            if not os.path.isdir(root_full):
                return f"Error: '{path}' is not a directory"

            rg = self._ripgrep()
            if rg:
                proc = subprocess.run(
                    [rg, "--files", "--sort", "path", "-g", glob, "--", path or "."],
                    cwd=self.workdir, capture_output=True, text=True,
                    check=False, timeout=SEARCH_TIMEOUT_S,
                )
                if proc.returncode >= 2:
                    return f"Error finding files: {proc.stderr.strip() or 'ripgrep failed'}"
                matches = [ln for ln in proc.stdout.splitlines() if ln]
            else:
                matches = []
                for root, dirs, files in os.walk(root_full):
                    dirs[:] = [
                        d for d in dirs
                        if d not in IGNORE_DIRS and not d.startswith(".")
                    ]
                    for fname in files:
                        rel = os.path.relpath(
                            os.path.join(root, fname), self.workdir
                        ).replace(os.sep, "/")
                        if fnmatch.fnmatch(rel, glob) or fnmatch.fnmatch(fname, glob):
                            matches.append(rel)
                matches.sort()

            if not matches:
                return f"No files matching '{glob}'" + (
                    f" under '{path}'" if path not in (None, "", ".") else ""
                )
            total = len(matches)
            shown = matches[:FIND_FILES_MAX_RESULTS]
            body = "\n".join(shown)
            if total > len(shown):
                body += (
                    f"\n\n[{total - len(shown)} more file(s) matched — "
                    f"narrow the glob or scope with path]"
                )
            return f"{total} file(s) matching '{glob}':\n{body}"
        except subprocess.TimeoutExpired:
            return f"Finding files matching '{glob}' timed out after {SEARCH_TIMEOUT_S}s"
        except (OSError, ValueError) as e:
            return f"Error finding files matching '{glob}': {e}"

    def _find_named_node(self, root, symbol_types, symbol, source_bytes):
        stack = [root]
        while stack:
            node = stack.pop()
            if node.type in symbol_types:
                name_node = node.child_by_field_name("name")
                if name_node and source_bytes[name_node.start_byte:name_node.end_byte].decode() == symbol:
                    return node
            stack.extend(node.children)
        return None

    def find_symbol(self, path: str, symbol: str, language: str):
        try:
            full = self._resolve(path)
            file_bytes = Path(full).read_bytes()

            parser = LANGUAGE_PARSERS.get(language)
            if parser is None:
                return f"Unsupported language: '{language}'"

            symbol_types = LANGUAGE_SYMBOL_TYPES.get(language, set())
            tree = parser.parse(file_bytes)
            node = self._find_named_node(tree.root_node, symbol_types, symbol, file_bytes)

            if node is None:
                return f"Symbol '{symbol}' not found in '{path}'"

            # Report the position of the *name*, not the start of the whole
            # definition (which may begin with decorators/comments/the def
            # keyword) — that's the position goto_definition/find_references/
            # hover expect, so this output can be chained straight into them.
            name_node = node.child_by_field_name("name") or node
            line = name_node.start_point[0] + 1
            character = name_node.start_point[1] + 1
            source = file_bytes[node.start_byte:node.end_byte].decode()

            return (
                f"Found '{symbol}' at {path}:{line}:{character} "
                f"(pass this line/character to goto_definition, "
                f"find_references, or hover)\n\n{source}"
            )
        except (OSError, ValueError, UnicodeDecodeError) as e:
            return f"Error finding symbol '{symbol}' in '{path}': {e}"

    # ---------- LSP-backed tools ----------
    # Thin formatting layer on top of LSPManager.ask_lsp /
    # open_file_and_get_diagnostics. Tool-facing coordinates are 1-based
    # (line and character) to match what an editor/model would naturally
    # produce; LSP itself is 0-based, so every call here does the -1
    # conversion on the way in and +1 on the way back out.

    @staticmethod
    def _uri_to_path(uri: str) -> str:
        return unquote(urlparse(uri).path)

    def _rel_from_uri(self, uri: str) -> str:
        full = self._uri_to_path(uri)
        try:
            rel = os.path.relpath(full, self.workdir)
        except ValueError:
            return full
        return rel.replace(os.sep, "/") if os.sep != "/" else rel

    @staticmethod
    def _normalize_locations(result) -> list[tuple[str, dict]]:
        """LSP definition/references results can be None, a single Location,
        a Location[], or a LocationLink[] — flatten all of those into a
        uniform list of (uri, range) pairs."""
        if not result:
            return []
        items = [result] if isinstance(result, dict) else result
        locs = []
        for item in items:
            if "targetUri" in item:  # LocationLink
                uri = item["targetUri"]
                rng = item.get("targetSelectionRange") or item["targetRange"]
            else:  # Location
                uri = item["uri"]
                rng = item["range"]
            locs.append((uri, rng))
        return locs

    def _format_locations(self, locs: list[tuple[str, dict]], max_results: int = 20) -> str:
        if not locs:
            return "No results."
        out = []
        for uri, rng in locs[:max_results]:
            rel = self._rel_from_uri(uri)
            start = rng.get("start", {})
            line_no = start.get("line", -1) + 1
            col_no = start.get("character", -1) + 1
            snippet = ""
            try:
                full = self._uri_to_path(uri)
                with open(full, "r", errors="replace") as f:
                    file_lines = f.readlines()
                if 0 <= start.get("line", -1) < len(file_lines):
                    snippet = file_lines[start["line"]].strip()
            except OSError:
                pass
            entry = f"{rel}:{line_no}:{col_no}"
            if snippet:
                entry += f"  {snippet}"
            out.append(entry)
        if len(locs) > max_results:
            out.append(f"... and {len(locs) - max_results} more")
        return "\n".join(out)

    @staticmethod
    def _extract_hover_text(contents) -> str:
        """Hover.contents can be a string, MarkupContent, MarkedString, or
        MarkedString[] depending on the server — normalize to plain text."""
        if contents is None:
            return ""
        if isinstance(contents, str):
            return contents
        if isinstance(contents, dict):
            return contents.get("value", "")
        if isinstance(contents, list):
            parts = [
                item if isinstance(item, str) else item.get("value", "")
                for item in contents
            ]
            return "\n\n".join(p for p in parts if p)
        return ""

    def goto_definition(self, path: str, line: int, character: int):
        try:
            self._resolve(path)  # enforce sandbox boundary before hitting the LSP
            result = self.lsp.ask_lsp(path, line - 1, character - 1, "definition")
            locs = self._normalize_locations(result)
            if not locs:
                return f"No definition found at {path}:{line}:{character}"
            return self._format_locations(locs)
        except (LSPTimeoutError, RuntimeError, ValueError, OSError) as e:
            return f"Error finding definition in '{path}' at {line}:{character}: {e}"

    def find_references(self, path: str, line: int, character: int):
        try:
            self._resolve(path)
            result = self.lsp.ask_lsp(path, line - 1, character - 1, "references")
            locs = self._normalize_locations(result)
            if not locs:
                return f"No references found at {path}:{line}:{character}"
            return self._format_locations(locs, max_results=50)
        except (LSPTimeoutError, RuntimeError, ValueError, OSError) as e:
            return f"Error finding references in '{path}' at {line}:{character}: {e}"

    def hover(self, path: str, line: int, character: int):
        try:
            self._resolve(path)
            result = self.lsp.ask_lsp(path, line - 1, character - 1, "hover")
            text = self._extract_hover_text((result or {}).get("contents"))
            return text.strip() if text and text.strip() else (
                f"No hover information at {path}:{line}:{character}"
            )
        except (LSPTimeoutError, RuntimeError, ValueError, OSError) as e:
            return f"Error getting hover info in '{path}' at {line}:{character}: {e}"

    def get_diagnostics(self, path: str):
        try:
            self._resolve(path)
            diags = self.lsp.open_file_and_get_diagnostics(path)
            if not diags:
                return f"No diagnostics for '{path}' (clean)."
            sev_names = {1: "Error", 2: "Warning", 3: "Info", 4: "Hint"}
            out = []
            for d in diags:
                start = d.get("range", {}).get("start", {})
                line_no = start.get("line", -1) + 1
                col_no = start.get("character", -1) + 1
                sev = sev_names.get(d.get("severity"), "Diagnostic")
                msg = d.get("message", "").strip()
                source = d.get("source")
                prefix = f"[{source}] " if source else ""
                out.append(f"{path}:{line_no}:{col_no} {sev}: {prefix}{msg}")
            return "\n".join(out)
        except (LSPTimeoutError, RuntimeError, ValueError, OSError) as e:
            return f"Error getting diagnostics for '{path}': {e}"


class AgentLoop:
    SYSTEM_PROMPT = """You are a coding agent that explores and analyzes code repositories using a small set of tools. You operate inside a sandboxed working directory and cannot access anything outside it.

Available tools:

- clone_repo(repo_url): clones a git repository into your working directory. Use this first if no repository is present yet.
- list_dir(path): lists files and directories at a given path, non-recursive. Use this to explore structure before reading files — don't guess at paths.
- read_file(path, start_line, end_line): returns the text of a file with a 1-based line-number gutter, formatted as `   42\tsource`, after a header giving the range shown and the file's total line count. The gutter is display only — it is not part of the file, so a `character` position counts columns in the source text after the tab. Line numbers are always true file positions, so a slice read from line 300 is numbered from 300, not from 1. start_line/end_line are optional; omit them to read the whole file, or use them to read a specific slice of a large file instead of the whole thing. Very long reads are truncated at a cap and tell you the start_line to resume from — page through with a follow-up read rather than assuming you saw the end.
- grep_search(pattern, path, glob, case_insensitive, fixed_string, context_lines): searches file *contents* across the repository for a regex and returns matches grouped by file as `line: content`. This is your primary way to find something when you don't already know which file it's in — reach for it before sweeping with list_dir and read_file. Set fixed_string when the pattern contains regex metacharacters you mean literally. Scope with path or glob to keep results tight. Results are capped, and the reply tells you the true total, so if you're at the cap, narrow rather than assuming you've seen everything. Feed a `file:line` hit into read_file (with start_line/end_line around it) or find_symbol to go deeper.
- find_files(glob, path): finds files by name or path pattern, e.g. '**/*_test.py' or 'config*'. Use it when you know roughly what a file is called but not where it lives, instead of walking the tree with repeated list_dir calls.
- find_symbol(path, symbol, language): uses tree-sitter to locate the exact definition of a function, class, or method by name in a specific file, returning that symbol's source code plus its 1-based line/character position. Use this instead of read_file when you already know the name of what you're looking for and want its precise definition rather than the whole file. `language` must be one of: python, go, javascript, typescript. The position it returns can be passed directly into goto_definition, find_references, or hover — no need to re-derive coordinates with read_file.
- goto_definition(path, line, character): uses the project's language server (LSP) to jump to where the symbol at a given position is actually defined, even if that's in a different file. Unlike find_symbol, this understands imports, scoping, and types, not just name-matching in one file — use it to resolve "where does this imported/called thing actually come from". `line` and `character` are 1-based, as in an editor; `character` can point anywhere inside the symbol's name.
- find_references(path, line, character): uses the language server to find every place the symbol at a given position is used across the whole indexed workspace, including its declaration. Use this before explaining the impact of changing something, or before a rename, to see everywhere it's used. Same 1-based line/character convention as goto_definition. Typical flow: find_symbol to locate a definition by name and get its position, then find_references on that position to see every usage.
- hover(path, line, character): uses the language server to get the type signature and any docstring/documentation for the symbol at a given position, without pulling in its full body. Good for a quick check of a function's signature or a variable's inferred type. Same 1-based line/character convention.
- get_diagnostics(path): uses the language server to return compiler/type-checker errors, warnings, and hints for a file, the same information an editor would show as squiggles. Useful for checking whether a file has real issues before or after reasoning about it.

The four LSP-backed tools (goto_definition, find_references, hover, get_diagnostics) only work for files whose language has a configured server: python, go, javascript, typescript. They also need a real line/character position to query — get one first from a find_symbol result (which reports the exact position of the symbol's name) or from read_file's line-number gutter, rather than guessing coordinates. Prefer find_symbol when you know the name: it gives you both line and character exactly. From read_file you get the line for free from the gutter, but you still have to count the column yourself, so use it for positions find_symbol can't give you.

Working principles:

1. Search before you walk. When you're looking for something specific and don't know where it lives, grep_search or find_files will find it in one call; list_dir plus read_file sweeps will take many. Use list_dir to orient yourself in an unfamiliar layout, not as a way to hunt for a known name. Don't assume file paths — verify them.
2. Prefer the cheapest tool that answers your question. grep_search to find where something is mentioned at all. find_files to locate a file by name. list_dir when you just need to know what exists in one place. find_symbol when you know the name of a definition and want its precise source rather than the whole file. read_file when you need surrounding context, ideally with start_line/end_line around a hit you already have. Reach for the LSP tools (goto_definition, find_references, hover, get_diagnostics) specifically when the question is about relationships across the codebase (where is this defined elsewhere, who calls this, what's its type, does this file have errors) rather than about a single file's own contents. A good default chain for "how does X work" is grep_search to find X, then find_symbol on the most promising hit, then find_references from the position it returns.
3. Tool results are ground truth, not suggestions. If a tool returns an error or "not found," don't assume the symbol/file exists elsewhere without checking — investigate with list_dir/read_file rather than guessing. If an LSP tool returns an error about the file's language not being configured, fall back to find_symbol/read_file instead of retrying.
4. Match the `language` argument to the actual file extension you're inspecting (.py -> python, .go -> go, .js -> javascript, .ts -> typescript). If a file's language isn't supported by find_symbol, fall back to read_file.
5. Be economical with read_file on large files — use start_line/end_line once you have a rough idea where something is, rather than reading entire files repeatedly.
6. All paths are relative to your sandboxed working directory. You cannot read or write anything outside it.

When asked to analyze or answer questions about a repository, work step by step: clone if needed, explore structure, then read or locate only the specific content relevant to the question. Explain your findings clearly once you've gathered enough information — don't guess at code you haven't actually read."""

    def __init__(self, repo: str, profile_log: Path):
        self.repo = repo
        self.client = Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
        self.sandbox = Sandbox()
        self.profile = ProfileLog(profile_log)
        self.messages = []

    def _execute_tool(self, name: str, tool_input: dict):
        """Name -> function dispatch. Returns (result, invoke_secs, run_secs)."""
        invoke_start = time.perf_counter()
        if name == "read_file":
            fn = self.sandbox.read_file
            args = (
                tool_input["path"],
                tool_input.get("start_line"),
                tool_input.get("end_line"),
            )
            kwargs = {}
        elif name == "grep_search":
            fn = self.sandbox.grep_search
            args = ()
            kwargs = {
                "pattern": tool_input["pattern"],
                "path": tool_input.get("path", "."),
                "glob": tool_input.get("glob"),
                "case_insensitive": tool_input.get("case_insensitive", False),
                "fixed_string": tool_input.get("fixed_string", False),
                "context_lines": tool_input.get("context_lines", 0),
            }
        elif name == "find_files":
            fn = self.sandbox.find_files
            args = (tool_input["glob"], tool_input.get("path", "."))
            kwargs = {}
        elif name == "list_dir":
            fn = self.sandbox.list_dir
            args = (tool_input.get("path", "."),)
            kwargs = {}
        elif name == "clone_repo":
            fn = self.sandbox.clone_repo
            args = (tool_input["repo_url"],)
            kwargs = {}
        elif name == "find_symbol":
            fn = self.sandbox.find_symbol
            args = (tool_input["path"], tool_input["symbol"], tool_input["language"])
            kwargs = {}
        elif name == "goto_definition":
            fn = self.sandbox.goto_definition
            args = (tool_input["path"], tool_input["line"], tool_input["character"])
            kwargs = {}
        elif name == "find_references":
            fn = self.sandbox.find_references
            args = (tool_input["path"], tool_input["line"], tool_input["character"])
            kwargs = {}
        elif name == "hover":
            fn = self.sandbox.hover
            args = (tool_input["path"], tool_input["line"], tool_input["character"])
            kwargs = {}
        elif name == "get_diagnostics":
            fn = self.sandbox.get_diagnostics
            args = (tool_input["path"],)
            kwargs = {}
        else:
            invoke_elapsed = time.perf_counter() - invoke_start
            return f"Error: unknown tool '{name}'", invoke_elapsed, 0.0
        invoke_elapsed = time.perf_counter() - invoke_start

        run_start = time.perf_counter()
        result = fn(*args, **kwargs)
        run_elapsed = time.perf_counter() - run_start
        return result, invoke_elapsed, run_elapsed

    def _run_agent_turn(self, max_turns=MAX_TURNS):
        """Runs tool-use turns until the model stops calling tools (i.e. gives a final text answer)."""
        for turn in range(1, max_turns + 1):
            llm_start = time.perf_counter()
            response = self.client.messages.create(
                model=MODEL,
                max_tokens=4096,
                system=self.SYSTEM_PROMPT,
                messages=self.messages,
                tools=TOOLS,
            )
            llm_elapsed = time.perf_counter() - llm_start
            self.profile.event(
                "llm_response",
                turn=turn,
                duration_s=llm_elapsed,
                stop_reason=response.stop_reason,
                input_tokens=getattr(response.usage, "input_tokens", None),
                output_tokens=getattr(response.usage, "output_tokens", None),
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

                result, invoke_elapsed, run_elapsed = self._execute_tool(
                    block.name, block.input
                )
                self.profile.event(
                    "tool_call",
                    turn=turn,
                    tool=block.name,
                    tool_input=block.input,
                    invoke_s=invoke_elapsed,
                    run_s=run_elapsed,
                    total_s=invoke_elapsed + run_elapsed,
                    result_chars=len(result) if isinstance(result, str) else None,
                )

                tool_results.append(
                    {"type": "tool_result", "tool_use_id": block.id, "content": result}
                )
            self.messages.append({"role": "user", "content": tool_results})

        print("\n[warning] hit max_turns without a final response")
        self.profile.event("max_turns_hit", max_turns=max_turns)

    def start_agent(self):
        print(f"=== Profile log: {self.profile.path} (session {self.profile.session_id}) ===")
        print("=== Bootstrapping: cloning and exploring repo ===")
        bootstrap = (
            f"Clone the repository at {self.repo}, then run an initial exploration "
            f"pass with list_dir (recurse into a few key subdirectories if the "
            f"top level looks like a monorepo or has an obvious src/ layout). "
            f"Briefly summarize what kind of project this looks like once done."
        )
        self.messages.append({"role": "user", "content": bootstrap})
        self.profile.event(
            "user_message", kind="bootstrap", chars=len(bootstrap), repo=self.repo
        )
        try:
            self._run_agent_turn()

            print("\n=== Repo ready. Ask questions about it (type 'exit' to quit) ===")
            while True:
                try:
                    user_input = input(f"\n{GREEN}[you]: ").strip()
                    print(RESET, end="")
                except EOFError:
                    break
                if user_input.lower() in ("exit", "quit"):
                    break
                if not user_input:
                    continue

                self.messages.append({"role": "user", "content": user_input})
                self.profile.event("user_message", kind="query", chars=len(user_input))
                self._run_agent_turn()
        finally:
            # tear down the LSP subprocess(es) so nothing is left running
            # after the session ends, whether it exits cleanly or errors out
            self.sandbox.lsp.shutdown_all()

        self.profile.event("session_end")
        print(f"\nProfile written to {self.profile.path}")
        print(f"Graph it with: python ex_3_profile.py --log {self.profile.path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=str, required=True)
    parser.add_argument(
        "--profile-log",
        type=Path,
        default=DEFAULT_PROFILE_LOG,
        help="JSONL file for timing events (default: ex_3_profile.jsonl next to this script)",
    )
    args = parser.parse_args()
    args.profile_log = resolve_log_path(args.profile_log)
    AgentLoop(args.repo, args.profile_log).start_agent()