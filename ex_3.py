# An experiment to see if we can use tree-sitter and LSP to find the exact definition of a function, class, or method by name in a specific file, returning just that symbol's source code.

from __future__ import annotations

import argparse
import json
import logging
import os
import queue
import subprocess
import threading
import time
import uuid
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import ClassVar

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
DEFAULT_LSP_LOG = SCRIPT_DIR / "ex_3_lsp.log"

# Dedicated logger, separate from the "profile" JSONL and from root logging
# (so importing anthropic/etc. doesn't spam this file). File handler gets
# attached in __main__ once we know the actual log path from argparse.
lsp_logger = logging.getLogger("lsp")
lsp_logger.setLevel(logging.DEBUG)
lsp_logger.propagate = False


def resolve_log_path(path: Path) -> Path:
    """Relative --lsp-log/--profile-log paths go next to this script, not cwd."""
    path = Path(path)
    if not path.is_absolute():
        path = SCRIPT_DIR / path
    path.parent.mkdir(parents=True, exist_ok=True)
    return path.resolve()


def setup_lsp_logging(path: Path) -> Path:
    path = resolve_log_path(path)
    handler = logging.FileHandler(path, mode="a", encoding="utf-8")
    handler.setFormatter(
        logging.Formatter("%(asctime)s [%(levelname)s] %(threadName)s: %(message)s")
    )
    lsp_logger.addHandler(handler)
    lsp_logger.info("=" * 60)
    lsp_logger.info("LSP debug logging started")
    return path

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
        "description": "uses tree-sitter to find exact symbols like functions and variables in the code",
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
]


# ============================================================================
# LSP client — generic JSON-RPC-over-stdio protocol client, per-language
# server configs, and a manager that spawns/queries the right server based
# on each file's extension (not tied to a single "project language" —
# self.language on Sandbox is unrelated to this; LSPManager decides which
# server to use per file, on demand, the first time that file is queried).
# Not yet wired into any agent tool (see Sandbox.lsp) — usage comes later.
# ============================================================================

class LSPTimeoutError(RuntimeError):
    pass


class LSPClient:
    def __init__(self, cmd: list[str], cwd: str, env: dict | None = None):
        self.cmd = cmd
        self.cwd = cwd
        lsp_logger.info(f"spawning: {' '.join(cmd)} (cwd={cwd})")
        self.proc = subprocess.Popen(
            cmd,
            cwd=cwd,
            env=env,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        lsp_logger.info(f"spawned pid={self.proc.pid} cmd={cmd[0]}")
        self._pending: dict[int, queue.Queue] = {}
        self._notifications: queue.Queue[dict] = queue.Queue()
        self._next_id = 1
        self._id_lock = threading.Lock()
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
        method = payload.get("method", "<response>")
        msg_id = payload.get("id", "-")
        lsp_logger.debug(f"--> [{self.cmd[0]}] id={msg_id} method={method}")
        try:
            self.proc.stdin.write(header + body)
            self.proc.stdin.flush()
        except (BrokenPipeError, OSError) as e:
            lsp_logger.error(f"write failed for {method}: {e}")
            raise RuntimeError(f"LSP server process died: {e}")

    # ---------- background loops ----------

    def _reader_loop(self):
        while self._alive:
            try:
                msg = self._read_message(self.proc.stdout)
            except (OSError, ValueError, json.JSONDecodeError) as e:
                lsp_logger.error(f"[{self.cmd[0]}] reader loop error: {e}")
                break
            if msg is None:
                lsp_logger.info(f"[{self.cmd[0]}] stdout closed — server exited")
                break
            if "id" in msg and ("result" in msg or "error" in msg):
                lsp_logger.debug(
                    f"<-- [{self.cmd[0]}] id={msg['id']} "
                    f"{'error' if 'error' in msg else 'result'}"
                )
                q = self._pending.pop(msg["id"], None)
                if q is not None:
                    q.put(msg)
            elif "method" in msg:
                lsp_logger.debug(f"<-- [{self.cmd[0]}] notification method={msg['method']}")
                # server->client request or notification (e.g. publishDiagnostics,
                # or window/workDoneProgress/create which expects a response)
                self._notifications.put(msg)
        self._alive = False

    def _stderr_drain(self):
        # must drain or the pipe fills and blocks the server; also useful for debugging
        for line in iter(self.proc.stderr.readline, b""):
            text = line.decode("utf-8", "replace").rstrip()
            if text:
                lsp_logger.debug(f"[{self.cmd[0]} stderr] {text}")

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
            lsp_logger.error(f"[{self.cmd[0]}] {method} timed out after {timeout}s")
            raise LSPTimeoutError(f"{method} timed out after {timeout}s")
        if "error" in response:
            lsp_logger.error(f"[{self.cmd[0]}] {method} returned error: {response['error']}")
            raise RuntimeError(f"{method} error: {response['error']}")
        return response.get("result")

    def notify(self, method: str, params: dict):
        self._write_message({"jsonrpc": "2.0", "method": method, "params": params})

    def get_notification(self, timeout: float = 0.0) -> dict | None:
        """Non-blocking-by-default pop from the notification queue."""
        try:
            return self._notifications.get(timeout=timeout)
        except queue.Empty:
            return None

    def drain_notifications(self) -> list[dict]:
        out = []
        while True:
            n = self.get_notification(timeout=0.0)
            if n is None:
                break
            out.append(n)
        return out

    def shutdown(self):
        if not self._alive:
            return
        lsp_logger.info(f"[{self.cmd[0]}] shutting down pid={self.proc.pid}")
        try:
            self.request("shutdown", {}, timeout=5.0)
            self.notify("exit", {})
        except (LSPTimeoutError, RuntimeError, OSError, BrokenPipeError) as e:
            lsp_logger.warning(f"[{self.cmd[0]}] shutdown handshake failed: {e}")
            self._alive = False
        else:
            self._alive = False
        try:
            self.proc.terminate()
            self.proc.wait(timeout=5.0)
            lsp_logger.info(f"[{self.cmd[0]}] process exited cleanly")
        except (OSError, subprocess.TimeoutExpired):
            lsp_logger.warning(f"[{self.cmd[0]}] did not exit in time, killing")
            self.proc.kill()


class LSPManager:
    @dataclass
    class ServerConfig:
        language: str                 # descriptive only, matches EXTENSION_MAP values
        cmd: list[str]                 # subprocess argv
        language_id: str               # LSP "languageId" for didOpen
        extensions: tuple[str, ...]    # which files this server should be asked about
        init_options: dict = field(default_factory=dict)   # merged into initialize params

    # Real class attributes — built once when the class is defined, shared
    # across every LSPManager instance. This is what was broken before:
    # they were being rebuilt as *instance* attributes inside __init__,
    # while _config_for_extension (a classmethod) kept reading them off
    # the class, which never had them. Same dict either way, but now it's
    # actually reachable from where it's read.
    SERVER_CONFIGS: ClassVar[dict[str, ServerConfig]] = {
        "python": ServerConfig(
            language="python",
            # npx avoids requiring a global npm install as a setup step
            cmd=["npx", "-y", "-p", "pyright", "pyright-langserver", "--stdio"],
            language_id="python",
            extensions=(".py", ".pyi"),
            # "openFilesOnly" (pyright's default) only analyzes files that
            # got an explicit didOpen. "workspace" makes it analyze/publish
            # diagnostics for the whole project on its own, so a background
            # warm-start (see LSPManager.warm_start) actually produces a
            # repo-wide pass instead of sitting idle until something is opened.
            init_options={"python.analysis": {"diagnosticMode": "workspace"}},
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
        client = self._clients.get(key)
        if client is not None:
            return client

        client = LSPClient(cfg.cmd, cwd=self.root)
        lsp_logger.info(f"[{cfg.language}] sending initialize request...")
        init_start = time.monotonic()
        result = client.request(
            "initialize",
            {
                "processId": os.getpid(),
                "rootUri": self.root_uri,
                "capabilities": {
                    "textDocument": {
                        "synchronization": {"didSave": True},
                        "publishDiagnostics": {"relatedInformation": True},
                    }
                },
                "initializationOptions": cfg.init_options,
            },
            timeout=30.0,  # some servers are slow to boot the first time
        )
        init_elapsed = time.monotonic() - init_start
        server_name = (result or {}).get("serverInfo", {}).get("name", "unknown")
        lsp_logger.info(
            f"[{cfg.language}] initialized in {init_elapsed:.2f}s "
            f"(server reports: {server_name})"
        )
        client.notify("initialized", {})
        self._clients[key] = client

        # Start a dedicated background reader for this client's notification
        # queue right away, rather than only draining it opportunistically
        # from inside open_file_and_get_diagnostics. This is what makes
        # workspace-wide diagnostics (see the python config's diagnosticMode)
        # actually visible — pyright can start publishing diagnostics for
        # files nobody has explicitly opened, and without this listener
        # those notifications would just sit unread in the queue forever.
        threading.Thread(
            target=self._notification_listener,
            args=(client, cfg),
            daemon=True,
            name=f"lsp-notify-{cfg.language}",
        ).start()

        return client

    def _notification_listener(self, client: LSPClient, cfg: ServerConfig):
        """
        Runs for the lifetime of `client`. Every publishDiagnostics
        notification updates self._diagnostics (so open_file_and_get_diagnostics
        can just read the cache instead of racing this thread for messages
        off the same queue), and logs a one-line summary so you can watch
        the server doing work in the log file in real time.
        """
        while client._alive:
            note = client.get_notification(timeout=0.5)
            if note is None:
                continue
            method = note.get("method")
            if method == "textDocument/publishDiagnostics":
                params = note.get("params", {})
                uri = params.get("uri", "<unknown>")
                diags = params.get("diagnostics", [])
                self._diagnostics[uri] = diags
                rel = uri.removeprefix(self.root_uri).lstrip("/")
                if diags:
                    lsp_logger.info(
                        f"[{cfg.language}] diagnostics: {rel} — {len(diags)} issue(s)"
                    )
                    for d in diags[:5]:  # cap so one messy file doesn't flood the log
                        sev = d.get("severity", "?")
                        line = d.get("range", {}).get("start", {}).get("line", "?")
                        msg = d.get("message", "").splitlines()[0][:120]
                        lsp_logger.debug(f"    L{line} sev={sev}: {msg}")
                else:
                    lsp_logger.debug(f"[{cfg.language}] diagnostics: {rel} — clean")
            elif method == "$/progress":
                value = note.get("params", {}).get("value", {})
                kind = value.get("kind")
                title = value.get("title") or value.get("message")
                if kind:
                    lsp_logger.info(f"[{cfg.language}] progress[{kind}]: {title}")
            else:
                lsp_logger.debug(f"[{cfg.language}] unhandled notification: {method}")

    def shutdown_all(self):
        lsp_logger.info("shutting down all LSP clients")
        for client in self._clients.values():
            client.shutdown()
        self._clients.clear()
        self._opened_files.clear()

    def warm_start(self, language: str) -> LSPClient | None:
        """
        Eagerly spawn + initialize the server for `language`, without
        waiting for any tool call. Lets indexing happen in the background
        while the agent is doing unrelated exploration (list_dir/read_file),
        instead of only starting on the first ask_lsp/open_file call.

        Meant to be called from a background thread (see Sandbox._warm_lsp) —
        this blocks on the initialize handshake, so calling it synchronously
        on the main thread would defeat the point.
        """
        cfg = self.SERVER_CONFIGS.get(language)
        if cfg is None:
            lsp_logger.warning(f"warm_start: no server configured for '{language}'")
            return None  # no server configured for this language — nothing to warm
        lsp_logger.info(f"warm_start: starting server for '{language}'")
        return self._client_for(cfg)

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
        if full_path in self._opened_files:
            return self._diagnostics.get(uri, [])

        ext = Path(full_path).suffix
        cfg = self._config_for_extension(ext)
        if cfg is None:
            raise ValueError(f"No LSP server configured for extension '{ext}'")

        client = self._client_for(cfg)
        text = Path(full_path).read_text(errors="replace")

        lsp_logger.debug(f"[{cfg.language}] didOpen: {rel_path}")
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
        self._opened_files.add(full_path)

        deadline = time.monotonic() + wait_timeout
        while time.monotonic() < deadline:
            if uri in self._diagnostics:
                return self._diagnostics[uri]
            time.sleep(0.1)

        # no diagnostics arrived in time; treat as clean (common for valid files)
        lsp_logger.debug(f"[{cfg.language}] no diagnostics for {rel_path} within {wait_timeout}s")
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

        # NOTE: at this point the workspace is very likely EMPTY. Cloning
        # doesn't happen here — it happens later, inside the agent's first
        # tool call (clone_repo), driven by the LLM during the bootstrap
        # turn. So detection here is best-effort against whatever's already
        # on disk (e.g. a previous run's leftover clone); on a fresh
        # workspace it will almost always come back None, and that's
        # expected, not an error. We used to hard-fail here if it came back
        # None — that was the bug: it required a repo to already be cloned
        # before Sandbox could even be constructed, which is backwards
        # given how bootstrap actually works.
        self.language = self._detect_project_language()
        self.lsp = LSPManager(self.workdir)

        if self.language is not None:
            self._start_lsp_warm_start()
        else:
            lsp_logger.info(
                "Sandbox: no language detected yet (workspace likely empty) "
                "— will detect + warm-start after clone_repo succeeds"
            )

    def _create_workspace(self):
        os.makedirs(self.workdir, exist_ok=True)

    def _start_lsp_warm_start(self):
        # Runs indexing in the background rather than blocking on it —
        # otherwise the server sits idle while the agent does its initial
        # list_dir/read_file exploration, and the first real LSP query
        # pays the full cold-start + indexing latency. A daemon thread
        # means it can't block whatever called this, and if the toolchain
        # for this language isn't installed (e.g. gopls missing), it fails
        # quietly here rather than crashing the session — ask_lsp will
        # surface a real error later if something actually tries to use it.
        #
        # Still not wired into any agent tool — no ask_lsp in TOOLS, no
        # dispatch in _execute_tool. This just gets the server ready.
        print(f"{RED}[lsp] warm-starting '{self.language}'{RESET}")
        threading.Thread(
            target=self._warm_lsp, daemon=True, name="lsp-warm-start"
        ).start()

    def _warm_lsp(self):
        lsp_logger.info(f"Sandbox: warm-starting LSP for detected language '{self.language}'")
        try:
            client = self.lsp.warm_start(self.language)
            if client is None:
                print(
                    f"{RED}[lsp] no server configured for language "
                    f"'{self.language}' — skipping warm start{RESET}"
                )
        except Exception as e:
            # Runs on a background thread — nothing is waiting on this, so
            # there's no caller to propagate the exception to. Log and move
            # on; the toolchain being missing (npx, gopls, etc.) shouldn't
            # take down the whole agent session before it's even started.
            lsp_logger.exception(f"warm start for '{self.language}' failed")
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

        # This is the first point where the workspace actually has content,
        # so it's the first point where language detection can succeed.
        # Re-run it now and, if a language wasn't already known (the usual
        # case — see the comment in Sandbox.__init__), kick off the LSP
        # warm-start here instead. Kept non-fatal: a language-detection
        # miss (e.g. an unsupported language, or a truly empty repo)
        # shouldn't break the clone itself.
        if self.language is None:
            self.language = self._detect_project_language()
            if self.language is not None:
                lsp_logger.info(
                    f"clone_repo: detected language '{self.language}' post-clone"
                )
                self._start_lsp_warm_start()
            else:
                lsp_logger.warning(
                    "clone_repo: still could not detect a project language after clone"
                )

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

    def read_file(self, path: str, start_line: int | None = None, end_line: int | None = None):
        try:
            full = self._resolve(path)
            with open(full, "r", errors="replace") as f:
                lines = f.readlines()
            if start_line is None and end_line is None:
                return "".join(lines)
            start = (start_line or 1) - 1
            end = end_line if end_line is not None else len(lines)
            return "".join(lines[start:end])
        except (OSError, ValueError) as e:
            return f"Error reading '{path}': {e}"

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

            if node:
                return file_bytes[node.start_byte:node.end_byte].decode()
            return f"Symbol '{symbol}' not found in '{path}'"
        except (OSError, ValueError, UnicodeDecodeError) as e:
            return f"Error finding symbol '{symbol}' in '{path}': {e}"


class AgentLoop:
    SYSTEM_PROMPT = """You are a coding agent that explores and analyzes code repositories using a small set of tools. You operate inside a sandboxed working directory and cannot access anything outside it.

Available tools:

- clone_repo(repo_url): clones a git repository into your working directory. Use this first if no repository is present yet.
- list_dir(path): lists files and directories at a given path, non-recursive. Use this to explore structure before reading files — don't guess at paths.
- read_file(path, start_line, end_line): returns the raw text of a file. start_line/end_line are optional; omit them to read the whole file, or use them to read a specific slice of a large file instead of the whole thing.
- find_symbol(path, symbol, language): uses tree-sitter to locate the exact definition of a function, class, or method by name in a specific file, returning just that symbol's source code. Use this instead of read_file when you already know the name of what you're looking for and want its precise definition rather than the whole file. `language` must be one of: python, go, javascript, typescript.

Working principles:

1. Explore before acting. Use list_dir to understand repo structure before reading files. Don't assume file paths or names — verify them.
2. Prefer the cheapest tool that answers your question. Use list_dir over read_file when you just need to know what exists. Use find_symbol over read_file when you know the specific name you're looking for and don't need surrounding context.
3. Tool results are ground truth, not suggestions. If a tool returns an error or "not found," don't assume the symbol/file exists elsewhere without checking — investigate with list_dir/read_file rather than guessing.
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
    parser.add_argument(
        "--lsp-log",
        type=Path,
        default=DEFAULT_LSP_LOG,
        help="Log file for LSP client/server activity (default: ex_3_lsp.log next to this script)",
    )
    args = parser.parse_args()
    args.lsp_log = setup_lsp_logging(args.lsp_log)
    args.profile_log = resolve_log_path(args.profile_log)
    print(f"=== LSP debug log: {args.lsp_log} ===")
    print(f"    tail -f {args.lsp_log}   # to watch it live in another terminal")
    AgentLoop(args.repo, args.profile_log).start_agent()