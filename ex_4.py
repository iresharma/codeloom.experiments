# Orchestrator -> workers: async agent loop with background specialists.

from __future__ import annotations

import argparse
import asyncio
import os
import subprocess
import uuid

from anthropic import AsyncAnthropic

MODEL = "claude-sonnet-4-6"
MAX_TURNS = 25
MAX_READ_LINES = 2000
MAX_READ_BYTES = 250_000
BINARY_SNIFF_BYTES = 8192

BLUE = "\033[94m"
GREEN = "\033[92m"
RED = "\033[91m"
CYAN = "\033[96m"
YELLOW = "\033[93m"
DIM = "\033[2m"
RESET = "\033[0m"

TOOLS = [
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
        "name": "read_file",
        "description": (
            "Read a file at the given path, relative to the working directory. "
            "Returns contents with a 1-based line-number gutter. Optional "
            "start_line/end_line read a slice. Long files are truncated."
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
]

ORCHESTRATOR_TOOLS = TOOLS + [
    {
        "name": "spawn_agent",
        "description": (
            "Start an isolated worker in the background with a single task. "
            "Returns immediately with the worker id. "
            "You MUST pass `paths`: the only files/directories the worker can touch. "
            "Paths of running workers must not overlap — a spawn that collides is rejected. "
            "List the repo yourself first, then give each worker a tight, disjoint slice. "
            "Spawn several workers only when those slices are independent."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "description": "Short name for the worker, e.g. 'explorer' or 'reviewer'",
                },
                "task": {
                    "type": "string",
                    "description": (
                        "The one task the worker should complete. Name only the "
                        "files/dirs in `paths`. Do not ask it to also look elsewhere."
                    ),
                },
                "paths": {
                    "type": "array",
                    "items": {"type": "string"},
                    "minItems": 1,
                    "description": (
                        "Disjoint path prefixes this worker may list or read, e.g. "
                        "['README.md', 'pyproject.toml'] or ['src/flask']. "
                        "Must not overlap another running worker. Do not pass '.' "
                        "unless this single worker owns the whole tree."
                    ),
                },
            },
            "required": ["name", "task", "paths"],
        },
    },
    {
        "name": "list_agents",
        "description": "List all workers with id, name, status, and current activity.",
        "input_schema": {"type": "object", "properties": {}},
    },
    {
        "name": "await_agent",
        "description": "Wait until one worker finishes and return its final answer.",
        "input_schema": {
            "type": "object",
            "properties": {
                "id": {
                    "type": "string",
                    "description": "Worker id returned by spawn_agent",
                },
            },
            "required": ["id"],
        },
    },
    {
        "name": "await_all",
        "description": "Wait until every running worker finishes and return all results.",
        "input_schema": {"type": "object", "properties": {}},
    },
]


def _norm_rel(path: str) -> str:
    p = (path or ".").replace("\\", "/").strip()
    if not p or p == ".":
        return "."
    return p.lstrip("./")


def _path_overlaps(a: str, b: str) -> bool:
    a, b = _norm_rel(a), _norm_rel(b)
    if a == "." or b == ".":
        return True
    return a == b or a.startswith(b + "/") or b.startswith(a + "/")


def _sets_overlap(left: list[str], right: list[str]) -> bool:
    return any(_path_overlaps(a, b) for a in left for b in right)


def _path_allowed(path: str, allowed: list[str]) -> bool:
    p = _norm_rel(path)
    for raw in allowed:
        a = _norm_rel(raw)
        if a == "." or p == a or p.startswith(a + "/"):
            return True
    return False


class ActivityLog:
    """Serialized terminal output so overlapping workers don't scramble lines."""

    def __init__(self):
        self._lock = asyncio.Lock()

    async def emit(self, speaker: str, kind: str, text: str, color: str = CYAN):
        label = f"{speaker} {kind}" if kind else speaker
        async with self._lock:
            print(f"\n{color}[{label}]: {text}{RESET}", flush=True)


class Sandbox:
    def __init__(self):
        self.workdir = os.path.abspath("ex_4_workspace")
        os.makedirs(self.workdir, exist_ok=True)

    def _resolve(self, rel_path: str) -> str:
        full = os.path.abspath(os.path.join(self.workdir, rel_path))
        try:
            if os.path.commonpath([full, self.workdir]) != self.workdir:
                raise ValueError(f"Path '{rel_path}' escapes the working directory")
        except ValueError:
            raise ValueError(f"Path '{rel_path}' escapes the working directory")
        return full

    def clone_repo(self, repo_url: str) -> str:
        if os.listdir(self.workdir):
            return f"Workspace already has files at {self.workdir}; skipping clone"
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
        except (OSError, ValueError) as e:
            return f"Error reading '{path}': {e}"


class Worker:
    """Isolated background agent: fixed system prompt, one task, no user chat."""

    SYSTEM_PROMPT = """You are an isolated coding worker. You receive exactly one task and a set of allowed paths. You never interact with a user.

- list_dir(path): lists files and directories at a given path, non-recursive.
- read_file(path, start_line, end_line): returns the text of a file with a 1-based line-number gutter. start_line/end_line are optional.

Stay in your lane:
- You may only list_dir/read_file paths inside your allowed set. Anything else fails.
- Do the minimum number of tool calls to answer the task. If the task names specific files, read only those.
- Do not expand the task. Do not peek at sibling directories or extra modules "for completeness" or a "thorough summary".
- As soon as you can answer, stop calling tools and return the answer.

All paths are relative to the sandboxed working directory."""

    def __init__(
        self,
        worker_id: str,
        name: str,
        task: str,
        paths: list[str],
        sandbox: Sandbox,
        client: AsyncAnthropic,
        log: ActivityLog,
    ):
        self.id = worker_id
        self.name = name
        self.task = task
        self.paths = paths
        self.sandbox = sandbox
        self.client = client
        self.log = log
        self.messages = []
        self.status = "pending"
        self.activity = "queued"
        self.result: str | None = None
        self.handle: asyncio.Task | None = None

    def snapshot(self) -> str:
        return f"{self.id}  {self.name}  {self.status}  {self.activity}  {self.paths}"

    async def run(self) -> str:
        self.status = "running"
        self.activity = "starting"
        await self.log.emit(
            self.name, "start", f"paths={self.paths} | {self.task}", CYAN
        )
        self.messages.append({
            "role": "user",
            "content": (
                f"Allowed paths (tool calls outside this set will fail): "
                f"{', '.join(self.paths)}\n\nTask: {self.task}"
            ),
        })
        try:
            self.result = await self._run_turns()
            self.status = "done"
            self.activity = "done"
            await self.log.emit(self.name, "done", self.result, CYAN)
            return self.result
        except Exception as e:
            self.status = "error"
            self.activity = f"error: {e}"
            self.result = f"Error: {e}"
            await self.log.emit(self.name, "error", str(e), RED)
            return self.result

    async def _run_turns(self, max_turns=MAX_TURNS) -> str:
        last_text = ""
        for _ in range(1, max_turns + 1):
            self.activity = "thinking"
            await self.log.emit(self.name, "thinking", "calling model", DIM)
            response = await self.client.messages.create(
                model=MODEL,
                max_tokens=4096,
                system=self.SYSTEM_PROMPT,
                messages=self.messages,
                tools=TOOLS,
            )

            texts = []
            for block in response.content:
                if block.type == "text" and block.text.strip():
                    await self.log.emit(self.name, "", block.text, CYAN)
                    texts.append(block.text)
            last_text = "\n".join(texts)

            self.messages.append({"role": "assistant", "content": response.content})

            if response.stop_reason != "tool_use":
                return last_text

            tool_results = []
            for block in response.content:
                if block.type != "tool_use":
                    continue
                self.activity = f"{block.name}({block.input})"
                await self.log.emit(self.name, "tool", self.activity, RED)
                result = await execute_tool(
                    self.sandbox, block.name, block.input, worker=self
                )
                tool_results.append(
                    {"type": "tool_result", "tool_use_id": block.id, "content": result}
                )
            self.messages.append({"role": "user", "content": tool_results})

        await self.log.emit(self.name, "warn", "hit max_turns", YELLOW)
        return last_text or f"[{self.name} hit max_turns without a final response]"


class WorkerPool:
    def __init__(self, sandbox: Sandbox, client: AsyncAnthropic, log: ActivityLog):
        self.sandbox = sandbox
        self.client = client
        self.log = log
        self.workers: dict[str, Worker] = {}

    def snapshot(self) -> str:
        if not self.workers:
            return "(no workers spawned)"
        lines = ["id        name  status  activity"]
        for worker in self.workers.values():
            lines.append(worker.snapshot())
        return "\n".join(lines)

    def results(self) -> str:
        if not self.workers:
            return "(no workers spawned)"
        parts = []
        for worker in self.workers.values():
            body = worker.result if worker.result is not None else f"({worker.status}: {worker.activity})"
            parts.append(f"=== {worker.name} ({worker.id}) [{worker.status}] ===\n{body}")
        return "\n\n".join(parts)

    def print_board(self):
        print(f"\n{DIM}--- workers ---\n{self.snapshot()}\n---------------{RESET}", flush=True)

    def spawn(self, name: str, task: str, paths: list[str]) -> str:
        paths = [_norm_rel(p) for p in paths if str(p).strip()]
        if not paths:
            return "Error: spawn_agent requires a non-empty paths list"

        running = [
            w for w in self.workers.values()
            if w.status in ("pending", "running")
        ]
        for other in running:
            if _sets_overlap(paths, other.paths):
                return (
                    f"Error: paths {paths} overlap running worker "
                    f"'{other.name}' ({other.id}) which owns {other.paths}. "
                    f"Give this worker a disjoint slice."
                )

        worker_id = uuid.uuid4().hex[:8]
        worker = Worker(
            worker_id, name, task, paths, self.sandbox, self.client, self.log
        )
        self.workers[worker_id] = worker
        worker.handle = asyncio.create_task(
            worker.run(), name=f"worker-{name}-{worker_id}"
        )
        return f"spawned '{name}' id={worker_id} paths={paths} status=running"

    async def await_one(self, worker_id: str) -> str:
        worker = self.workers.get(worker_id)
        if worker is None:
            return f"unknown worker id '{worker_id}'"
        if worker.handle is not None:
            await worker.handle
        return worker.result or "(no result)"

    async def await_all(self) -> str:
        pending = [
            w.handle for w in self.workers.values()
            if w.handle is not None and not w.handle.done()
        ]
        if pending:
            await asyncio.gather(*pending, return_exceptions=True)
        return self.results()


async def execute_tool(
    sandbox: Sandbox,
    name: str,
    tool_input: dict,
    pool: WorkerPool | None = None,
    worker: Worker | None = None,
) -> str:
    if name in ("list_dir", "read_file"):
        path = tool_input.get("path", ".") if name == "list_dir" else tool_input["path"]
        if worker is not None and not _path_allowed(path, worker.paths):
            return (
                f"Error: path '{path}' is outside this worker's allowed paths "
                f"({', '.join(worker.paths)}). Stay in scope and finish the task."
            )
        if name == "list_dir":
            return sandbox.list_dir(path)
        return sandbox.read_file(
            path,
            tool_input.get("start_line"),
            tool_input.get("end_line"),
        )
    if pool is None:
        return f"Error: unknown tool '{name}'"
    if name == "spawn_agent":
        return pool.spawn(
            tool_input["name"],
            tool_input["task"],
            tool_input.get("paths") or [],
        )
    if name == "list_agents":
        return pool.snapshot()
    if name == "await_agent":
        return await pool.await_one(tool_input["id"])
    if name == "await_all":
        return await pool.await_all()
    return f"Error: unknown tool '{name}'"


class Orchestrator:
    SYSTEM_PROMPT = """You are the orchestrator. You talk to the user. The user will never mention workers, agents, or spawning — you decide that on your own. If a question touches more than one file or area, spawn workers. Do not wait to be asked.

Workers do the file reading. Overlapping workers waste tokens — never spawn two workers that would look at the same files.

- list_dir(path): lists files and directories at a given path, non-recursive. Use this yourself first so you can partition work.
- read_file(path, start_line, end_line): file text with a 1-based line-number gutter. Only for a tiny one-file check; otherwise spawn.
- spawn_agent(name, task, paths): start a background worker. `paths` is required and is the only tree that worker can touch. Running workers' paths must not overlap or the spawn is rejected. Do not pass '.' unless one worker owns the whole job.
- list_agents(): every worker's id, status, and current activity.
- await_agent(id): wait for one worker and get its final answer.
- await_all(): wait for every running worker and get all of their answers.

When to spawn (default: yes):
- Compare two things, explain a subsystem, or anything that needs more than one file → spawn, one worker per disjoint slice.
- A yes/no about a path you already know → you may list_dir/read_file yourself.

How to delegate:
1. list_dir yourself (and maybe one more level) before spawning anyone.
2. Split into disjoint slices — e.g. one worker gets ['README.md', 'pyproject.toml'], another gets ['src']. Never both on src. Never both on '.'.
3. The task text must only ask about those paths. Do not tell a README worker to "also peek at src".
4. Prefer specific files over a parent directory. A directory path is for walking that tree, not a license to also read siblings.
5. After spawning, await them before you answer the user.

Workers never talk to the user. Summarize their findings. All paths are relative to the sandboxed working directory."""

    def __init__(self, repo: str):
        self.repo = repo
        self.client = AsyncAnthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
        self.sandbox = Sandbox()
        self.log = ActivityLog()
        self.pool = WorkerPool(self.sandbox, self.client, self.log)
        self.messages = []

    async def _run_turns(self, max_turns=MAX_TURNS):
        for _ in range(1, max_turns + 1):
            await self.log.emit("orchestrator", "thinking", "calling model", DIM)
            response = await self.client.messages.create(
                model=MODEL,
                max_tokens=4096,
                system=self.SYSTEM_PROMPT,
                messages=self.messages,
                tools=ORCHESTRATOR_TOOLS,
            )

            for block in response.content:
                if block.type == "text" and block.text.strip():
                    await self.log.emit("orchestrator", "", block.text, BLUE)

            self.messages.append({"role": "assistant", "content": response.content})

            if response.stop_reason != "tool_use":
                return

            tool_results = []
            for block in response.content:
                if block.type != "tool_use":
                    continue
                await self.log.emit("orchestrator", "tool", f"{block.name}({block.input})", RED)
                result = await execute_tool(
                    self.sandbox, block.name, block.input, pool=self.pool
                )
                tool_results.append(
                    {"type": "tool_result", "tool_use_id": block.id, "content": result}
                )
            self.messages.append({"role": "user", "content": tool_results})

        await self.log.emit("orchestrator", "warn", "hit max_turns", YELLOW)

    async def start(self):
        print("=== Cloning repo into workspace ===")
        clone_result = await asyncio.to_thread(self.sandbox.clone_repo, self.repo)
        print(clone_result)

        bootstrap = (
            f"The repository {self.repo} is already cloned into the working directory. "
            "list_dir the top level yourself first. Then spawn at most a couple of "
            "workers with disjoint `paths` (for example metadata files vs src, never "
            "both on src). Await them and briefly summarize what kind of project this is."
        )
        self.messages.append({"role": "user", "content": bootstrap})
        await self._run_turns()

        print(
            "\n=== Repo ready. Ask questions (type 'exit' to quit, 'status' for workers) ==="
        )
        while True:
            self.pool.print_board()
            try:
                user_input = await asyncio.to_thread(
                    lambda: input(f"\n{GREEN}[you]: ").strip()
                )
                print(RESET, end="")
            except EOFError:
                break
            if user_input.lower() in ("exit", "quit"):
                break
            if not user_input:
                continue
            if user_input.lower() == "status":
                self.pool.print_board()
                continue

            self.messages.append({"role": "user", "content": user_input})
            await self._run_turns()

        pending = [
            w.handle for w in self.pool.workers.values()
            if w.handle is not None and not w.handle.done()
        ]
        if pending:
            print(f"\n{DIM}waiting for {len(pending)} worker(s) to finish…{RESET}")
            await asyncio.gather(*pending, return_exceptions=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=str, required=True)
    args = parser.parse_args()
    try:
        asyncio.run(Orchestrator(args.repo).start())
    except KeyboardInterrupt:
        print("\ninterrupted")
