import argparse
import os
import subprocess
from pathlib import Path

import tree_sitter_go as tsgo
import tree_sitter_javascript as tsjs
import tree_sitter_python as tspython
import tree_sitter_typescript as tsts
from anthropic import Anthropic
from tree_sitter import Language, Parser

MODEL = "claude-sonnet-4-6"
MAX_TURNS = 25          # hard cap so a buggy loop can't run forever


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


class Sandbox:
    def __init__(self):
        self.workdir = os.path.abspath("ex_3_workspace")
        self._create_workspace()

    def _create_workspace(self):
        os.makedirs(self.workdir, exist_ok=True)

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
            cwd=self.workdir, capture_output=True, text=True
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
        except Exception as e:
            return f"Error listing '{path}': {e}"

    def read_file(self, path: str, start_line: int = None, end_line: int = None):
        try:
            full = self._resolve(path)
            with open(full, "r", errors="replace") as f:
                lines = f.readlines()
            if start_line is None and end_line is None:
                return "".join(lines)
            start = (start_line or 1) - 1
            end = end_line if end_line is not None else len(lines)
            return "".join(lines[start:end])
        except Exception as e:
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
        except Exception as e:
            return f"Error finding symbol '{symbol}' in '{path}': {e}"

def execute_tool(sandbox: Sandbox, name: str, tool_input: dict) -> str:
    """Name -> function dispatch. This is the whole 'engine'."""
    if name == "read_file":
        return sandbox.read_file(
            tool_input["path"],
            tool_input.get("start_line"),
            tool_input.get("end_line"),
        )
    elif name == "list_dir":
        return sandbox.list_dir(tool_input.get("path", "."))
    elif name == "clone_repo":
        return sandbox.clone_repo(tool_input["repo_url"])
    elif name == "find_symbol":
        return sandbox.find_symbol(
            tool_input["path"],
            tool_input["symbol"],
            tool_input["language"],
        )
    else:
        return f"Error: unknown tool '{name}'"


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

def run_agent_turn(client, sandbox, messages, max_turns=MAX_TURNS):
    """Runs tool-use turns until the model stops calling tools (i.e. gives a final text answer)."""
    for turn in range(1, max_turns + 1):
        response = client.messages.create(
            model=MODEL,
            max_tokens=4096,
            system=SYSTEM_PROMPT,
            messages=messages,
            tools=TOOLS,
        )

        for block in response.content:
            if block.type == "text" and block.text.strip():
                print(f"\n[assistant]: {block.text}")

        messages.append({"role": "assistant", "content": response.content})

        if response.stop_reason != "tool_use":
            return

        tool_results = []
        for block in response.content:
            if block.type != "tool_use":
                continue
            print(f"[tool_call]: {block.name}({block.input})")
            result = execute_tool(sandbox, block.name, block.input)
            tool_results.append(
                {"type": "tool_result", "tool_use_id": block.id, "content": result}
            )
        messages.append({"role": "user", "content": tool_results})

    print("\n[warning] hit max_turns without a final response")


def main(repo: str):
    client = Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    sandbox = Sandbox()
    messages = []

    print("=== Bootstrapping: cloning and exploring repo ===")
    messages.append({
        "role": "user",
        "content": (
            f"Clone the repository at {repo}, then run an initial exploration "
            f"pass with list_dir (recurse into a few key subdirectories if the "
            f"top level looks like a monorepo or has an obvious src/ layout). "
            f"Briefly summarize what kind of project this looks like once done."
        ),
    })
    run_agent_turn(client, sandbox, messages)

    print("\n=== Repo ready. Ask questions about it (type 'exit' to quit) ===")
    while True:
        try:
            user_input = input("\n[you]: ").strip()
        except EOFError:
            break
        if user_input.lower() in ("exit", "quit"):
            break
        if not user_input:
            continue

        messages.append({"role": "user", "content": user_input})
        run_agent_turn(client, sandbox, messages)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=str, required=True)
    args = parser.parse_args()
    main(args.repo)


# git@github.com:iresharma/Variable-bitrate-Live-streamin-server.git