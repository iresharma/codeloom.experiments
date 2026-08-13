# tree-sitter and lsp basics

import argparse, json, subprocess, sys, tempfile, os
from pathlib import Path
from tree_sitter import Language, Parser
import tree_sitter_python as tspython
import anthropic

PY_LANGUAGE = Language(tspython.language())
parser = Parser(PY_LANGUAGE)


def find_target(tree, source_bytes, target_name):
    stack = [tree.root_node]
    while stack:
        node = stack.pop()
        if node.type in ("function_definition", "class_definition"):
            name_node = node.child_by_field_name("name")
            if name_node and source_bytes[name_node.start_byte:name_node.end_byte].decode() == target_name:
                return node
        stack.extend(node.children)
    return None


def run_pyright(filepath):
    result = subprocess.run(["pyright", "--outputjson", str(filepath)], capture_output=True, text=True)
    try:
        return json.loads(result.stdout)
    except json.JSONDecodeError:
        return {"generalDiagnostics": []}


def diagnostics_in_range(diag_json, start_line, end_line):
    return [
        d for d in diag_json.get("generalDiagnostics", [])
        if d["severity"] == "error" and start_line <= d["range"]["start"]["line"] <= end_line
    ]


def generate_replacement(client, original_code, instruction, prior_errors=None):
    prompt = f"Python code:\n```python\n{original_code}\n```\n\nInstruction: {instruction}\n"
    if prior_errors:
        prompt += f"\nPrevious attempt had these type errors, fix them:\n{json.dumps(prior_errors, indent=2)}\n"
    prompt += "\nRespond with ONLY the replacement code. No markdown fences, no explanation."

    resp = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=2000,
        messages=[{"role": "user", "content": prompt}],
    )
    return resp.content[0].text.strip()

def print_tree(node, source_bytes, indent=0):
    text = source_bytes[node.start_byte:node.end_byte].decode()
    snippet = text.split("\n")[0][:40]
    print(f"{'  ' * indent}{node.type} [{node.start_point[0]}:{node.end_point[0]}] '{snippet}'")
    for child in node.children:
        print_tree(child, source_bytes, indent + 1)

def edit_target(filepath, target_name, instruction, max_retries=3):
    filepath = Path(filepath)
    source_bytes = filepath.read_bytes()
    client = anthropic.Anthropic()
    prior_errors = None

    for attempt in range(1, max_retries + 1):
        tree = parser.parse(source_bytes)
        print_tree(tree.root_node, source_bytes)
        node = find_target(tree, source_bytes, target_name)
        if node is None:
            print(f"target '{target_name}' not found")
            return False

        original_code = source_bytes[node.start_byte:node.end_byte].decode()
        print(f"original code: {original_code}")
        new_code = generate_replacement(client, original_code, instruction, prior_errors)
        candidate_bytes = source_bytes[:node.start_byte] + new_code.encode() + source_bytes[node.end_byte:]

        if parser.parse(candidate_bytes).root_node.has_error:
            print(f"attempt {attempt}: syntax error, retrying")
            prior_errors = [{"message": "generated code produced invalid Python syntax"}]
            continue

        with tempfile.NamedTemporaryFile(suffix=".py", delete=False, dir=filepath.parent) as tmp:
            tmp.write(candidate_bytes)
            tmp_path = tmp.name

        diag_json = run_pyright(tmp_path)
        os.unlink(tmp_path)

        start_line = node.start_point[0]
        end_line = start_line + new_code.count("\n")
        errors = diagnostics_in_range(diag_json, start_line, end_line)

        if not errors:
            filepath.write_bytes(candidate_bytes)
            print(f"applied edit on attempt {attempt}")
            return True

        print(f"attempt {attempt}: {len(errors)} type errors, retrying")
        prior_errors = errors

    print("failed after max retries")
    return False


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("file")
    ap.add_argument("target", help="function or class name to edit")
    ap.add_argument("instruction")
    args = ap.parse_args()
    ok = edit_target(args.file, args.target, args.instruction)
    sys.exit(0 if ok else 1)