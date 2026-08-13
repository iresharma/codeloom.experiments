import os

def print_tree(node, source_bytes, indent=0, named_only=True):
    text = source_bytes[node.start_byte:node.end_byte].decode(errors="replace")
    snippet = text.split("\n")[0][:40]
    print(f"{'  ' * indent}{node.type} [{node.start_point[0]}:{node.end_point[0]}] '{snippet}'")
    children = node.named_children if named_only else node.children
    for child in children:
        print_tree(child, source_bytes, indent + 1, named_only)


def print_diagnostics(diag_json, filepath=None):
    diags = diag_json.get("generalDiagnostics", [])
    summary = diag_json.get("summary", {})

    if filepath:
        target = os.path.abspath(str(filepath))
        diags = [d for d in diags if os.path.abspath(d.get("file", "")) == target]

    if not diags:
        print("no diagnostics")
    else:
        for d in diags:
            start = d["range"]["start"]
            end = d["range"]["end"]
            severity = d["severity"].upper()
            rule = f" ({d['rule']})" if "rule" in d else ""
            loc = f"{start['line'] + 1}:{start['column'] + 1}"
            if end["line"] != start["line"]:
                loc += f"-{end['line'] + 1}:{end['column'] + 1}"
            print(f"[{severity}] {loc} {d['message']}{rule}")

    if summary:
        counts = ", ".join(
            f"{summary[k]} {k.replace('Count', '')}"
            for k in ("errorCount", "warningCount", "informationCount")
            if k in summary
        )
        print(f"-- {counts} --")