"""Open interactive profiling graphs from ex_3_profile.jsonl.

Usage:
  python ex_3_profile.py
  python ex_3_profile.py --log ex_3_profile.jsonl --session latest
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

DEFAULT_LOG = Path(__file__).resolve().parent / "ex_3_profile.jsonl"


def load_events(path: Path) -> list[dict]:
    events = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            events.append(json.loads(line))
    return events


def latest_session_id(events: list[dict]) -> str | None:
    for ev in reversed(events):
        if ev.get("session_id"):
            return ev["session_id"]
    return None


def filter_session(events: list[dict], session_id: str | None) -> list[dict]:
    if session_id is None or session_id == "all":
        return events
    if session_id == "latest":
        sid = latest_session_id(events)
        if sid is None:
            return []
        return [e for e in events if e.get("session_id") == sid]
    return [e for e in events if e.get("session_id") == session_id]


def plot_time_breakdown(events: list[dict]):
    llm = sum(float(e.get("duration_s", 0)) for e in events if e["event"] == "llm_response")
    invoke = sum(float(e.get("invoke_s", 0)) for e in events if e["event"] == "tool_call")
    run = sum(float(e.get("run_s", 0)) for e in events if e["event"] == "tool_call")
    if llm + invoke + run <= 0:
        return

    labels = ["LLM response", "Tool invocation", "Tool run"]
    values = [llm, invoke, run]
    colors = ["#4C78A8", "#F58518", "#E45756"]

    fig, ax = plt.subplots(figsize=(7, 5), num="Time breakdown")
    ax.pie(
        values,
        labels=labels,
        autopct=lambda p: f"{p:.1f}%" if p > 0.5 else "",
        colors=colors,
        startangle=90,
    )
    ax.set_title("Total time breakdown")
    total = sum(values)
    ax.text(
        0,
        -1.25,
        f"total = {total:.2f}s  |  llm={llm:.2f}s  invoke={invoke*1000:.2f}ms  run={run:.2f}s",
        ha="center",
        fontsize=9,
    )
    fig.tight_layout()


def plot_timeline(events: list[dict]):
    """LLM and tool steps on separate panels so ms-scale tools stay readable."""
    llm_steps = []
    tool_steps = []
    for ev in events:
        if ev["event"] == "llm_response":
            llm_steps.append(
                {
                    "label": f"llm#{ev.get('turn', '?')}",
                    "duration_s": float(ev.get("duration_s", 0)),
                }
            )
        elif ev["event"] == "tool_call":
            tool_steps.append(
                {
                    "label": f"{len(tool_steps)+1}:{ev.get('tool', 'tool')}",
                    "invoke_ms": float(ev.get("invoke_s", 0)) * 1000,
                    "run_ms": float(ev.get("run_s", 0)) * 1000,
                }
            )

    if not llm_steps and not tool_steps:
        return

    n_rows = (1 if llm_steps else 0) + (1 if tool_steps else 0)
    fig_h = 0
    if llm_steps:
        fig_h += max(3.0, 0.32 * len(llm_steps) + 1.2)
    if tool_steps:
        fig_h += max(3.0, 0.32 * len(tool_steps) + 1.2)

    fig, axes = plt.subplots(
        n_rows,
        1,
        figsize=(11, fig_h),
        num="Step timeline",
        squeeze=False,
    )
    ax_i = 0

    if llm_steps:
        ax = axes[ax_i][0]
        ax_i += 1
        y = np.arange(len(llm_steps))
        durations = [s["duration_s"] for s in llm_steps]
        ax.barh(y, durations, color="#4C78A8", edgecolor="none")
        ax.set_yticks(y)
        ax.set_yticklabels([s["label"] for s in llm_steps], fontsize=8)
        ax.invert_yaxis()
        ax.set_xlabel("Duration (seconds)")
        ax.set_title("LLM response timeline")
        ax.grid(axis="x", alpha=0.3)
        for i, d in enumerate(durations):
            ax.text(d, i, f" {d:.2f}s", va="center", fontsize=7, color="#333")
        ax.set_xlim(left=0)

    if tool_steps:
        ax = axes[ax_i][0]
        y = np.arange(len(tool_steps))
        invoke = [s["invoke_ms"] for s in tool_steps]
        run = [s["run_ms"] for s in tool_steps]
        ax.barh(y, invoke, color="#F58518", edgecolor="none", label="invocation")
        ax.barh(y, run, left=invoke, color="#E45756", edgecolor="none", label="run")
        ax.set_yticks(y)
        ax.set_yticklabels([s["label"] for s in tool_steps], fontsize=8)
        ax.invert_yaxis()
        ax.set_xlabel("Duration (milliseconds)")
        ax.set_title("Tool call timeline (own scale)")
        ax.legend(loc="lower right", fontsize=8)
        ax.grid(axis="x", alpha=0.3)
        for i, (inv, rn) in enumerate(zip(invoke, run)):
            total = inv + rn
            if total > 0:
                label = f" {total:.2f}ms" if total < 1000 else f" {total/1000:.2f}s"
                ax.text(total, i, label, va="center", fontsize=7, color="#333")
        ax.set_xlim(left=0)

    fig.suptitle("Step timeline — separate scales for LLM vs tools", fontsize=11, y=1.01)
    fig.tight_layout()


def plot_tool_stacked(events: list[dict]):
    tools = [e for e in events if e["event"] == "tool_call"]
    if not tools:
        return

    labels = [f"{i+1}:{e.get('tool')}" for i, e in enumerate(tools)]
    invoke = [float(e.get("invoke_s", 0)) * 1000 for e in tools]
    run = [float(e.get("run_s", 0)) * 1000 for e in tools]

    x = np.arange(len(tools))
    fig, ax = plt.subplots(
        figsize=(max(7, 0.55 * len(tools) + 2), 5),
        num="Tool invoke vs run",
    )
    ax.bar(x, invoke, label="invocation", color="#F58518")
    ax.bar(x, run, bottom=invoke, label="run", color="#E45756")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Duration (milliseconds)")
    ax.set_title("Per tool-call: invocation vs run")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()


def plot_tool_averages(events: list[dict]):
    by_tool: dict[str, list[tuple[float, float]]] = defaultdict(list)
    for e in events:
        if e["event"] != "tool_call":
            continue
        by_tool[e.get("tool", "?")].append(
            (float(e.get("invoke_s", 0)) * 1000, float(e.get("run_s", 0)) * 1000)
        )
    if not by_tool:
        return

    names = sorted(by_tool)
    avg_invoke = [sum(v[0] for v in by_tool[n]) / len(by_tool[n]) for n in names]
    avg_run = [sum(v[1] for v in by_tool[n]) / len(by_tool[n]) for n in names]
    counts = [len(by_tool[n]) for n in names]

    x = np.arange(len(names))
    width = 0.35
    fig, ax = plt.subplots(figsize=(8, 5), num="Tool averages")
    ax.bar(x - width / 2, avg_invoke, width, label="avg invocation", color="#F58518")
    ax.bar(x + width / 2, avg_run, width, label="avg run", color="#E45756")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{n}\n(n={c})" for n, c in zip(names, counts)])
    ax.set_ylabel("Duration (milliseconds)")
    ax.set_title("Average tool timings by name")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()


def plot_llm_calls(events: list[dict]):
    llms = [e for e in events if e["event"] == "llm_response"]
    if not llms:
        return

    turns = [e.get("turn", i + 1) for i, e in enumerate(llms)]
    durations = [float(e.get("duration_s", 0)) for e in llms]
    tokens_in = [e.get("input_tokens") or 0 for e in llms]
    tokens_out = [e.get("output_tokens") or 0 for e in llms]

    fig, axes = plt.subplots(2, 1, figsize=(9, 7), sharex=True, num="LLM latency & tokens")

    axes[0].bar(range(len(llms)), durations, color="#4C78A8")
    axes[0].set_ylabel("Duration (s)")
    axes[0].set_title("LLM response latency by call")
    axes[0].grid(axis="y", alpha=0.3)

    axes[1].bar(range(len(llms)), tokens_in, label="input tokens", color="#72B7B2")
    axes[1].bar(
        range(len(llms)),
        tokens_out,
        bottom=tokens_in,
        label="output tokens",
        color="#54A24B",
    )
    axes[1].set_ylabel("Tokens")
    axes[1].set_xlabel("LLM call index")
    axes[1].set_xticks(range(len(llms)))
    axes[1].set_xticklabels([str(t) for t in turns])
    axes[1].legend()
    axes[1].grid(axis="y", alpha=0.3)

    fig.tight_layout()


def print_summary(events: list[dict]):
    llm = [e for e in events if e["event"] == "llm_response"]
    tools = [e for e in events if e["event"] == "tool_call"]
    llm_t = sum(float(e.get("duration_s", 0)) for e in llm)
    inv_t = sum(float(e.get("invoke_s", 0)) for e in tools)
    run_t = sum(float(e.get("run_s", 0)) for e in tools)
    total = llm_t + inv_t + run_t

    sid = events[0].get("session_id") if events else "?"
    print(f"session: {sid}")
    print(f"events: {len(events)}  llm_calls: {len(llm)}  tool_calls: {len(tools)}")
    if total > 0:
        print(
            f"time: total={total:.2f}s  llm={llm_t:.2f}s ({100*llm_t/total:.1f}%)  "
            f"invoke={inv_t*1000:.2f}ms  run={run_t:.2f}s ({100*run_t/total:.1f}%)"
        )


def main():
    parser = argparse.ArgumentParser(description="Interactive profile graphs from ex_3 JSONL log")
    parser.add_argument("--log", type=Path, default=DEFAULT_LOG)
    parser.add_argument(
        "--session",
        default="latest",
        help="'latest', 'all', or a specific session_id",
    )
    args = parser.parse_args()

    if not args.log.exists():
        raise SystemExit(f"No profile log at {args.log}. Run ex_3.py first.")

    events = filter_session(load_events(args.log), args.session)
    if not events:
        raise SystemExit(f"No events for session={args.session!r} in {args.log}")

    print_summary(events)

    plot_time_breakdown(events)
    plot_timeline(events)
    plot_tool_stacked(events)
    plot_tool_averages(events)
    plot_llm_calls(events)

    print("Opening interactive graph windows (close them to exit)...")
    plt.show()


if __name__ == "__main__":
    main()
