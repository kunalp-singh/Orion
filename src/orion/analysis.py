"""Analysis and plotting for raw Orion JSONL experiment records."""

from collections import Counter, defaultdict
import json
from pathlib import Path
from typing import Any


def load_records(paths: list[str | Path]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for path in paths:
        source = Path(path)
        if not source.exists():
            raise FileNotFoundError(f"Experiment log does not exist: {source}")
        records.extend(json.loads(line) for line in source.read_text(encoding="utf-8").splitlines() if line)
    if not records:
        raise ValueError("Experiment logs contain no records")
    return records


def generate_plots(paths: list[str | Path], output_dir: str | Path) -> list[Path]:
    try:
        import matplotlib.pyplot as plt
    except ImportError as error:
        raise RuntimeError("Plot generation requires matplotlib; install the dev dependencies") from error

    records = load_records(paths)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    created: list[Path] = []

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        if record.get("step") is not None:
            grouped[record.get("policy") or "unknown"].append(record)
    if not grouped:
        raise ValueError("Experiment logs contain no completed step records")

    def save(name: str) -> None:
        path = output / name
        plt.tight_layout()
        plt.savefig(path)
        plt.close()
        created.append(path)

    for policy, rows in grouped.items():
        plt.plot([row["step"] for row in rows], [row["context_tokens_after"] or 0 for row in rows], label=policy)
    plt.xlabel("Step")
    plt.ylabel("Context tokens")
    plt.legend()
    save("context_tokens_vs_step.png")

    for policy, rows in grouped.items():
        cumulative = []
        total = 0
        for row in rows:
            total += row["total_tokens"] or 0
            cumulative.append(total)
        plt.plot([row["step"] for row in rows], cumulative, label=policy)
    plt.xlabel("Step")
    plt.ylabel("Cumulative tokens")
    plt.legend()
    save("cumulative_tokens_vs_step.png")

    latency_by_policy = {
        policy: sum(row["latency_ms"] or 0 for row in rows) / len(rows)
        for policy, rows in grouped.items()
    }
    plt.bar(list(latency_by_policy), list(latency_by_policy.values()))
    plt.ylabel("Average latency (ms)")
    save("average_latency_by_policy.png")

    rule_rows = [
        row
        for row in records
        if row.get("step") is not None
        and row.get("action") is not None
        and row.get("policy") in {"rule_based", "rule_based_orion"}
    ]
    actions = Counter(row["action"] for row in rule_rows)
    if actions:
        plt.bar(list(actions), list(actions.values()))
    plt.ylabel("Count")
    plt.xticks(rotation=30)
    save("rule_based_action_distribution.png")

    totals = {policy: sum(row["total_tokens"] or 0 for row in rows) for policy, rows in grouped.items()}
    plt.bar(list(totals), list(totals.values()))
    plt.ylabel("Total tokens")
    save("total_tokens_by_policy.png")

    success = {
        policy: sum(row.get("success") is True for row in rows) / len(rows)
        for policy, rows in grouped.items()
        if rows
    }
    if success:
        plt.bar(list(success), list(success.values()))
        plt.ylabel("Per-step success rate")
        save("success_rate_by_policy.png")
    return created
