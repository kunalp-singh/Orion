"""Raw JSONL experiment logging and CSV export."""

import csv
import json
from pathlib import Path
from typing import Any


class ExperimentLogger:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, record: dict[str, Any]) -> None:
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, default=str) + "\n")

    def export_csv(self, path: str | Path) -> Path:
        rows = [json.loads(line) for line in self.path.read_text(encoding="utf-8").splitlines() if line]
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        keys = sorted({key for row in rows for key in row})
        with output.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=keys)
            writer.writeheader()
            writer.writerows({key: json.dumps(row.get(key)) if isinstance(row.get(key), (dict, list)) else row.get(key) for key in keys} for row in rows)
        return output
