"""
Checkpoint-style JSON I/O.

Every step reads its input file, reads its OWN output file (if it already
exists), skips any record whose id is already present in the output, does
work only for the remaining records, and writes results incrementally
(atomic write after every single record) so a crash or manual stop never
loses completed work and a re-run always continues from where it left off.
"""
import json
import os
import tempfile
from typing import Dict, List


def load_json_list(path: str) -> List[dict]:
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        content = f.read().strip()
        if not content:
            return []
        return json.loads(content)


def _atomic_write(path: str, data) -> None:
    dir_ = os.path.dirname(path)
    os.makedirs(dir_, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(dir=dir_, prefix=".tmp_", suffix=".json")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        os.replace(tmp_path, path)
    except Exception:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise


def save_json_list(path: str, records: List[dict]) -> None:
    _atomic_write(path, records)


def done_ids(records: List[dict], key: str = "id") -> set:
    return {r[key] for r in records if key in r}


class ResumableWriter:
    """
    Wraps an output file: loads what's already there, exposes which ids are
    already done, and lets the caller append one record at a time with an
    immediate atomic flush to disk (so partial progress within a step also
    survives interruption).
    """

    def __init__(self, path: str, key: str = "id"):
        self.path = path
        self.key = key
        self.records: List[dict] = load_json_list(path)
        self._done = done_ids(self.records, key)

    def is_done(self, record_id: str) -> bool:
        return record_id in self._done

    def append(self, record: dict) -> None:
        record_id = record[self.key]
        if record_id in self._done:
            # overwrite (e.g. re-processing after a manual fix)
            self.records = [r for r in self.records if r.get(self.key) != record_id]
        self.records.append(record)
        self._done.add(record_id)
        save_json_list(self.path, self.records)

    def all(self) -> List[dict]:
        return self.records

    def passed(self, status_field: str, passed_values=("pass", "chain_complete", "accept")) -> List[dict]:
        return [r for r in self.records if r.get(status_field) in passed_values]
