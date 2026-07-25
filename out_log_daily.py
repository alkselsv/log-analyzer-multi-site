"""Дневной out.log: один обычный файл, append за текущие сутки, сброс при смене дня."""

from __future__ import annotations

import csv
from datetime import datetime
from pathlib import Path
from typing import Iterable, List, Mapping, Sequence, Tuple, Union

RowInput = Union[Sequence[str], Mapping[str, str]]


def day_string(when: datetime | None = None) -> str:
    return (when or datetime.now()).strftime("%Y-%m-%d")


def _normalize_row(row: RowInput, fieldnames: Sequence[str]) -> List[str]:
    if isinstance(row, Mapping):
        return [str(row.get(name, "")) for name in fieldnames]
    return [str(value) for value in row]


def _write_header(out_path: Path, fieldnames: Sequence[str]) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as handle:
        csv.writer(handle).writerow(fieldnames)


def _file_belongs_to_day(out_path: Path, day: str) -> bool:
    if not out_path.is_file():
        return False
    if out_path.stat().st_size == 0:
        return False

    with out_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            record_date = str(row.get("date", "")).strip()
            return record_date.startswith(day)
    return False


def ensure_out_log_file(
    out_path: Path | str,
    fieldnames: Sequence[str],
    day: str | None = None,
) -> Path:
    """Обычный файл с заголовком; при смене суток перезаписывает файл."""
    out_path = Path(out_path)
    day = day or day_string()

    if not _file_belongs_to_day(out_path, day):
        _write_header(out_path, fieldnames)
    return out_path


def append_out_log(
    out_path: Path | str,
    fieldnames: Sequence[str],
    rows: Iterable[RowInput],
    day: str | None = None,
) -> Tuple[Path, int]:
    """Дописывает delta в out.log за текущий день; в новые сутки начинает файл заново."""
    out_path = Path(out_path)
    day = day or day_string()
    rows = list(rows)

    if not rows:
        ensure_out_log_file(out_path, fieldnames, day=day)
        return out_path, 0

    same_day = _file_belongs_to_day(out_path, day)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("a" if same_day else "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        if not same_day:
            writer.writerow(fieldnames)
        for row in rows:
            writer.writerow(_normalize_row(row, fieldnames))
    return out_path, len(rows)


def write_out_delta(
    delta_path: Path | str,
    fieldnames: Sequence[str],
    rows: Iterable[RowInput],
) -> int:
    delta_path = Path(delta_path)
    rows = list(rows)
    if not rows:
        if delta_path.exists():
            delta_path.unlink()
        return 0

    delta_path.parent.mkdir(parents=True, exist_ok=True)
    with delta_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            if isinstance(row, Mapping):
                writer.writerow({name: str(row.get(name, "")) for name in fieldnames})
            else:
                writer.writerow(dict(zip(fieldnames, _normalize_row(row, fieldnames))))
    return len(rows)


def reset_out_log_for_today(
    out_path: Path | str,
    fieldnames: Sequence[str],
    day: str | None = None,
) -> Path:
    """Сброс out.log к пустому файлу с заголовком (для cron в полночь)."""
    out_path = Path(out_path)
    _write_header(out_path, fieldnames)
    return out_path
