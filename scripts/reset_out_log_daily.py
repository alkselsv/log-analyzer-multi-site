#!/usr/bin/env python3
"""CLI для cron: сброс out.log к заголовку на новые сутки."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from out_log_daily import reset_out_log_for_today  # noqa: E402

DEVICE_FIELDNAMES = ["date", "session_id", "probability", "ip", "user_agent"]
COMBINED_FIELDNAMES = [
    "date",
    "device_type",
    "session_id",
    "probability",
    "ip",
    "user_agent",
]


def fieldnames_for_path(out_path: Path):
    if "mobile" in out_path.parts or "desktop" in out_path.parts:
        return DEVICE_FIELDNAMES
    return COMBINED_FIELDNAMES


def main(argv: list[str]) -> int:
    if argv == ["--all-sites"]:
        from site_registry import discover_sites

        paths: list[Path] = []
        for site in discover_sites():
            paths.extend(
                [
                    site.out_log_path,
                    site.site_dir / "outputs" / "mobile" / f"{site.site_id}.out.log",
                    site.site_dir / "outputs" / "desktop" / f"{site.site_id}.out.log",
                ]
            )
        argv = [str(path) for path in paths]

    if not argv:
        print(
            "Usage: reset_out_log_daily.py <out_log_path> ... | --all-sites",
            file=sys.stderr,
        )
        return 1

    for raw in argv:
        out_path = Path(raw)
        if not out_path.is_absolute():
            out_path = (ROOT / out_path).resolve()
        fields = fieldnames_for_path(out_path)
        reset_out_log_for_today(out_path, fields)
        print(f"[OK] reset {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
