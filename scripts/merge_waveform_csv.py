#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def merge_csv(inputs: list[Path], output: Path, key_columns: list[str]) -> dict[str, object]:
    output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    rows: list[dict[str, str]] = []
    seen: set[tuple[str, ...]] = set()
    source_counts: dict[str, int] = {}
    for path in inputs:
        if not path.exists() or path.stat().st_size == 0:
            source_counts[str(path)] = 0
            continue
        with path.open('r', encoding='utf-8', newline='') as fh:
            reader = csv.DictReader(fh)
            if not reader.fieldnames:
                source_counts[str(path)] = 0
                continue
            for name in reader.fieldnames:
                if name not in fieldnames:
                    fieldnames.append(name)
            count = 0
            for row in reader:
                key = tuple(str(row.get(col, '')) for col in key_columns)
                if key in seen:
                    continue
                seen.add(key)
                rows.append(dict(row))
                count += 1
            source_counts[str(path)] = count
    if not fieldnames:
        fieldnames = key_columns
    with output.open('w', encoding='utf-8', newline='') as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction='ignore')
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name, '') for name in fieldnames})
    meta = {
        'inputs': [str(path) for path in inputs],
        'output': str(output),
        'source_counts': source_counts,
        'row_count': len(rows),
        'key_columns': key_columns,
        'not_prediction': True,
    }
    output.with_suffix(output.suffix + '.metadata.json').write_text(json.dumps(meta, indent=2, sort_keys=True), encoding='utf-8')
    print(json.dumps(meta, indent=2, sort_keys=True))
    return meta


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--key-columns', required=True)
    parser.add_argument('inputs', nargs='+', type=Path)
    args = parser.parse_args()
    merge_csv(args.inputs, args.output, [item.strip() for item in args.key_columns.split(',') if item.strip()])


if __name__ == '__main__':
    main()
