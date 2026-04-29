#!/usr/bin/env python3
"""
Convert constraint JSON (e.g. jewelry-check.json) to prior file CSV for the Streamlit app.

Supports both formats:
- Array format: [src, tgt, comment, type]
- Object format: {source: [...], target: [...], description: str, type: str}

Usage:
    python scripts/json_to_prior_csv.py input.json [output.csv]
"""

import csv
import json
import sys
from pathlib import Path


def bf_relationship(mode: str) -> str:
    """Map recoding mode to B/F Relationship."""
    return {"SS": "Single to Single", "SM": "Single to Multi",
            "MS": "Multi to Single", "MM": "Multi to Multi"}.get(mode, "Single to Single")


def to_source(val) -> str:
    """Format source/target for CSV (list as ['a','b'] string)."""
    if isinstance(val, list):
        return "[" + ", ".join(f"'{x}'" for x in val) + "]"
    return str(val)


def first_or_list(val):
    """Get first element if single, else return as list for to_source."""
    if isinstance(val, list):
        return val[0] if len(val) == 1 else val
    return val


def main():
    if len(sys.argv) < 2:
        print(__doc__, file=sys.stderr)
        sys.exit(1)

    input_path = Path(sys.argv[1])
    output_path = Path(sys.argv[2]) if len(sys.argv) > 2 else input_path.with_suffix(".csv")

    with open(input_path, encoding="utf-8") as f:
        data = json.load(f)

    constraints = data.get("constraints", data)
    rows = []
    row_id = 1
    cols = ["ID", "Target", "Source", "Constraint", "B/F Relationship", "Comment", "Is Implemented", "Custom Query"]

    fix_comments = {
        ("hQuota", "D29"): "Block D29 if hQuota ≠ Fine Jewelry",
        ("hQuota", "D38"): "Block D38 if hQuota ≠ Fine Jewelry",
        ("hQuota", "F07"): "Block F07 if hQuota ≠ Fine Jewelry",
    }

    def get_comment(src, tgt, desc):
        key = (first_or_list(src), first_or_list(tgt)) if isinstance(src, list) else (src, tgt)
        return fix_comments.get(key, desc or "")

    # BF_SS
    for c in constraints.get("BF_SS", []):
        if isinstance(c, dict):
            src = c["source"][0] if c["source"] else ""
            tgt = c["target"][0] if c["target"] else ""
            comment = get_comment(c["source"], c["target"], c.get("description", ""))
        else:
            src, tgt, comment = c[0], c[1], c[2] if len(c) > 2 else ""
            comment = fix_comments.get((src, tgt), comment)
        rows.append([row_id, tgt, src, "Block/Force", "Single to Single", comment, "Yes", ""])
        row_id += 1

    # BF_SM
    for c in constraints.get("BF_SM", []):
        if isinstance(c, dict):
            src = c["source"][0] if c["source"] else ""
            tgt = c["target"]
            if isinstance(tgt, list):
                tgt = to_source(tgt) if len(tgt) > 1 else tgt[0]
            comment = c.get("description", "")
        else:
            src, tgt, comment = c[0], c[1], c[2] if len(c) > 2 else ""
        rows.append([row_id, tgt, src, "Block/Force", "Single to Multi", comment or "", "Yes", ""])
        row_id += 1

    # BF_MM
    for c in constraints.get("BF_MM", []):
        if isinstance(c, dict):
            src = to_source(c["source"])
            tgt = to_source(c["target"])
            comment = c.get("description", "")
        else:
            src = c[0] if isinstance(c[0], list) else c[0]
            tgt = c[1] if isinstance(c[1], list) else c[1]
            comment = c[3] if len(c) > 3 else ""
            src, tgt = to_source(src), to_source(tgt)
        rows.append([row_id, tgt, src, "Parallel Piping", "", comment or "", "Yes", ""])
        row_id += 1

    # NOTAs
    for item in constraints.get("NOTAs", []):
        if isinstance(item, dict):
            cols_list = item["columns"]
            nota = item["exclusive_column"]
            if nota == "D23r99" and cols_list and cols_list[0].startswith("D34r"):
                nota = "D34r99"  # Fix typo
            src = to_source(cols_list)
        else:
            prefix, nota = item[0], item[1]
            if prefix == "D34r" and nota == "D23r99":
                nota = "D34r99"
            src = prefix
        rows.append([row_id, nota, src, "None of the above", "", "", "Yes", ""])
        row_id += 1

    # Recodings
    for item in constraints.get("recodings", []):
        if isinstance(item, dict):
            src, tgt = item["source"], item["target"]
            mode = item.get("type", "SS")
            comment = item.get("description", "")
            src = to_source(src) if isinstance(src, list) else src
            tgt = first_or_list(tgt) if isinstance(tgt, list) else tgt
        else:
            src, tgt = item[0], item[1]
            mode = item[2] if len(item) > 2 else "SS"
            comment = item[3] if len(item) > 3 else ""
            src = to_source(src) if isinstance(src, list) else src
        rows.append([row_id, tgt, src, "Recoding", bf_relationship(mode), comment, "Yes", ""])
        row_id += 1

    # Uniqueness
    for item in constraints.get("uniqueness", []):
        if isinstance(item, dict):
            tgt = to_source(item["columns"])
        else:
            cols_val = item[0] if isinstance(item[0], list) else item
            tgt = to_source(cols_val) if isinstance(cols_val, list) else cols_val
        rows.append([row_id, tgt, "", "Uniqueness", "", "", "Yes", ""])
        row_id += 1

    # Count
    for item in constraints.get("count", []):
        if isinstance(item, dict):
            tgt = to_source(item["columns"])
            src = item.get("exclude_column", "")
        else:
            prefix = item[0]
            src = item[1] if len(item) > 1 else ""
            tgt = to_source(prefix) if isinstance(prefix, list) else prefix
        rows.append([row_id, tgt, src, "Count", "", "", "Yes", ""])
        row_id += 1

    # Custom
    for item in constraints.get("custom", []):
        if isinstance(item, dict):
            ctype = item.get("name", "")
            comment = item.get("description", "")
            code = item.get("df_check", "")
            impl = "No" if item.get("expected") is False else "Yes"
        else:
            ctype, comment = item[0], item[1]
            code = item[2] if len(item) > 2 else ""
            impl = "Yes" if (len(item) <= 3 or item[3]) else "No"
        rows.append([row_id, "", "", ctype, "", comment, impl, code])
        row_id += 1

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(cols)
        w.writerows(rows)

    print(f"Converted {input_path} -> {output_path} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
