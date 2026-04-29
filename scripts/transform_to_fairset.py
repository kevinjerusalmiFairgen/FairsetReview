#!/usr/bin/env python3
"""
Transform constraint JSON files (e.g. jewelry-check.json) into Fairset Review format.

Fairset expects a flat JSON with top-level keys: BF_SS, BF_SM, BF_MM, NOTAs,
recodings, uniqueness, count, custom, etc. Each value is an array of constraints.

Usage:
    python scripts/transform_to_fairset.py input.json [output.json]
"""

import json
import sys
from pathlib import Path


def transform_to_fairset(data: dict) -> dict:
    """
    Transform a constraints JSON (possibly nested) into Fairset Review format.
    
    Handles:
    - Nested structure: data['constraints'] -> flat structure
    - Adds is_implemented (true) where Fairset expects it for BF_SS, BF_SM, BF_MM
    - Fixes NOTA typo: D34r constraint had D23r99, correct to D34r99
    """
    # Extract constraints from nested structure if present
    if "constraints" in data:
        constraints = data["constraints"].copy()
    else:
        constraints = {k: v for k, v in data.items() 
                       if k in ("BF_SS", "BF_SM", "BF_MM", "BF_MS", "NOTAs", 
                                "recodings", "uniqueness", "count", "custom",
                                "custom_query", "AOTAs", "BF_SM_Grid", "BF_Mixed_Type", "NOTAs_grid")}

    result = {}

    # BF_SS: [col1, col2, detail, block_force] -> add is_implemented
    if "BF_SS" in constraints:
        result["BF_SS"] = []
        for c in constraints["BF_SS"]:
            row = list(c)
            if len(row) == 4:
                row.append(True)
            result["BF_SS"].append(row)

    # BF_SM: same pattern
    if "BF_SM" in constraints:
        result["BF_SM"] = []
        for c in constraints["BF_SM"]:
            row = list(c)
            if len(row) == 4:
                row.append(True)
            result["BF_SM"].append(row)

    # BF_MM: [prefix1, prefix2, cols_drop, detail, block_force] -> add is_implemented
    if "BF_MM" in constraints:
        result["BF_MM"] = []
        for c in constraints["BF_MM"]:
            row = list(c)
            if len(row) == 5:
                row.append(True)
            result["BF_MM"].append(row)

    # BF_MS, NOTAs, recodings, uniqueness, count: pass through
    for key in ("BF_MS", "NOTAs", "recodings", "uniqueness", "count", 
                "AOTAs", "BF_SM_Grid", "BF_Mixed_Type", "NOTAs_grid", "custom_query"):
        if key in constraints:
            result[key] = constraints[key]

    # Fix NOTA typo: D34r with D23r99 -> D34r99
    if "NOTAs" in result:
        for item in result["NOTAs"]:
            if len(item) >= 2 and item[0] == "D34r" and item[1] == "D23r99":
                item[1] = "D34r99"
                break

    # Fix descriptions with copy-paste errors (D15 -> correct target)
    if "BF_SS" in result:
        fixes = {
            ("hQuota", "D29"): "Block D29 if hQuota ≠ Fine Jewelry",
            ("hQuota", "D38"): "Block D38 if hQuota ≠ Fine Jewelry",
            ("hQuota", "F07"): "Block F07 if hQuota ≠ Fine Jewelry",
        }
        for c in result["BF_SS"]:
            key = (c[0], c[1])
            if key in fixes:
                c[2] = fixes[key]

    # custom: ensure 4 elements [type, desc, code, is_implemented]
    if "custom" in constraints:
        result["custom"] = []
        for c in constraints["custom"]:
            row = list(c)
            while len(row) < 4:
                row.append(True)
            result["custom"].append(row)

    # recodings: fix "contaisn" typo
    if "recodings" in result:
        for c in result["recodings"]:
            if len(c) >= 4 and "contaisn" in str(c[3]):
                c[3] = str(c[3]).replace("contaisn", "contains")

    return result


def main():
    if len(sys.argv) < 2:
        print(__doc__, file=sys.stderr)
        sys.exit(1)

    input_path = Path(sys.argv[1])
    output_path = Path(sys.argv[2]) if len(sys.argv) > 2 else input_path.with_stem(
        input_path.stem + "_fairset"
    )

    with open(input_path, encoding="utf-8") as f:
        data = json.load(f)

    result = transform_to_fairset(data)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=4, ensure_ascii=False)

    print(f"Transformed {input_path} -> {output_path}")


if __name__ == "__main__":
    main()
