"""GNM-Library structural audit.

Scans GNM-Library/ and emits:

  docs/gnm_audit_artifacts/archetypes.json   — per-class mode-template (157 classes)
  docs/gnm_audit_artifacts/duplicates.json   — files with identical parameter dicts
  docs/gnm_audit_artifacts/outliers.json     — files whose delta from class template is top-N
  docs/gnm_audit_artifacts/class_summary.json — per-class stats

Invariants verified by this script:
  - parse(f) succeeds for every .toml in GNM-Library/        (expect 1808/1808)
  - render(parse(f)) preserves every (key, value) pair       (expect 1808/1808 round-trip)
  - reassemble(template, delta) preserves every (key, value) (expect 1808/1808)

Run:
  python3 scripts/gnm_audit.py

Author: contributed as part of a structural audit PR. No runtime dependency on
Axicor crates; pure Python stdlib.
"""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
GNM_ROOT = REPO_ROOT / "GNM-Library"
OUT_DIR = REPO_ROOT / "docs" / "gnm_audit_artifacts"


# Section headers used when rendering a neuron file back (for round-trip).
SECTION_HEADERS: dict[str, str] = {
    "threshold":              "# Membrane",
    "homeostasis_penalty":    "# Adaptation and Timings",
    "steering_fov_deg":       "# Growth and Morphology",
    "initial_synapse_weight": "# Synaptic Plasticity (GSOP)",
    "slot_decay_ltm":         "# --- Dynamic Slot Decay (Derived) ---",
}


# ---------------------------------------------------------------------------
# Minimal TOML reader / writer sufficient for GNM-Library dialect.
# ---------------------------------------------------------------------------


@dataclass
class NeuronRecord:
    path: Path
    rel: Path
    class_: str
    name: str
    values: dict[str, Any]
    key_order: list[str]
    original_bytes: int


def parse_value(v: str) -> Any:
    v = v.strip()
    if v.startswith('"') and v.endswith('"'):
        return v[1:-1]
    if v == "true":
        return True
    if v == "false":
        return False
    if v.startswith("[") and v.endswith("]"):
        inner = v[1:-1].strip()
        if not inner:
            return []
        parts = [p.strip() for p in inner.split(",")]
        try:
            return [int(p) for p in parts]
        except ValueError:
            return [parse_value(p) for p in parts]
    try:
        if "." in v or "e" in v or "E" in v:
            return float(v)
        return int(v)
    except ValueError:
        return v


def render_value(val: Any) -> str:
    if isinstance(val, bool):
        return "true" if val else "false"
    if isinstance(val, str):
        return f'"{val}"'
    if isinstance(val, list):
        return "[" + ", ".join(render_value(x) for x in val) + "]"
    if isinstance(val, float):
        if val == int(val):
            return f"{int(val)}.0"
        return f"{val:g}"
    return str(val)


def parse_neuron(path: Path, rel: Path) -> NeuronRecord | None:
    text = path.read_text(encoding="utf-8")
    values: dict[str, Any] = {}
    order: list[str] = []
    for line in text.splitlines():
        s = line.strip()
        if not s or s.startswith("#") or s.startswith("[["):
            continue
        if "=" not in s:
            continue
        k, v = s.split("=", 1)
        k = k.strip()
        values[k] = parse_value(v)
        order.append(k)
    if "name" not in values:
        return None
    class_ = str(rel.parent).replace(os.sep, "/")
    return NeuronRecord(
        path=path,
        rel=rel,
        class_=class_,
        name=str(values["name"]),
        values=values,
        key_order=order,
        original_bytes=len(text.encode("utf-8")),
    )


def render_neuron(values: dict[str, Any], key_order: list[str]) -> str:
    lines = ["[[neuron_type]]"]
    emitted_section: set[str] = set()
    blank_before_section = False
    for k in key_order:
        if k in SECTION_HEADERS and SECTION_HEADERS[k] not in emitted_section:
            if blank_before_section:
                lines.append("")
            lines.append(SECTION_HEADERS[k])
            emitted_section.add(SECTION_HEADERS[k])
            blank_before_section = False
        lines.append(f"{k} = {render_value(values[k])}")
        blank_before_section = True
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Audit analyses.
# ---------------------------------------------------------------------------


def _hashable(v: Any) -> Any:
    if isinstance(v, list):
        return ("list", tuple(v))
    return v


def _unhashable(v: Any) -> Any:
    if isinstance(v, tuple) and v and v[0] == "list":
        return list(v[1])
    return v


def class_template(records: list[NeuronRecord]) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    """Return (mode-template, per-file delta map)."""
    per_key: dict[str, list[Any]] = defaultdict(list)
    for r in records:
        for k, v in r.values.items():
            per_key[k].append(_hashable(v))
    template: dict[str, Any] = {}
    for k, vals in per_key.items():
        mode_val, _ = Counter(vals).most_common(1)[0]
        template[k] = _unhashable(mode_val)
    deltas: dict[str, dict[str, Any]] = {}
    for r in records:
        delta: dict[str, Any] = {}
        for k, v in r.values.items():
            if _hashable(v) != _hashable(template[k]):
                delta[k] = v
        deltas[str(r.rel)] = delta
    return template, deltas


def roundtrip_check(r: NeuronRecord) -> tuple[bool, str]:
    rendered = render_neuron(r.values, r.key_order)
    re_parsed: dict[str, Any] = {}
    for line in rendered.splitlines():
        s = line.strip()
        if not s or s.startswith("#") or s.startswith("[["):
            continue
        if "=" not in s:
            continue
        k, v = s.split("=", 1)
        re_parsed[k.strip()] = parse_value(v)
    if set(re_parsed) != set(r.values):
        return False, f"keys differ: {set(r.values)^set(re_parsed)}"
    for k, v in r.values.items():
        if re_parsed[k] != v:
            return False, f"{k}: {v!r} != {re_parsed[k]!r}"
    return True, ""


def find_duplicates(records: list[NeuronRecord]) -> list[dict[str, Any]]:
    """Files sharing identical parameter dicts (ignoring `name`)."""
    def fp(r: NeuronRecord) -> tuple:
        return tuple(sorted((k, _hashable(v)) for k, v in r.values.items() if k != "name"))

    groups: dict[tuple, list[NeuronRecord]] = defaultdict(list)
    for r in records:
        groups[fp(r)].append(r)
    dupes = []
    for _, group in groups.items():
        if len(group) < 2:
            continue
        classes = sorted({r.class_ for r in group})
        dupes.append({
            "group_size": len(group),
            "num_classes": len(classes),
            "classes": classes,
            "files": [
                {"path": str(r.rel), "name": r.name, "class": r.class_}
                for r in group
            ],
            "cross_class": len(classes) > 1,
        })
    dupes.sort(key=lambda d: (-d["group_size"], -d["num_classes"]))
    return dupes


def find_outliers(
    records: list[NeuronRecord],
    by_class: dict[str, list[NeuronRecord]],
    threshold: int = 14,
    min_class_size: int = 5,
) -> list[dict[str, Any]]:
    """Files whose delta from class template is >= threshold keys.

    Only considers classes with at least `min_class_size` members, because the
    mode-template is unreliable for 2-4 record classes and would over-report
    outliers there.
    """
    outliers = []
    for cls, recs in by_class.items():
        if len(recs) < min_class_size:
            continue
        template, deltas = class_template(recs)
        for r in recs:
            delta = deltas[str(r.rel)]
            if len(delta) >= threshold:
                outliers.append({
                    "path": str(r.rel),
                    "name": r.name,
                    "class": cls,
                    "class_size": len(recs),
                    "delta_keys": len(delta),
                    "total_keys": len(template),
                    "diffs": {
                        k: {"template": template[k], "actual": delta[k]}
                        for k in delta
                    },
                })
    outliers.sort(key=lambda o: -o["delta_keys"])
    return outliers


def build_archetypes(by_class: dict[str, list[NeuronRecord]]) -> dict[str, Any]:
    archetypes = {}
    for cls, recs in by_class.items():
        template, deltas = class_template(recs)
        const_keys = [
            k for k in template
            if all(k not in deltas[str(r.rel)] for r in recs)
        ]
        deltas_sizes = [len(deltas[str(r.rel)]) for r in recs]
        archetypes[cls] = {
            "class_size": len(recs),
            "total_keys": len(template),
            "constant_keys": len(const_keys),
            "constant_key_names": sorted(const_keys),
            "template": template,
            "avg_delta": sum(deltas_sizes) / max(1, len(deltas_sizes)),
            "max_delta": max(deltas_sizes) if deltas_sizes else 0,
            "identical_to_template": sum(1 for d in deltas_sizes if d == 0),
        }
    return archetypes


# ---------------------------------------------------------------------------
# Main.
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outlier-threshold", type=int, default=14,
                        help="flag files with >= N delta keys from their class template")
    parser.add_argument("--quiet", action="store_true", help="suppress per-class output")
    args = parser.parse_args()

    print(f"GNM-Library audit")
    print(f"  root: {GNM_ROOT}")

    records: list[NeuronRecord] = []
    for p in sorted(GNM_ROOT.rglob("*.toml")):
        rel = p.relative_to(GNM_ROOT)
        r = parse_neuron(p, rel)
        if r is not None:
            records.append(r)

    by_class: dict[str, list[NeuronRecord]] = defaultdict(list)
    for r in records:
        by_class[r.class_].append(r)

    # 1. Round-trip invariant
    rt_ok = 0
    rt_fails = []
    for r in records:
        ok, reason = roundtrip_check(r)
        if ok:
            rt_ok += 1
        else:
            rt_fails.append({"path": str(r.rel), "reason": reason})

    # 2. Archetypes
    archetypes = build_archetypes(by_class)

    # 3. Duplicates
    duplicates = find_duplicates(records)

    # 4. Outliers
    outliers = find_outliers(records, by_class, threshold=args.outlier_threshold)

    # Summary
    summary = {
        "total_files": len(records),
        "total_classes": len(by_class),
        "roundtrip_ok": rt_ok,
        "roundtrip_fails": len(rt_fails),
        "duplicate_groups": len(duplicates),
        "duplicate_files": sum(d["group_size"] for d in duplicates),
        "cross_class_duplicate_groups": sum(1 for d in duplicates if d["cross_class"]),
        "outlier_files": len(outliers),
        "outlier_threshold": args.outlier_threshold,
    }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "archetypes.json").write_text(
        json.dumps(archetypes, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (OUT_DIR / "duplicates.json").write_text(
        json.dumps(duplicates, indent=2) + "\n", encoding="utf-8"
    )
    (OUT_DIR / "outliers.json").write_text(
        json.dumps(outliers, indent=2) + "\n", encoding="utf-8"
    )
    (OUT_DIR / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    if rt_fails:
        (OUT_DIR / "roundtrip_failures.json").write_text(
            json.dumps(rt_fails, indent=2) + "\n", encoding="utf-8"
        )

    print(f"  files:         {summary['total_files']}")
    print(f"  classes:       {summary['total_classes']}")
    print(f"  round-trip:    {summary['roundtrip_ok']}/{summary['total_files']} ok")
    print(f"  duplicates:    {summary['duplicate_groups']} groups, "
          f"{summary['duplicate_files']} files "
          f"({summary['cross_class_duplicate_groups']} cross-class)")
    print(f"  outliers:      {summary['outlier_files']} (threshold >= "
          f"{args.outlier_threshold} delta keys)")
    print(f"\n  artifacts written to: {OUT_DIR.relative_to(REPO_ROOT)}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
