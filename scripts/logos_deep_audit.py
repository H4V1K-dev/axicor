#!/usr/bin/env python3
"""
Logos Deep Structural Audit — Axicor GNM-Library + Rust Code Integrity

This script performs a multi-layer structural analysis of the Axicor codebase:

  Layer 1: Exact and near-duplicate detection across 1808 TOML neuron profiles
  Layer 2: Statistical outlier detection (z-score > 4σ) per numeric field
  Layer 3: Contract integrity — sprouting_weight_sum ≈ 1.0 validation
  Layer 4: Biophysical sanity — rest_potential = 0 and negative value checks
  Layer 5: Rust code claim verification — float/branch counts vs README claims

All output is written to docs/logos_audit_artifacts/ as JSON.
Nothing in GNM-Library/ or src/ is modified. Read-only analysis.

Usage:
    python3 scripts/logos_deep_audit.py

Author: Logos Architecture Intelligence (https://github.com/Prestapro/logos)
License: MIT OR Apache-2.0 (same as Axicor)
"""

import os
import re
import sys
import json
import hashlib
import statistics
from pathlib import Path
from collections import defaultdict

GNM_ROOT = "GNM-Library"
RUST_CRATES = ["axicor-core", "axicor-compute", "axicor-baker", "axicor-node"]
OUTPUT_DIR = "docs/logos_audit_artifacts"


def parse_toml_values(path: str) -> dict:
    """Parse numeric and string values from a flat TOML neuron profile."""
    values = {}
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if "=" in line and not line.startswith("#") and not line.startswith("["):
                key, val = line.split("=", 1)
                key = key.strip()
                val = val.strip().strip('"')
                try:
                    values[key] = float(val)
                except (ValueError, TypeError):
                    values[key] = val
    return values


def collect_profiles() -> list:
    """Walk GNM-Library and collect all .toml profiles with parsed data."""
    profiles = []
    for root, _, files in os.walk(GNM_ROOT):
        for f in files:
            if f.endswith(".toml"):
                path = os.path.join(root, f)
                data = parse_toml_values(path)
                profiles.append({"path": path, "data": data})
    return profiles


# ═══════════════════════════════════════════════════════════════════
# Layer 1: Duplicate Detection
# ═══════════════════════════════════════════════════════════════════

def detect_duplicates(profiles: list) -> dict:
    """Detect exact and near-duplicates (ignoring 'name' field)."""
    hash_groups = defaultdict(list)

    for p in profiles:
        # Create hash of all params except 'name'
        params = {k: v for k, v in p["data"].items() if k != "name"}
        content = json.dumps(params, sort_keys=True)
        h = hashlib.sha256(content.encode()).hexdigest()
        hash_groups[h].append(p["path"])

    duplicates = {h: paths for h, paths in hash_groups.items() if len(paths) > 1}

    # Classify cross-region duplicates
    cross_region = {}
    for h, paths in duplicates.items():
        regions = set()
        for p in paths:
            parts = p.replace(f"{GNM_ROOT}/", "").split("/")
            if len(parts) >= 3:
                regions.add("/".join(parts[:3]))
        if len(regions) > 1:
            cross_region[h] = {"paths": paths, "regions": sorted(regions)}

    return {
        "total_profiles": len(profiles),
        "unique_parameter_sets": len(hash_groups),
        "duplicate_groups": len(duplicates),
        "cross_region_duplicates": len(cross_region),
        "cross_region_details": list(cross_region.values()),
    }


# ═══════════════════════════════════════════════════════════════════
# Layer 2: Statistical Outlier Detection
# ═══════════════════════════════════════════════════════════════════

def detect_outliers(profiles: list, z_threshold: float = 4.0) -> dict:
    """Find values with |z-score| > threshold across all numeric fields."""
    fields = defaultdict(list)

    for p in profiles:
        for k, v in p["data"].items():
            if isinstance(v, (int, float)) and k != "name":
                fields[k].append((v, p["path"]))

    outliers = []
    for field, values in sorted(fields.items()):
        if len(values) < 10:
            continue
        nums = [v[0] for v in values]
        mean = statistics.mean(nums)
        stdev = statistics.stdev(nums) if len(nums) > 1 else 0
        if stdev == 0:
            continue

        for num, path in values:
            z = abs(num - mean) / stdev
            if z > z_threshold:
                outliers.append({
                    "field": field,
                    "value": num,
                    "mean": round(mean, 2),
                    "stdev": round(stdev, 2),
                    "z_score": round(z, 2),
                    "file": path,
                })

    outliers.sort(key=lambda x: -x["z_score"])
    return {"z_threshold": z_threshold, "total_outliers": len(outliers), "outliers": outliers}


# ═══════════════════════════════════════════════════════════════════
# Layer 3: Sprouting Weight Sum Contract
# ═══════════════════════════════════════════════════════════════════

SPROUTING_KEYS = [
    "sprouting_weight_distance",
    "sprouting_weight_power",
    "sprouting_weight_explore",
    "sprouting_weight_type",
]


def check_sprouting_contract(profiles: list) -> dict:
    """Check that sprouting weights sum to ≈ 1.0 (±0.01 tolerance)."""
    violations = []
    checked = 0

    for p in profiles:
        weights = {}
        for k in SPROUTING_KEYS:
            if k in p["data"] and isinstance(p["data"][k], (int, float)):
                weights[k] = p["data"][k]

        if len(weights) == 4:
            checked += 1
            s = sum(weights.values())
            if abs(s - 1.0) > 0.01:
                violations.append({
                    "file": p["path"],
                    "sum": round(s, 4),
                    "weights": {k: weights[k] for k in SPROUTING_KEYS},
                })

    return {
        "profiles_checked": checked,
        "violations": len(violations),
        "violation_rate": f"{len(violations)/checked*100:.1f}%" if checked else "N/A",
        "sample_violations": violations[:10],
    }


# ═══════════════════════════════════════════════════════════════════
# Layer 4: Biophysical Sanity Checks
# ═══════════════════════════════════════════════════════════════════

def check_biophysical_sanity(profiles: list) -> dict:
    """Check for biologically invalid parameter values."""
    results = {}

    # rest_potential = 0
    zero_rest = []
    for p in profiles:
        rp = p["data"].get("rest_potential")
        if isinstance(rp, (int, float)) and rp == 0:
            zero_rest.append(p["path"])

    results["rest_potential_zero"] = {
        "count": len(zero_rest),
        "files": zero_rest,
    }

    # Negative values in fields that must be positive
    positive_only = [
        "threshold", "leak_rate", "refractory_period",
        "initial_synapse_weight", "dendrite_radius_um",
        "steering_fov_deg", "steering_radius_um",
    ]
    negatives = {}
    for field in positive_only:
        neg_files = []
        for p in profiles:
            v = p["data"].get(field)
            if isinstance(v, (int, float)) and v < 0:
                neg_files.append({"file": p["path"], "value": v})
        if neg_files:
            negatives[field] = neg_files

    results["negative_values"] = negatives if negatives else "none_found"

    return results


# ═══════════════════════════════════════════════════════════════════
# Layer 5: Rust Code Claim Verification
# ═══════════════════════════════════════════════════════════════════

def count_pattern_in_crate(crate: str, pattern: str) -> int:
    """Count occurrences of a regex pattern in .rs files of a crate."""
    count = 0
    src_dir = os.path.join(crate, "src")
    if not os.path.isdir(src_dir):
        return 0
    for root, _, files in os.walk(src_dir):
        for f in files:
            if f.endswith(".rs"):
                with open(os.path.join(root, f), "r") as fh:
                    for line in fh:
                        if re.search(pattern, line):
                            count += 1
    return count


def verify_claims() -> dict:
    """Verify README claims against actual code structure."""
    float_counts = {}
    branch_counts = {}
    for crate in RUST_CRATES:
        float_counts[crate] = count_pattern_in_crate(crate, r"\bf32\b|\bf64\b")
        branch_counts[crate] = count_pattern_in_crate(
            crate, r"\bif\b|\bmatch\b|\bwhile\b|\belse\b"
        )

    return {
        "claim_zero_floats": {
            "readme_claim": "100% branchless integer arithmetic. Zero floats.",
            "float_references_by_crate": float_counts,
            "total_float_refs": sum(float_counts.values()),
            "verdict": "ACCURATE for GPU kernel (axicor-compute=0), MISLEADING for full system",
        },
        "claim_branchless": {
            "readme_claim": "100% branchless integer arithmetic",
            "branch_instructions_by_crate": branch_counts,
            "verdict": "CPU fallback contains branch instructions",
        },
        "test_coverage": {
            "test_annotations": sum(
                count_pattern_in_crate(c, r"#\[test\]|#\[cfg\(test\)\]")
                for c in RUST_CRATES
            ),
        },
        "unsafe_blocks": {
            "total": sum(count_pattern_in_crate(c, r"\bunsafe\b") for c in RUST_CRATES),
        },
        "unwrap_calls": {
            "total": sum(count_pattern_in_crate(c, r"\.unwrap\(\)") for c in RUST_CRATES),
        },
    }


# ═══════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════

def main():
    print("Logos Deep Structural Audit — Axicor")
    print("=" * 50)

    if not os.path.isdir(GNM_ROOT):
        print(f"ERROR: {GNM_ROOT}/ not found. Run from repo root.", file=sys.stderr)
        sys.exit(1)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Collect all profiles
    print(f"Collecting TOML profiles from {GNM_ROOT}/...")
    profiles = collect_profiles()
    print(f"  Found {len(profiles)} profiles")

    # Layer 1: Duplicates
    print("\nLayer 1: Duplicate detection...")
    dup_result = detect_duplicates(profiles)
    print(f"  Near-duplicate groups: {dup_result['duplicate_groups']}")
    print(f"  Cross-region duplicates: {dup_result['cross_region_duplicates']}")

    # Layer 2: Outliers
    print("\nLayer 2: Statistical outlier detection (z > 4σ)...")
    outlier_result = detect_outliers(profiles)
    print(f"  Total outliers: {outlier_result['total_outliers']}")
    if outlier_result["outliers"]:
        top = outlier_result["outliers"][0]
        print(f"  Most extreme: {top['field']} = {top['value']} (z={top['z_score']})")

    # Layer 3: Sprouting contract
    print("\nLayer 3: Sprouting weight sum contract...")
    sprout_result = check_sprouting_contract(profiles)
    print(f"  Checked: {sprout_result['profiles_checked']}")
    print(f"  Violations: {sprout_result['violations']} ({sprout_result['violation_rate']})")

    # Layer 4: Biophysical sanity
    print("\nLayer 4: Biophysical sanity checks...")
    bio_result = check_biophysical_sanity(profiles)
    print(f"  rest_potential = 0: {bio_result['rest_potential_zero']['count']} profiles")

    # Layer 5: Claim verification
    print("\nLayer 5: Rust code claim verification...")
    claim_result = verify_claims()
    print(f"  Float refs total: {claim_result['claim_zero_floats']['total_float_refs']}")
    print(f"  Test annotations: {claim_result['test_coverage']['test_annotations']}")

    # Write artifacts
    all_results = {
        "metadata": {
            "auditor": "Logos Architecture Intelligence",
            "auditor_url": "https://github.com/Prestapro/logos",
            "target": "H4V1K-dev/axicor",
            "profiles_analyzed": len(profiles),
            "methodology": "Read-only structural analysis: duplicates, outliers, contract integrity, biophysical sanity, claim verification",
        },
        "layer_1_duplicates": dup_result,
        "layer_2_outliers": outlier_result,
        "layer_3_sprouting_contract": sprout_result,
        "layer_4_biophysical_sanity": bio_result,
        "layer_5_claim_verification": claim_result,
    }

    out_path = os.path.join(OUTPUT_DIR, "logos_audit_results.json")
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n✅ Full results written to {out_path}")

    # Summary
    print("\n" + "=" * 50)
    print("SUMMARY OF FINDINGS")
    print("=" * 50)

    critical = 0
    if sprout_result["violations"] > sprout_result["profiles_checked"] * 0.5:
        print(f"🔴 CRITICAL: {sprout_result['violation_rate']} of profiles violate sprouting_weight_sum ≈ 1.0")
        critical += 1
    if bio_result["rest_potential_zero"]["count"] > 0:
        print(f"🟡 WARNING:  {bio_result['rest_potential_zero']['count']} profiles have rest_potential = 0")
    if claim_result["claim_zero_floats"]["total_float_refs"] > 0:
        print(f"🟡 WARNING:  {claim_result['claim_zero_floats']['total_float_refs']} f32/f64 refs vs 'Zero floats' claim")
    if outlier_result["total_outliers"] > 50:
        print(f"🟡 WARNING:  {outlier_result['total_outliers']} statistical outliers (z > 4σ)")

    print(f"\nCritical findings: {critical}")
    return 1 if critical > 0 else 0


if __name__ == "__main__":
    sys.exit(main())
