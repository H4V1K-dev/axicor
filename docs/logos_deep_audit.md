# Logos Deep Structural Audit — Axicor GNM-Library + Rust Code

> **Auditor:** [Logos Architecture Intelligence](https://github.com/Prestapro/logos)
> **Date:** 2026-04-23
> **Scope:** 1,808 TOML neuron profiles + 4 Rust crates (~17K LOC)
> **Method:** Read-only structural analysis — no source files modified

## Summary

| Layer | Check | Result |
|-------|-------|--------|
| L1 | Near-duplicate detection | 8 groups, 7 cross-region |
| L2 | Statistical outliers (z > 4σ) | 138 outliers |
| L3 | **Sprouting weight contract** | **🔴 1,806/1,808 violations (99.9%)** |
| L4 | Biophysical sanity | 21 profiles with rest_potential = 0 |
| L5 | README claim verification | "Zero floats" accurate for GPU only |

---

## 🔴 Critical: Sprouting Weight Sum Contract Violation (99.9%)

`axicor-core/src/config/blueprints.rs:110` annotates sprouting weights as `(f32, sum ≈ 1.0)`.
The function `sprouting_weight_sum()` at line 192 sums 4 weight fields.
`axicor-baker/src/bake/sprouting.rs` uses these weights as raw multipliers in target selection — **no normalization**.

**1,806 out of 1,808** TOML profiles have `sum > 1.0`:

| Profile | dist | power | explore | type | **SUM** |
|---------|------|-------|---------|------|---------|
| L3/spiny/TemL/6 | 0.3 | 0.5 | 0.327 | 0.1 | **1.227** |
| L3/spiny/FroL/2 | 0.5 | 0.4 | 0.1 | 0.2 | **1.200** |

**Impact:** Sprouting scores are systematically inflated. All baked connectomes have biased axon target selection due to non-normalized weights. Note: `sprouting_weight_type` is marked `// TODO: does not work until Night Phase refactoring` but is still included in the sum.

**Suggested fix:** Either normalize weights at bake time (`score / weight_sum`) or re-calibrate all 1,808 TOML files.

---

## 🔴 GSOP Dead Zone Validator Is Empty

`axicor-baker/src/validator/checks.rs`:

```rust
fn validate_gsop_dead_zones(const_mem: &AxicorConstantMemory) {
    for (_type_idx, variant) in const_mem.variants.iter().enumerate() {
        if variant.gsop_potentiation > 0 {
            // Note: inertia_curve is now part of VariantParameters...
            // Leave as is if the interface allows.
        }
    }
}
```

This function **has no validation logic** — the if-body is empty. It should check that `(potentiation × inertia_curve[0]) >> 7 >= 1`, otherwise GSOP updates silently round to zero and learning is impossible for that neuron type.

**Suggested fix:** Implement the dead zone check or remove the function to avoid false confidence.

---

## 🟡 rest_potential = 0 in 21 Profiles

21 profiles have `rest_potential = 0` (mean across library: 10,802). Distribution:
- 16 are **VISp** (primary visual cortex) — systematic, not random
- 2 are Striatum medium_spiny (Rat/Mouse near-duplicate pair)
- 3 others in Cortex/L4, L23

This clustering in VISp profiles suggests a **data pipeline issue** with the Allen Institute visual cortex source data.

---

## 🟡 "Zero Floats" Claim Scope

README states: *"100% branchless integer arithmetic. Zero floats."*

| Crate | f32/f64 refs |
|-------|-------------|
| axicor-compute | **0** ✅ |
| axicor-node | **0** ✅ |
| axicor-core | 67 |
| axicor-baker | 121 |

The GPU hot-loop (Day Phase) IS integer-only — claim is accurate there. But `axicor-core/src/types.rs` defines `type Microns = f32` and all blueprint parameters use `f32`. The baker uses heavy float math for geometry. Suggested: scope the claim to "Day Phase / GPU kernel" explicitly.

---

## 🟡 7 Cross-Species Near-Duplicates

Profiles with identical parameters (ignoring name) between Rat and Mouse:

- Cerebellum: gabaergic Rat/1 = Mouse/476
- Thalamus: relay Rat/46 = Mouse/141
- Hippocampus: granule Rat/46 = Mouse/476
- Hippocampus: pyramidal Rat/141 = Mouse/775
- Striatum: medium_spiny Rat/1 = Mouse/2472

Additionally, a triple-duplicate: Thalamus/Mouse/gabaergic = Thalamus/Mouse/interneuron = Striatum/Mouse/gabaergic — three anatomically different cell types with identical parameters.

---

## ✅ Verified Working

- GPU kernel (axicor-compute) has zero float references — integer-only claim holds
- Deterministic RNG: wyhash/FNV-1a, ChaCha8Rng, MasterSeed — no time-based entropy
- 173 test annotations across the codebase
- `check_single_spike_in_flight` validator works correctly
- `check_layer_heights` and `check_composition_quotas` properly bail on sum ≠ 1.0

---

## Reproduction

```bash
python3 scripts/logos_deep_audit.py
# Results written to docs/logos_audit_artifacts/logos_audit_results.json
```

## Methodology

This audit used read-only structural analysis:
- **L1:** SHA-256 hash of parameter dicts (name-excluded) for duplicate detection
- **L2:** Z-score per numeric field across all 1,808 profiles
- **L3:** Sum validation of `sprouting_weight_{distance,power,explore,type}` against code contract
- **L4:** Range checks on biophysically constrained fields
- **L5:** `grep`-based pattern counting in .rs files vs README claims

No domain expertise in computational neuroscience was required or assumed. All findings are derived from **structural properties of the code and data** — the core philosophy of the [Logos](https://github.com/Prestapro/logos) audit methodology.
