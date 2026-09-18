# Archived Gate C evidence (2026-09-18)

Historical Gate C decisions. **Not a modelvalidation certificate**: no `certificate.json`,
no `anchors.json`, no admission under the shared framework. Kept so the numbers that once
sat inside `quantark/intraday/evidence/` remain readable after the runtime stopped shipping
them (spec `docs/superpowers/specs/2026-09-18-intraday-modelvalidation-certification-design.md`, section 9).

| File | What it is |
|---|---|
| `gate_c_results.json` | 3432 price cells, schema `intraday-gate-c/1`, run at the revision its `git_sha` names |
| `gate_c_greeks.json` | 50 demonstrated Greek families, schema `intraday-gate-c-greeks/2` |
| `source-records/` | raw `.jsonl` rows and logs the sweeps produced, when present on the archiving machine |
| `manifest.json` | SHA-256 and byte count of every archived file, the producing commands and the admission rule |

The Gate C rule was `|route - ref| <= budget + 3 * ref_uncertainty` with a precheck
`3 * ref_uncertainty <= budget`. It admits a candidate whose true error could still exceed the
budget by the reference radius; the schema-2 rule does not. Nothing here is relabelled.
