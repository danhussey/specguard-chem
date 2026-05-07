# Appendix: Oracle Scrambling

Oracle scrambling tests whether public views and non-oracle deterministic baseline outputs are invariant to hidden answer fields.

The audit randomizes expected, expected_action, evidence, oracle_type, witness/proof/certificate fields, split, task_id, and bundle_id, then rebuilds public views. The current release passes: public views are byte-identical under scrambling and non-oracle baseline outputs are identical.

See `benchmarks/releases/sgchem_v1.0/audits/oracle_scrambling_report.md`.
