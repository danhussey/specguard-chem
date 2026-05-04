# Appendix: Prompt Leakage

The prompt leakage audit builds the exact prompt string sent to each configured adapter for every sgchem_v1.0 task.

It fails on hidden oracle field names, literal witness values, proof/certificate strings, split identifiers, internal task IDs, bundle IDs, and answer-encoding internal task labels. The current audit checks 688 prompts and reports zero leaks.

See `benchmarks/releases/sgchem_v1.0/audits/model_prompt_leakage_report.md`.
