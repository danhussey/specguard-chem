# Audits And Reproducibility

Strict validation passes with valid=true and num_errors=0 for 656 tasks and 118 bundles. The validator checks schema fields, oracle evidence, repair and audit semantics, explicit contradiction certificates, boundary/invariance groups, split leakage, protocol access, and safety-scope text.

Prompt leakage audits check 656 actual model prompts and report zero oracle field leaks, zero literal witness leaks, zero label leaks, and zero audit_accept/audit_reject visible-name leaks.

Oracle-scrambling audits deep-copy raw tasks, randomize hidden oracle fields, rebuild public views, and confirm that public views and non-oracle deterministic baseline outputs are unchanged.

Negative controls cover the corrupted release cases, including output-schema/action-set mismatches and public micro-window prompts, and show that strict validation fails on representative broken invariants.

Clean-clone reproduction passed from the anonymous archive. The reproduction runs pytest, strict validation, prompt leakage audit, oracle scrambling audit, test baselines, and paper figure generation from the archive contents.

The only rc2-local yellow flag is hosted anonymous dataset URL accessibility. After upload, run `scripts/finalize_hosted_url.py` and `scripts/check_paper_consistency.py --mode final`.
