# Audits And Reproducibility

Strict validation passes with valid=true and num_errors=0 for 688 tasks and 120 bundles. The validator checks schema fields, oracle evidence, repair and audit semantics, explicit contradiction certificates, boundary/invariance groups, split leakage, protocol access, and safety-scope text.

Prompt leakage audits check 688 actual model prompts and report zero oracle field leaks, zero literal witness leaks, zero label leaks, and zero audit_accept/audit_reject visible-name leaks.

Oracle-scrambling audits deep-copy raw tasks, randomize hidden oracle fields, rebuild public views, and confirm that public views and non-oracle deterministic baseline outputs are unchanged.

Negative controls cover 18 corrupted release cases and show that strict validation fails on representative broken invariants.

Clean-clone reproduction passed from the anonymous archive. The reproduction runs pytest, strict validation, prompt leakage audit, oracle scrambling audit, test baselines, and paper figure generation from the archive contents.

The only rc2-local yellow flag is hosted anonymous dataset URL accessibility. After upload, run `scripts/finalize_hosted_url.py` and `scripts/check_paper_consistency.py --mode final`.
