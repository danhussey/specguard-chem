# Artifact Integrity Gates v2

| gate | status | source |
| --- | --- | --- |
| strict validation | pass | validate_dataset_strict.log |
| prompt leakage audit | pass | model_prompt_leakage_audit.log |
| oracle scrambling negative controls | pass | oracle_scrambling_audit.log |
| split leakage | collected | release audits |
| clean-clone reproduction | not run | not requested in local run |
| Croissant validation | pass | validate_croissant.log |
| hosted URL preflight | pass | neurips_ed_preflight.log |
