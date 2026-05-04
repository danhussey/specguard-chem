# Appendix: Schema

The v1 task schema requires task_id, suite, bundle_id, task_type, task_family, protocol, prompt_template, rendered_agent_input, agent_visible_hash, input, spec_id, spec_instance_hash, scoring, expected, expected_action, oracle_type, evidence, budgets, and generation.

The task schema also carries structural difficulty metadata through `difficulty_tags` and `challenge_slice`. These fields are assigned from task/spec/oracle structure and are not based on baseline performance.

See `tasks/schema.json` and `src/specguard_chem/config.py`.
