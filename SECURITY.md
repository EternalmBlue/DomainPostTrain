# Security

Do not use this repository to publish secrets, private documents, internal system prompts, credentials, customer data, or license-restricted corpora.

`grpo.reward_judge.api_key` is the primary credential setting and is stored as plaintext YAML. Put real credentials only in the Git-ignored `configs/domain_post_training.local.yaml`; the tracked template must keep `api_key: null`. A Git ignore rule prevents accidental discovery, not deliberate staging, so verify the staged snapshot before every commit and rotate the key immediately if it is exposed. As an optional alternative, leave `api_key` empty and use `api_key_env` to read the credential from an environment variable.

Generated config snapshots, metadata, and Markdown reports recursively redact known secret fields. The source local YAML remains plaintext, and redaction cannot protect arbitrary credentials copied into prompts or datasets. Before sharing a branch or training artifacts, run a secret scan over the Git index, Git history when relevant, and `outputs/`. For an exact-key check, compare without printing the key or matching file contents; report only the match count and require zero matches.

When `grpo.reward_judge` points to a remote service, the service receives the full training prompt, candidate completion, reference answer, constraints, and category metadata. Do not send private or regulated training data until the provider's access controls, data-use policy, retention policy, and regional requirements have been reviewed. Use a locally hosted compatible judge when those data cannot leave your environment.

If you find a security issue in the pipeline code, open a private advisory or contact the maintainer directly. Include reproduction steps, expected impact, and any safe mitigation you have already tested.
