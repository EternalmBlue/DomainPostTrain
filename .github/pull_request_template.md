## Summary

Describe the change and the user-facing workflow it affects.

## Change Type

- [ ] Documentation
- [ ] Config
- [ ] Data contract
- [ ] CPT
- [ ] Fact-SFT
- [ ] DPO
- [ ] Evaluation
- [ ] Inference
- [ ] Export
- [ ] CI / project maintenance

## Validation

- [ ] `python -m compileall pipeline scripts serve_inference.py`
- [ ] JSONL data shape checked when touching `data/`
- [ ] Smoke test run, or reason documented below
- [ ] README/config docs updated when user-facing behavior changed

Smoke test result or skip reason:

```text

```

## Safety Checklist

- [ ] No private documents, customer data, credentials, tokens, or private prompts are included.
- [ ] No generated model artifacts, `outputs/`, `models/`, `.venv/`, or `__pycache__/` are included.
- [ ] Example data remains static and domain-neutral enough for a public repository.
