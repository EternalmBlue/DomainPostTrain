# Contributing

Contributions should keep DomainPostTrain domain-neutral. Do not add proprietary customer data, real credentials, private URLs, or product-specific corpora to the repository.

Before submitting changes, run:

```bash
python -m compileall pipeline scripts
python scripts/training/train_pipeline.py --config configs/domain_post_training.yaml --smoke_test --device cpu
```

If the smoke test cannot run because the local machine lacks ML dependencies, include the failure reason and the static checks you did run.
