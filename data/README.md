# Packaged Mock Training Data

This directory contains static mock data for the DomainPostTrain example project. The fictional domain is `AsterHelp`, an internal support knowledge-base assistant.

The data is intentionally small and readable:

- `data/cpt/source_documents/*.md` contains CPT-style source documents.
- `data/sft/*.jsonl` contains `instruction`/`output` examples.
- `data/dpo/preference_examples.jsonl` contains `prompt`/`chosen`/`rejected` preference pairs.
- `data/eval/quality_questions.jsonl` contains post-training quality evaluation questions.

Do not put private documents, credentials, customer tickets, source code, internal prompts, or license-restricted data into this directory before publishing a derivative repository.
