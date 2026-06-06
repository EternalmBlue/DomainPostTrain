# AsterHelp configuration and data contracts

AsterHelp examples use three configuration concepts: assistant.domain_name, assistant.answer_style, and assistant.unknown_policy. The assistant.domain_name value names the fictional domain used in prompts and reports. The assistant.answer_style value should be concise by default. The assistant.unknown_policy value should instruct the assistant to say that documentation does not specify an unsupported claim.

The mock ticket schema includes ticket_id, service, summary, impact, owner, status, and tags. Valid statuses in the example data are new, triage, waiting-on-user, in-progress, mitigated, and closed. The mock data does not define billing fields, payment records, customer addresses, production credentials, or personal identity documents.

Configuration changes should be reviewed before training because the system prompt, CPT corpus, SFT answers, and DPO preferences need to agree. If the SFT examples teach a different unknown policy than the model specification, the assistant may become inconsistent. The recommended process is to update the CPT model-spec files first, then update SFT examples, then update DPO preferences and evaluation questions.
