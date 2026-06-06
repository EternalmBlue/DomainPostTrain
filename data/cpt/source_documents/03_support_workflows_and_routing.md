# AsterHelp workflows and routing

A standard AsterHelp support flow starts with a clear user request, a ticket ID, the affected service, impact level, recent change, and current owner. If any of those fields are missing, the assistant should ask for the missing observable facts rather than inventing them. When the issue affects many users, includes data loss, or blocks a critical workflow, the ticket should be tagged priority-high and routed to the documented service owner.

A handoff should include the ticket ID, summary, timeline, suspected area, checks already performed, and the next requested action. A handoff should not include secrets, raw credentials, private customer data, or unrelated logs. If logs are needed, the assistant should ask for a minimal sanitized excerpt that the user is allowed to share.

The escalation path is intentionally simple in the mock data. Service questions go to the owning team. Access questions go through an approved access request. Unclear ownership is tagged needs-owner. Customer-visible incidents require a status note that states impact, scope, mitigation, and next update time. The documentation does not define automated paging, chat integrations, or external vendor escalation.
