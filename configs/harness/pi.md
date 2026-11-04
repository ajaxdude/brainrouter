# Pi Extension Integration

Pi uses extensions, not MCP. To call brainrouter reviews from a pi extension:

## HTTP endpoints

- `POST http://127.0.0.1:9099/review/api/request` — start a review
  Body: `{ "taskId": "task-xxx", "summary": "...", "details": "..." }`
  Returns: `{ "status": "approved|needs_revision|escalated", "feedback": "...", "sessionId": "...", "iterationCount": N, "reviewerType": "llm|human", "reviewModel": "requested <selector>; actual <backend> → <provider> (<model>)" }`
  (`reviewModel` names the reviewer model and whether it ran local or cloud; falls back to the requested card selector if no LLM call succeeded, and is `null` only if the session carries neither — e.g. a session predating the card snapshot.)

- `POST http://127.0.0.1:9099/review/api/request-async` — start a review without waiting (recommended for agents that have other work to do while the review runs)
  Body: same as `/review/api/request`
  Returns immediately: `{ "sessionId": "...", "status": "pending" }`
  Then poll `GET /review/api/sessions/:id` until `status` is terminal (`approved|needs_revision|escalated`). NEVER re-POST to fetch a verdict — it starts a duplicate review. The MCP `request_review` tool exposes this as `wait: false`, with the `wait_review` tool as the blocking verdict fetch.

- `GET http://127.0.0.1:9099/review/api/sessions` — list sessions

- `GET http://127.0.0.1:9099/review/api/sessions/:id` — session detail

- `POST http://127.0.0.1:9099/review/session/:id/resolve` — human resolve
  Body: `{ "feedback": "lgtm" }` (starts with ok/lgtm/approved/looks good/ship it → approved)

## Dashboard

Open http://127.0.0.1:9099/review/ in a browser to see live sessions.

## Pi extension (coming soon)

A dedicated pi extension that wraps these endpoints will be shipped separately.
