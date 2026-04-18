# Pi Extension Integration

Pi uses extensions, not MCP. To call brainrouter reviews from a pi extension:

## HTTP endpoints

- `POST http://127.0.0.1:9099/review/api/request` — start a review
  Body: `{ "taskId": "task-xxx", "summary": "...", "details": "..." }`
  Returns: `{ "status": "approved|needs_revision|escalated", "feedback": "...", "sessionId": "...", "iterationCount": N }`

- `GET http://127.0.0.1:9099/review/api/sessions` — list sessions

- `GET http://127.0.0.1:9099/review/api/sessions/:id` — session detail

- `POST http://127.0.0.1:9099/review/session/:id/resolve` — human resolve
  Body: `{ "feedback": "lgtm" }` (starts with ok/lgtm/approved/looks good/ship it → approved)

## Dashboard

Open http://127.0.0.1:9099/review/ in a browser to see live sessions.

## Pi extension (coming soon)

A dedicated pi extension that wraps these endpoints will be shipped separately.
