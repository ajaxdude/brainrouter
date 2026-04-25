# Brainrouter Codebase Audit

**Date:** 2026-04-24
**Scope:** All Rust source files, HTML templates, and config
**Purpose:** Hand off to another agent for fixes

---

## Critical

### C1. SSE stream parser breaks on multi-byte UTF-8 split across chunks
**File:** `src/review/review_loop.rs` L265
```rust
let text = std::str::from_utf8(&chunk)?;
```
Each SSE chunk is raw bytes from reqwest's `bytes_stream()`. A multi-byte UTF-8 character can be split between two consecutive chunks. `from_utf8` will return `Err` and the review loop will abort with a spurious error.
**Fix:** Buffer bytes and parse UTF-8 at line boundaries (e.g. use `String::from_utf8_lossy` or accumulate into a byte buffer, splitting only on `\n`).

### C2. `peek_manifest_model` awaits before timeout wrapping
**File:** `src/router.rs` L239
```rust
let (stream, model_key) = peek_manifest_model(stream).await;
return Ok((
    wrap_with_timeout(stream),
    ...
));
```
`peek_manifest_model` calls `stream.next().await` on the raw stream *before* `wrap_with_timeout` is applied. If Manifest accepts the TCP connection but stalls before sending the first byte, this hangs forever — the 180s stall timeout is never active for the first chunk.
**Fix:** Apply `TimeoutStream` wrapping *before* passing the stream into `peek_manifest_model`, or add an independent `tokio::time::timeout` around the `.next().await` inside `peek_manifest_model`.

### C3. Irrefutable let-binding on `ProviderResponse::Stream`
**File:** `src/router.rs` L487, `src/review/review_loop.rs` L259
```rust
let ProviderResponse::Stream(stream) = resp;       // router.rs
let ProviderResponse::Stream(mut stream) = provider_response;  // review_loop.rs
```
`ProviderResponse` is an enum with one variant today, but this pattern will panic at runtime if a new variant is added. Both locations should use `match` for exhaustive pattern matching.
**Fix:** Replace with `match resp { ProviderResponse::Stream(stream) => { ... } }`.

---

## High

### H1. MCP server default socket path is root-only
**File:** `src/mcp_server.rs` L22, `src/daemon.rs` L38
```rust
#[arg(long, default_value = "/run/brainrouter.sock")]
```
`/run/` is root-writable only. The actual systemd service uses `/run/user/$UID/brainrouter.sock`. If a user runs `brainrouter mcp` without `--socket`, it will fail to connect.
**Fix:** Default to `/run/user/<uid>/brainrouter.sock` using `std::env::var("XDG_RUNTIME_DIR")` or document that `--socket` is required.

### H2. MCP server does not validate `sessionId` before URL interpolation
**File:** `src/mcp_server.rs` L187
```rust
let path = format!("/review/api/sessions/{}", session_id);
```
The `session_id` comes from untrusted MCP input and is interpolated directly into a URL path. A malicious value like `../../api/restart/brainrouter` could route to unintended endpoints (depends on daemon path matching).
**Fix:** Validate `session_id` is alphanumeric + dashes only, or URL-encode it.

### H3. MCP `http_uds_request` ignores HTTP status codes
**File:** `src/mcp_server.rs` L228-239
The response headers are stripped but the HTTP status is never checked. A 500 from the daemon is parsed as JSON and returned as success to the MCP client.
**Fix:** Parse the status line (first line of response) and return an error for non-2xx status codes.

### H4. MCP GET/POST detection is fragile
**File:** `src/mcp_server.rs` L206
```rust
let method = if body_bytes == b"{}" { "GET" } else { "POST" };
```
This breaks if the body is `null`, `""`, or `[]` (all valid empty payloads). GET requests also send a `Content-Length: 2` header with `{}` body, which some HTTP servers reject.
**Fix:** Decide method based on the tool being called (GET for reads, POST for writes) rather than body content. For GET, omit the Content-Length and body entirely.

---

## Medium

### M1. Dead types in `src/types.rs`
**File:** `src/types.rs`
The following public types are defined but never used anywhere in the codebase:
- `ChatCompletionResponse` (L68)
- `ResponseChoice` (L80)
- `Usage` (L89)
- `ModelListResponse` (L97) — shadowed by a private struct in `server.rs`
- `ModelInfo` (L104)

**Fix:** Remove these dead types.

### M2. Dead type alias `SharedSessionManager`
**File:** `src/session.rs` L207
```rust
pub type SharedSessionManager = Arc<SessionManager>;
```
Never used anywhere in the codebase.
**Fix:** Remove it.

### M3. Unused `Router::route` convenience method
**File:** `src/router.rs` L106-111
```rust
pub async fn route(&self, request: ChatCompletionRequest) -> Result<ProviderResponse> {
    self.route_tagged(request, None, String::new()).await.map(|(resp, _)| resp)
}
```
No callers exist in the codebase.
**Fix:** Remove it.

### M4. Unused public functions in `src/review/context.rs`
- `load_agents()` (L55) — called only by `gather()`, should be private
- `section_max()` (L91) — never called anywhere
- `truncate()` (L71) — called only within this module, should be private

**Fix:** Make `load_agents` and `truncate` private (`pub` → `fn`). Remove `section_max()`.

### M5. Stale `dashboard.html` template
**File:** `src/escalation/templates/dashboard.html`
The main dashboard is served from `main_dashboard.html` (embedded in `server.rs`). The old `dashboard.html` is still embedded in `escalation/mod.rs` and served at `/review/`. This is confusing — two different dashboards exist.
**Fix:** Either remove `dashboard.html` and serve the review session list from `main_dashboard.html`, or clearly document why two dashboards exist. At minimum, stop compiling `dashboard.html` into the binary if it is no longer needed.

### M6. O(N) ring buffer removal
**File:** `src/routing_events.rs` L97-99
```rust
while events.len() > MAX_EVENTS {
    events.remove(0);  // O(N) shift on every emit
}
```
`Vec::remove(0)` shifts all elements left. With MAX_EVENTS=500 this is ~500 copies per emit when the buffer is full.
**Fix:** Use `VecDeque` (O(1) push/pop front) or a proper ring buffer.

### M7. Sorting all events under mutex lock
**File:** `src/routing_events.rs` L103-108
```rust
pub fn get_all(&self) -> Vec<RouteEvent> {
    let events = self.events.lock().unwrap();
    let mut list: Vec<_> = events.iter().cloned().collect();
    list.sort_by(|a, b| b.timestamp.cmp(&a.timestamp));
    list
}
```
Clones all 500 events, then sorts them, while holding the mutex (clone + sort both happen before the lock is released because `events` is in scope). The sort is unnecessary since events are appended in chronological order — just reversing the vec would suffice.
**Fix:** Drop the lock after cloning, then reverse (not sort) outside the lock. Or push_front with VecDeque so the natural order is newest-first.

### M8. Two separate mutex locks in `RoutingEvents`
**File:** `src/routing_events.rs` L64-66
```rust
events: std::sync::Mutex<Vec<RouteEvent>>,
counter: std::sync::Mutex<u64>,
```
Two independent mutexes that are always locked together (in `emit`). This is a minor lock-ordering hazard.
**Fix:** Combine into a single `Mutex<(Vec<RouteEvent>, u64)>` or use `AtomicU64` for the counter.

### M9. `handle_versions` shells out to `strings` for version detection
**File:** `src/server.rs` L659-674
Falls back to running the `strings` command on the llama-swap binary. This is fragile and slow.
**Fix:** Now that the upgrade handler downloads release binaries with version tags, store the installed version in a file (e.g. `~/.local/share/brainrouter/llama-swap.version`) during upgrade and read it back here.

### M10. Fallback triggers for 400 Bad Request
**File:** `src/router.rs` L249-255
When Manifest returns a 4xx error (bad request, invalid model, etc.), the router falls through to llama-swap. A request that is structurally invalid will fail on both providers, wasting resources and confusing the dashboard.
**Fix:** Only fall back when `is_backend_fault` is true.

### M11. Inconsistent URL protocol validation
**File:** `src/config.rs`
`manifest.base_url` is validated for `http://` or `https://` prefix (in `daemon.rs` L112-120), but `llama_swap.base_url` has no protocol check.
**Fix:** Apply the same validation to both.

---

## Low

### L1. Massive `handle_request` match statement
**File:** `src/server.rs` L122-367
Single 245-line match statement handles all routes. Every route handler wraps responses with the same `body.map_err(|_| unreachable!()).boxed_unsync()` boilerplate (appears 20+ times).
**Fix:** Extract a helper `fn wrap_full(resp: Response<Full<Bytes>>) -> Response<UnsyncBoxBody<...>>` to eliminate the repetition. Consider splitting route groups into submodules.

### L2. `unreachable!()` in body mapping
**File:** `src/server.rs` — appears ~20 times
```rust
.map_err(|_| unreachable!())
```
`Full<Bytes>` is infallible, so `unreachable!()` is technically correct, but a panic in a server is always risky. Using `|e: Infallible| match e {}` is the zero-overhead proof.
**Fix:** Replace `|_| unreachable!()` with `|e: Infallible| match e {}`.

### L3. `truncate()` can return more than `max` bytes
**File:** `src/review/context.rs` L71-88
After truncating to `max` bytes, the function appends a warning suffix (`\n\n[WARNING: truncated to...]`), making the returned string longer than `max`.
**Fix:** Subtract the warning length from the truncation point, or document that `max` is approximate.

### L4. `home_dir()` fallback to `/root`
**File:** `src/server.rs` L7
```rust
std::env::var("HOME").unwrap_or_else(|_| "/root".to_string())
```
If `$HOME` is unset (e.g. in containers), this silently resolves all paths under `/root`, which is almost certainly wrong for a non-root user.
**Fix:** Fail explicitly when `$HOME` is unset in a daemon context (the check in `daemon.rs` warns but doesn't prevent startup).

### L5. `reqwest::Client` created per-request in version checks
**File:** `src/server.rs` L518-520, L533-535, L796-799, L809-812, L824-827, L842-846
Six different `reqwest::Client` instances are created per `/api/versions` call. Each allocates a connection pool.
**Fix:** Create a single shared `reqwest::Client` (or store one in `AppState`) and reuse it.

### L6. `esc()` XSS prevention in dashboard relies on innerHTML-based encoding
**File:** `src/escalation/templates/main_dashboard.html` L335
```js
function esc(s) { const d=document.createElement('div'); d.appendChild(document.createTextNode(s)); return d.innerHTML; }
```
This is correct but non-standard. Functions like `providerDisplay()` and `modelDisplay()` return raw HTML strings that are concatenated and inserted via `innerHTML`.
**Fix:** Consider using `textContent` where HTML markup is not needed to reduce the attack surface.

### L7. Dashboard dedup uses O(N^2) linear scan
**File:** `src/escalation/templates/main_dashboard.html` ~L444
```js
const existing = deduped.find(d => d.dedupKey === key);
```
Inside a loop over all events. For the 500-event buffer this is 250K comparisons per refresh (every 500ms).
**Fix:** Use a `Map` for O(1) lookups during deduplication.

### L8. Duplicate `BodyExt` import in `handle_request_review`
**File:** `src/escalation/mod.rs` L89
```rust
use http_body_util::BodyExt;  // already imported at module top (L11)
```
**Fix:** Remove the inner import.

### L9. Inconsistent path resolution for agent contract
**File:** `src/review/context.rs` L57
```rust
let path = format!("{}/.omp/agent/LLAMACPP.md", home);
```
Hardcodes `$HOME/.omp` but the review service uses `XDG_CONFIG_HOME` for its own state file.
**Fix:** Use consistent path resolution (both XDG or both `$HOME`-relative).

### L10. `use std::io::Read` import inside `spawn_blocking` closure
**File:** `src/server.rs` L904
```rust
use std::io::Read;
```
This `use` inside the closure body is valid but unusual.
**Fix:** Move to the module-level imports for clarity.

### L11. `Provider` trait is not `Send + Sync` constrained
**File:** `src/provider/mod.rs`
The trait method returns `Pin<Box<dyn Future<...> + Send + '_>>` but the trait itself has no `Send + Sync` bound. This works because only `OpenAiProvider` implements it, but adding a non-Send impl would be a silent soundness issue.
**Fix:** Add `Send + Sync` supertraits: `pub trait Provider: Send + Sync`.

### L12. Review escalation reason is always `LlmError` for escalated reviews
**File:** `src/review/review_loop.rs` L135-136
When the LLM returns an "escalated" status voluntarily, the reason is recorded as `LlmError`, which is misleading. It should be something like `LlmEscalated` or `Voluntary`.
**Fix:** Add a new `EscalationReason::LlmEscalated` variant or use the existing status text.

---

## Summary

| Severity | Count |
|----------|-------|
| Critical | 3 |
| High | 4 |
| Medium | 11 |
| Low | 12 |
| **Total** | **30** |

### Priority order for fixing:
1. **C1** (SSE UTF-8 split) — causes spurious review failures in production
2. **C2** (timeout bypass on first chunk) — can hang requests indefinitely
3. **H2** (session ID injection) — security issue
4. **H1** (default socket path) — breaks MCP without `--socket`
5. **M1-M4** (dead code) — noise reduction, easy wins
6. **M6-M7** (ring buffer perf) — impacts dashboard responsiveness
7. Everything else in order
