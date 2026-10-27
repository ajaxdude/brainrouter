# brainrouter QA — guided tour + test checklist

**Last verified:** 2026-10-26 against prod binary **1.1.13** (master `16bc573`),
README PRD at the same commit. Every command below was executed successfully on
the prod box. Paths, ports, model names, and provider names are **machine-specific**
to that box; on a fresh install adjust them (defaults noted where they differ).

Some of this reflects **post-PRD behavior** shipped across the v2 program:
provider registry (v2 L6 — `docs/design/v2-streamlining.L6.md`), the `v2.0-p6`
manifest-into-registry migration (`src/migration.rs`), `review.fallback_selector`
(`src/review/review_loop.rs`, `tests/review_fallback_test.rs`), the
`cloud/<provider>/<id>` selector grammar, and `provider_budget.json`. Where these
contradict README/PRD, the code + these verified outputs are current truth until
the PRD is updated.

Mental model in one paragraph: brainrouter is an OpenAI-compatible front door on
`:9099`. Harnesses (OMP, Claude, scripts) point at it and pick a *selector* —
`auto`, a bare local model key, or `cloud/...`. brainrouter owns the decision:
local llama-swap models, cloud providers, role cards, fallback. It also runs a
code-review loop (LLM reviewer → verdict → human escalation) and a live dashboard.

---

## 0. Prereqs (read this first)

- Config default for a fresh install: `$XDG_CONFIG_HOME/brainrouter/brainrouter.yaml`
  (falls back to `~/.config/brainrouter/brainrouter.yaml`). The systemd unit on the
  prod box instead passes `~/brainrouter-v2/brainrouter.yaml` (GUI-mode boot; the
  `brainrouter-headless.yaml` twin only matters on headless boots).
- **No hot reload.** Any yaml edit needs `systemctl --user restart brainrouter`
  (~10 s blip; in-flight sessions hiccup, runtime dashboard toggles revert).
- Secrets: this box uses `~/.brainrouter-synthetic.env` + `~/.brainrouter-deepinfra.env`,
  both `chmod 600`, untracked/gitignored, exported by `~/.local/bin/brainrouter-start.sh`.
  Upstream convention is `/etc/brainrouter/env` (root:aistack 640) or
  `~/.config/brainrouter/.env`. Never put key values in git-tracked files.
- All HTTP checks assume `B=http://127.0.0.1:9099`, `BR=~/brainrouter-v2/target/release/brainrouter`.

```bash
systemctl --user is-active brainrouter        # want: active
curl -s $B/health                             # want: 200 JSON
curl -s $B/api/versions | python3 -m json.tool # want: brainrouter >= 1.1.13
```

## 1. System status

```bash
$BR cli status     # llama-swap, providers, classifier, cloud fallback at a glance
$BR cli versions   # vs /api/versions above
$BR cli inference  # which model is resident right now, state, slot progress
```

Pass: no component shown as down except engines you intentionally do not run
(adapter probe WARNs for `llama-server`/`strix_server` not on PATH are expected —
those engines live in containers; the daemon runs in shim mode against llama-swap).

## 2. Model catalog (what can I ask for?) — input for section 3, no separate pass gate

```bash
curl -s $B/v1/models | python3 -c 'import json,sys; [print(m["id"]) for m in json.load(sys.stdin)["data"]]'
```

Selector cheat sheet (all verified on prod):

| Selector | Meaning |
|---|---|
| `auto` | classifier decides local vs cloud |
| `dirk-qwen3.8-27b-q4-subs` (bare key) | **that** local model, no substitution, no fallback |
| `local` | local routing, model chosen for you |
| `cloud` | cloud, provider auto |
| `cloud/hf:deepseek-ai/DeepSeek-V4.1-Flash-1017` | exact model on provider `manifest` (Synthetic gateway) |
| `cloud/deepinfra/deepseek-ai/DeepSeek-V4-Flash` | exact model on provider `deepinfra` (post-PRD L6 registry) |

## 3. Core routing — make a request and watch the decision

```bash
# 3a. explicit local pick (should NOT swap models, NO fallback on error)
curl -s $B/v1/chat/completions -H 'content-type: application/json' \
  -d '{"model":"dirk-qwen3.8-27b-q4-subs","messages":[{"role":"user","content":"say ok"}],"max_tokens":8}'

# 3b. cloud exact (Synthetic; streams SSE — `data:` lines)
curl -s $B/v1/chat/completions \
  -d '{"model":"cloud/hf:deepseek-ai/DeepSeek-V4.1-Flash-1017","messages":[{"role":"user","content":"say ok"}],"max_tokens":8}'

# 3c. the receipt for both: per-request decisions, conv_id, stage, requested vs actual model
curl -s $B/api/routing-events | python3 -m json.tool | head -40
```

Pass: both stream a completion; routing-events shows `stage=local_primary` for 3a
with `requested_model == model_key` (explicit picks are authoritative — a fallback
here would be a bug) and a cloud stage for 3b.

## 4. Role cards (the single model config)

```bash
$BR cli roles get                       # {assignments, policy}
curl -s $B/api/roles | python3 -m json.tool
# move one (goes through the running daemon, persists to role_cards.json):
$BR cli roles set review cloud/hf:deepseek-ai/DeepSeek-V4.1-Flash-1017
$BR cli roles get | grep -A1 review
```

Pass: assignment appears immediately (daemon write). **After any restart, re-check
it**: the review card may have reset to `auto` — that is known defect
https://github.com/ajaxdude/brainrouter/issues/2 firing, not a new regression.

## 5. Providers + credentials (vault)

Already registered on prod; a fresh box needs the add steps first (they are
idempotent — re-running updates in place):

```bash
# fresh-box setup (post-PRD L6 registry):
#   $BR cli credential add deepinfra-jwt --kind env --env DEEPINFRA_JWT --grants cloud
#   $BR cli provider add deepinfra --kind openai-compatible \
#       --base-url https://api.deepinfra.com/v1/openai \
#       --credential env:DEEPINFRA_JWT --model deepseek-ai/DeepSeek-V4-Flash
# note: `provider test` health-probes DECLARED models, so the --model flag above
# is what makes the next line pass.
$BR cli provider list                   # prod: 'manifest' (env:SYNTHETIC_API_KEY) + 'deepinfra' (env:DEEPINFRA_JWT)
$BR cli provider test deepinfra
$BR cli credential list                 # values NEVER printed — that's the point
```

Pass: provider test succeeds, or fails **only** with DeepInfra's known quirk (their
`/models` probe 401s; actual chat completions work — proved with a
`cloud/deepinfra/...` selector as in section 3). Cloud spend ledger:
`~/.local/state/brainrouter/provider_budget.json` (monthly totals, written by the
v2.0-p6 migration).

## 6. The review loop (the flagship feature)

How it works: agent stages a diff → `request_review` → reviewer LLM reads the PRD,
`AGENTS.md`, and `git diff HEAD` (uncommitted only) → `approved` / `needs_revision`
→ after 5 bad iterations or LLM errors it **parks for a human** (infra escalation,
not a code verdict). Credit-outage failover (post-PRD, v2):
`review.fallback_selector` in the yaml retries the review **once on the DeepInfra
provider** before escalating — this fired for real on 2026-10-26 when Synthetic
returned 429.

Full end-to-end proof you can run in 3 minutes, no OMP session needed:

```bash
D=$(mktemp -d /tmp/qa-proof-XXXX) && cd $D && git init -q
printf 'def add(a,b):\n    return a + b\n' > calc.py
git add -A && git -c user.email=t@t -c user.name=t commit -qm base
printf 'def add(a,b):\n    return a - b  # bug\n' > calc.py && git add calc.py
$BR cli review request qa-proof-$(date +%Y%m%d)-001 'planted bug: add() subtracts' --cwd $D
#   (add --async to not block; then: $BR cli review list / $BR cli review get <id>)
# expect: STATUS: needs_revision flagging the subtraction
$BR cli review lgtm <session-id>     # close it (same action as the dashboard button)
cd / && rm -rf $D
```

Also check: `curl -s $B/api/review-config | python3 -m json.tool` shows
`fallback_selector` set. Human escalation surface: the **dashboard** (`/dashboard`)
shows sessions; JSON list at `/review/api/sessions`, detail at `/review/session/:id`.
(`GET /review/` itself redirects to `/dashboard`.)
Pass: verdict correct; session closed; no parked sessions left (`$BR cli review list`).

## 7. Dashboard (live observability)

Open `http://127.0.0.1:9099/dashboard`. Fire a couple of section-3 requests first.

- **Sankey** (`SK_COLS`, `main_dashboard.html`): six columns — HARNESS (from
  User-Agent), SESSION (OMP session titles), ROUTING (decision class), BACKEND,
  ENGINE, MODEL. Click a node → whole path highlights.
- **Cards**: one card per *conversation*; server-side `conv_id` = stable hash of
  the conversation prefix (system prompt + first user message), or the client
  session header when present; turns nest inside. Click a card → its sankey path lights up.
- **In-flight panel**: `curl -s $B/api/inflight` — live rows (model, elapsed, bytes).
  A ds4 generation takes ~2 s, so race it with a long request or trust the JSON.
- **Known data gap**: the DeepSeek DS4 llama-server exposes no `/slots` and no
  usage chunk, so PP-progress stays 0 and tps chips stay empty for that model.
  Inherent to that build, not a bug (proven by tests + llama-swap `/running`).

## 8. Swap / engines / runtime toggles — informational; toggle checks live here

```bash
$BR cli context status          # llama-swap context size (also: context set <tokens|auto>)
$BR cli nudge status            # thinking-budget nudge (tiers light/deep)
$BR cli prompt-rewrite status   # local pass-through toggle
$BR cli restart llama-swap      # last resort; kills resident models
```

Note: there is no live `/api/context` HTTP route on 1.1.13 (404) — use the CLI.
Toggles flip immediately but are runtime state: a daemon restart reverts them to yaml values.

## 9. Full test battery (run after any code change)

Cargo target names are the file stems in `tests/` (e.g. `routing_profiles_test`,
not `routing_profiles`).

```bash
cd ~/brainrouter-v2   # or the repo clone you're testing
cargo test --locked --lib                                  # ~500 unit tests at 1.1.13 (PRD baseline said 391)
cargo test --locked --test v2_l1_acceptance --test v2_l2_acceptance --test v2_l3_acceptance \
  --test v2_l4_acceptance --test v2_l5_acceptance 2>&1 | grep -E 'test result|FAILED'
cargo test --locked --test v2_l5_smoke --test v2_l6_acceptance --test v2_l7_acceptance \
  --test review_fallback_test --test routing_profiles_test 2>&1 | grep -E 'test result|FAILED'
```

Plus the repo's standing validation gates (fast, run them all before shipping):

```bash
cargo clippy --locked --all-targets
cargo build --locked --bin brainrouter
bash scripts/check-html-js.sh          # dashboard JS syntax gate
node --test scripts/test-benchmark-ui.cjs
```

Expected reds on the prod box: exactly two L5 tests that need a live strixite
engine (the strixite `strix_server` binary/container serving `:5300`; it is not
installed here) — `l5_external_observer_ready_at_endpoint_attach_adopt_and_ownership`
and `l5_acs_in_process_review_stream_pin_without_inflight_row`. Anything else =
investigate before shipping.

## 10. Sign-off table

| # | Check | Pass |
|---|---|---|
| 0 | daemon active, versions >= 1.1.13 | ☐ |
| 1 | cli status green (probe WARNs OK) | ☐ |
| 2+3 | model catalog + local/cloud curl, routing-events correct stage | ☐ |
| 4 | roles set persists; re-check after restart (issue #2) | ☐ |
| 5 | provider deepinfra reachable, vault silent | ☐ |
| 6 | planted-bug review → needs_revision → closed | ☐ |
| 7 | dashboard sankey (6 cols) / cards / inflight live | ☐ |
| 8 | context/nudge/prompt-rewrite status commands answer | ☐ |
| 9 | battery: only the 2 known strixite-dependent reds | ☐ |

## Known gotchas (short list)

1. Restart can reset the review role card — tracked:
   https://github.com/ajaxdude/brainrouter/issues/2. Check section 4 after restarts.
2. `request_review` only sees **uncommitted** diff (`git diff HEAD`) — committed
   batches need human re-review via the dashboard (`/review/api/sessions` for data).
3. Parked/escalated sessions hang until a human resolves (`$BR cli review lgtm` or
   dashboard button). `request_review` blocking ~30 min then timing out is NOT a failure.
4. Synthetic can 429 (rate/credit); `fallback_selector` covers it — watch
   `~/.local/state/brainrouter/provider_budget.json`.
5. Config edits without restart = silently ignored. No hot reload, by design.
