# brainrouter QA — scenario suite + UI walkthrough

Each scenario is one end-to-end question: **"pick X as model/reviewer, point
harness Y at it — does it work?"** Run top to bottom; each is self-contained.
Last verified 2026-10-26 on binary **1.1.13** (master `16bc573`) — including the
UI walkthrough (S10), which was executed with a real browser, not asserted.
Names/ports are the prod box's; on a fresh install adjust (defaults in
README/PRD). `HANDOFF-*.md` files, if present, are machine-local notes — not
documentation.

Shortcuts:

```bash
B=http://127.0.0.1:9099
BR=~/brainrouter-v2/target/release/brainrouter
```

---

## S1 — Does brainrouter itself answer? (2 min)

1. `systemctl --user is-active brainrouter` → **`active`**
2. `curl -s $B/health` → **200 JSON**
3. `curl -s $B/api/versions` → **brainrouter >= 1.1.13**
4. Open `http://127.0.0.1:9099/dashboard` → **renders; browser console (F12) shows zero errors; version badge non-empty; health pill resolves within ~5 s** (a permanently-"loading" pill = JS boot crash — check console, this was a real bug, fixed 867dac7)

If not: `journalctl --user -u brainrouter -n 50`. Adapter WARNs about
`llama-server`/`strix_server` missing on PATH are normal (containerized engines, shim mode).

## S2 — Pick **dirk-qwen3.8-27b-q4-subs** as the model, curl as the harness (2 min)

1. ```bash
   curl -s $B/v1/chat/completions -H 'content-type: application/json' \
     -d '{"model":"dirk-qwen3.8-27b-q4-subs","messages":[{"role":"user","content":"reply with the single word ok"}],"max_tokens":16}'
   ```

   → **streamed `data:` chunks, no error.**
2. `curl -s $B/api/routing-events | python3 -m json.tool | head -30`
   → **`stage=local_primary`, `requested_model == model`** — a substitution/fallback here would be a bug (explicit picks are authoritative).
3. `$BR cli inference` → **subs model resident.**

## S3 — Pick **DeepSeek-V4.1 (Synthetic)** as the chat model (1 min)

1. ```bash
   curl -s $B/v1/chat/completions \
     -d '{"model":"cloud/hf:deepseek-ai/DeepSeek-V4.1-Flash-1017","messages":[{"role":"user","content":"reply with the single word ok"}],"max_tokens":16}'
   ```

   → **streamed chunks.**
2. routing-events → **cloud stage, provider `manifest`.**
3. `curl -s $B/api/service-health` → **`"manifest": "healthy"`** — once p6-absorbed this is the circuit breaker (same predicate routing uses), so a 429 rate-limit does NOT flip it; an open breaker (3 backend faults) does.
4. If you get 429: Synthetic is subscription-throttled — that's S5's trigger, keep going.

## S4 — Pick **DeepSeek-V4.1 as the reviewer**. Does the review loop work? (3 min)

1. Planted-bug repo:

   ```bash
   D=$(mktemp -d /tmp/qa-XXXX) && cd $D && git init -q
   printf 'def add(a,b):\n    return a + b\n' > calc.py
   git add -A && git -c user.email=t@t -c user.name=t commit -qm base
   printf 'def add(a,b):\n    return a - b  # bug\n' > calc.py && git add calc.py
   ```

2. `$BR cli roles get` → **review = cloud/hf:deepseek-ai/DeepSeek-V4.1-Flash-1017** (if not, `cli roles set review` it; **re-check after every daemon restart** — issue #2).
3. `$BR cli review request qa-$(date +%m%d)-001 'planted bug: add subtracts' --cwd $D`
   → **(1–3 min) `STATUS: needs_revision` naming the subtraction.**
4. `$BR cli review lgtm <session-id>`; `cd / && rm -rf $D` → **closed.**

## S5 — Does credit-outage failover route the reviewer to DeepInfra? (2 min)

- **Proven banked check:** `$BR cli review get a8f10372-0b86-4f2e-ab0d-4e860f310158`
  → **`review_model` contains "fell back to ... deepinfra"** — happened on a real
  Synthetic 429. Reading this line = pass.
- **Config gate:** `curl -s $B/api/review-config` → **`fallback_selector` = cloud/deepinfra/deepseek-ai/DeepSeek-V4-Flash.**
- **Force it:** set the card directly and rerun S4 step 3:
  `$BR cli roles set review cloud/deepinfra/deepseek-ai/DeepSeek-V4-Flash`
  → verdict from deepinfra; then **restore the card.**

## S6 — Does a human clear reviews from the browser? (3 min — UI, see S10 for the click path)

CLI-equivalent: `$BR cli review lgtm <id>`. The real UI button (S10 step 5) is the
thing to verify: escalated card → **LGTM** → `$BR cli review get <id>` →
**`status: approved`, `reviewer_type: human`, `human_feedback: "lgtm"`.**

## S7 — Put **brainrouter inside a real harness**. Does the conversation look right? (5 min)

1. Wire any OpenAI-compatible harness to `http://127.0.0.1:9099/v1` (any API key string):

   ```bash
   python3 - <<'PY'
   import openai
   c = openai.OpenAI(base_url="http://127.0.0.1:9099/v1", api_key="x")
   for turn in ("remember the word pineapple", "what word did I ask you to remember?"):
       print(c.chat.completions.create(model="dirk-qwen3.8-27b-q4-subs",
             messages=[{"role":"user","content":turn}]).choices[0].message.content)
   PY
   ```

   (OMP users: add an OpenAI-compatible provider, `base_url` `http://127.0.0.1:9099/v1`, `auth: none`, pick any key from `curl -s $B/v1/models`.)
2. → **ONE card for both turns** (cards group by conversation; turns nest).
3. `curl -s $B/api/routing-events` → **both turns share one `conv_id`.**

## S8 — Can you watch a generation live? (1 min)

Terminal A: `"write 400 words about routing daemons"`, `max_tokens: 400`.
Terminal B immediately: `curl -s $B/api/inflight` → **one row (model, elapsed, bytes).**
DS4-class models finish in ~2 s — long requests only.

## S9 — Does it survive a restart? (3 min; do when no review is in flight)

1. `systemctl --user restart brainrouter`
2. `curl -s $B/api/versions` → **active, same version.**
3. `$BR cli roles get` → **review card unchanged.** Reset to `auto` = issue
   https://github.com/ajaxdude/brainrouter/issues/2 — set it back, comment if new.
4. S3 curl → **cloud answers** (start.sh re-exports `~/.brainrouter-*.env`, both 600).
5. `curl -s $B/api/review-config` → **fallback_selector still set** (yaml, untouched by migrations).

## S10 — UI walkthrough (browser; the part CLI can't test) (10 min)

Verified end-to-end on 2026-10-26 at 1440×900 — zero console errors page-wide.

1. **Overview** `/dashboard` → pill green, version badge shown, posture line
   (`reasoning/review/HankNDory/PR`) populated, not "loading".
2. **Sankey + click linkage:** fire an S2 request first. Six columns render
   (HARNESS/SESSION/ROUTING/BACKEND/ENGINE + MODEL). **Click a card** → its whole
   path stays lit, everything else dims. **Click a sankey node** (e.g. `curl`) →
   same path highlight from the other direction.
3. **Card filters:** `reviews` tab lists review sessions with per-session
   Route→…→Done phase bars and full LLM feedback ("Full LLM review" block);
   search box (top) filters cards by text.
4. **Escalated review = action buttons:** to see them, force one —
   `$BR cli roles set review cloud/deepinfra/no-such-model-xyz`, submit an S4-style
   review `--async`, wait ~30 s → card turns red **Escalated · Human action required**
   with **Send Feedback / Continue iterating / LGTM**. Click **LGTM** → card clears;
   `$BR cli review get <id>` → approved by human. **Restore the card.**
5. **Nav sweep:** Models / Engines / Config / Credentials / Plugins each render
   with distinct content; no page throws a console error. Credential VALUES are
   never displayed.
6. **Guard rail (don't test destructively):** "Flush Models" opens a confirm
   dialog — that dialog is the guard; cancelling leaves models resident.
7. Narrow window (<~900 px) collapses the card grid — responsive design, not a bug.

## Appendix A — what you can pick

| Role | Pick | Selector / command |
|---|---|---|
| Local models | any key from `curl -s $B/v1/models` | bare key |
| Cloud chat / reviewer | DeepSeek-V4.1 (Synthetic) | `cloud/hf:deepseek-ai/DeepSeek-V4.1-Flash-1017` |
| Cloud reviewer backup | DeepSeek-V4-Flash (DeepInfra) | `cloud/deepinfra/deepseek-ai/DeepSeek-V4-Flash` |
| Reviewer role | set it | `$BR cli roles set review <selector>` |
| Undecided | classifier/registry decides | `auto` / `cloud` / `local` |

## Appendix B — full test battery (any code change)

Cargo target names are `tests/` file stems (e.g. `routing_profiles_test`).

```bash
cargo test --locked --lib                    # 501 unit tests at 1.1.13
cargo test --locked --test v2_l1_acceptance --test v2_l2_acceptance --test v2_l3_acceptance \
  --test v2_l4_acceptance --test v2_l5_acceptance 2>&1 | grep -E 'test result|FAILED'
cargo test --locked --test v2_l5_smoke --test v2_l6_acceptance --test v2_l7_acceptance \
  --test review_fallback_test --test routing_profiles_test 2>&1 | grep -E 'test result|FAILED'
cargo clippy --locked --all-targets
cargo build --locked --bin brainrouter
bash scripts/check-html-js.sh
node --test scripts/test-benchmark-ui.cjs
```

Expected reds on the prod box: exactly the two L5 tests needing a live strixite
engine (`strix_server` on `:5300`) —
`l5_external_observer_ready_at_endpoint_attach_adopt_and_ownership`,
`l5_acs_in_process_review_stream_pin_without_inflight_row`. Anything else: investigate.

## Appendix C — sign-off

| # | Check | Pass |
|---|---|---|
| S1 | daemon up, dashboard boots error-free | ☐ |
| S2 | explicit local pick, no substitution | ☐ |
| S3 | cloud pick + honest breaker health | ☐ |
| S4 | reviewer catches planted bug | ☐ |
| S5 | DeepInfra failover (banked or forced) | ☐ |
| S6+S10.4 | human LGTM button resolves escalated session | ☐ |
| S7 | harness conv groups to one card | ☐ |
| S8 | in-flight row visible | ☐ |
| S9 | restart keeps card/config (else issue #2) | ☐ |
| B | battery green except the 2 known reds | ☐ |

## Gotchas

1. Review card can reset on restart — issue #2; re-check `cli roles get`.
2. Reviews read **uncommitted** diff only (`git diff HEAD`); committed batches
   need human re-review via the dashboard reviews tab.
3. Escalated reviews park until a human acts (S10.4 buttons or `cli review lgtm`);
   a 30-min `request_review` timeout is not a failure — poll `cli review get`.
4. Synthetic 429s (subscription limits) are normal and handled by
   `fallback_selector`; spend ledger: `~/.local/state/brainrouter/provider_budget.json`.
5. No hot reload: yaml edits need `systemctl --user restart brainrouter`.
6. Runtime toggles (nudge, prompt-rewrite, privacy chips) revert on restart —
   that's expected; yaml is truth.
