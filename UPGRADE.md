# UPGRADE notes

Per-phase upgrade/rollback procedures. Phase tags + the per-leg restore
contract live in `docs/design/v2-streamlining.md` ("Rollout, migration, and
rollback"); the change list is in `CHANGELOG.md`.

## v2.0-p7 (L7): `swap:` section + the flip to the native supervisor

L7 absorbs llama-swap. The section formerly spelled `llama_swap:` is now
`swap:`; every kept key moves verbatim (`fallback_model`, `subs_model`,
`local_models`, `local_system_prompt`, `nudge.model_key`) and `base_url` is
dropped (no longer used: native mode talks to the swap supervisor; the shim
URL is fixed in shim mode). A NON-default `base_url` — an endpoint the
operator chose deliberately — is kept and marked deprecated instead. Two new keys:

```yaml
swap:
  mode: shim            # shim (today's behavior) | native (the flip)
  config_path: "/opt/ai/llama-swap/config.yaml"   # optional import source
  fallback_model: "your-local-model-key"
```

### First boot (automatic)

`v2.0-p7` renames the section in place (snapshot: 
`$XDG_STATE_HOME/brainrouter/migrations/v2.0-p7/config.yaml`; `.bak` sidecar),
drops `base_url` with the note above, and leaves a file that boots. Preview
and roll back without state writes:

```sh
brainrouter serve --dry-run           # includes the v2.0-p7 preview
brainrouter migration restore v2.0-p7 # puts the llama_swap: section + base_url back
```

### fedora hosts: set `config_path` FIRST

The native catalog imports from `swap.config_path` (default
`~/.config/llama-swap/config.yaml`). On fedora the live llama-swap config is
the docker-mounted file, so the pilot step is:

1. `swap.config_path: "/opt/ai/llama-swap/config.yaml"` in `brainrouter.yaml`,
2. restart (the import runs at boot; `v2.0-p4` uses this path when present),
3. `brainrouter swap catalog list` — every model key from the llama-swap
   config must be listed,
4. parity check (one request per key through brainrouter) **before** flipping.

Without step 1 the fedora native catalog is empty and the flip strands the
host.

### The flip (shim → native)

1. Flush the shim so it holds no models: dashboard **Engines → Unload models**
   (or `brainrouter cli flush-models`). This frees VRAM.
2. Set `swap.mode: native` and restart brainrouter.

Native boot REFUSES while llama-swap `/running` still reports loaded models:

```
swap.mode: native refuses to boot — the llama-swap shim still reports loaded
models: model-a, model-b. Flush them first ... then restart.
```

That refusal is the safety interlock (no silent double residency). After a
clean flip no component dials `:8081`; the supervisor owns every catalog key
(`ensure` = attach-or-spawn, budget-admitted), flush/restart/model-listing/
inference-status/registration all read the supervisor or the catalog, and the
`/api/llama-swap-config` surface serves the swap-catalog view (POST is refused:
the shim file is the import source, not the governing catalog).

### Rollback (native → shim)

1. Stop brainrouter. Native children are stopped by the scheduler on shutdown
   (SIGTERM-grace, per-GGUF flock released at child exit) — verify `swap
   status` shows no residents after the stop.
2. Set `swap.mode: shim` and restart. llama-swap loads models on demand again;
   the shim Residency source is live again and the config is unchanged.

The flip is reversible until the deletion pass (soak exit: ≥7 days native, a
full swap cycle of every catalog key, zero unexplained double-residency /
no-fit / `blocked` events, budget view consistent, owner sign-off).

### Groups (kickoff inventory)

The owner's imported llama-swap config carries **2 groups**: `dirk-dual`
(swap=false, exclusive=false) and `non-dual` (swap=true, exclusive=true). The
native supervisor currently runs single-model entries; slot-sharing for group
members is an OWNER-SCOPE decision before any group key routes natively —
groups never silently degrade. See `docs/design/v2-streamlining.L7.log.md`
("Groups kickoff artifact").

## v2.0-p6 (L6): providers, Manifest absorbed, LAN federation

L6 replaces the hardcoded Manifest backend with a provider registry. New
state under `$XDG_STATE_HOME/brainrouter/`:

- `providers.json` (0600) — `{providers: [record]}`; `credential_id` names a
  credential in the L2 vault (or `env:VAR` for the migrated v1 indirection).
  Key VALUES never live here.
- `provider_budget.json` (0600) — the monthly provider accumulator, keyed
  `YYYY-MM`. The L4 swap `budget.json` is a separate ledger (unrelated).

### First boot (automatic)

The `v2.0-p6` migration consumes exactly
`manifest.enabled` + `manifest.base_url` + `manifest.api_key_env` and creates:

```json
{"id": "manifest", "kind": "manifest", "base_url": "<v1 url>",
 "credential_id": "env:<api_key_env>", "auto": true, "models": [],
 "lan_peer": false, "enabled": true, "budget_mode": "warn"}
```

`models: []` is intentional: the v1 gateway accepts ids it did not advertise,
so `cloud/<id>` passes through verbatim (v1 parity). The `manifest:` block is
then removed from the YAML; `config.yaml.bak` and
`migrations/v2.0-p6/config.yaml` hold the pre-migration bytes. Preview with:

```sh
brainrouter serve --dry-run          # includes the v2.0-p6 preview
brainrouter migration restore v2.0-p6 # round-trips YAML, removes providers+ledger
```

Use the CLI (mutations go through the daemon) or the dashboard Credentials →
Providers section:

```sh
# Bare `cloud` = the enabled manifest-auto provider (v1 behavior).
brainrouter cli provider list

# Reference deployment: DeepInfra (OpenAI-compatible) as the reviewer cloud.
brainrouter cli credential add deepinfra-key --kind store --value '<paste>'
brainrouter cli provider add deepinfra --kind openai-compatible \
  --base-url https://api.deepinfra.com/v1/openai --credential deepinfra-key \
  --model deepseek-ai/DeepSeek-V4-Flash
brainrouter cli provider test deepinfra
brainrouter cli roles set review cloud/deepinfra/deepseek-ai/DeepSeek-V4-Flash

# Anthropic wire provider (x-api-key + anthropic-version, never Bearer):
brainrouter cli provider add anthropic-cloud --kind anthropic \
  --base-url https://api.anthropic.com/v1 --credential anthropic-key \
  --model claude-sonnet-4-5
```

Manual provider fields: `kind` (`openai-compatible|anthropic|manifest`),
`base_url`, `credential_id` (omit = auth-less local gateway), `auto`
(manifest kind only), `models[{id, alias?, ctx, tools, tier,
price:{in,out}}]`, `lan_peer`, `enabled`, `monthly_budget_usd`,
`budget_mode` (`warn|stop`). Tier is `fast|balanced|deep` and drives pool
ranking; price is `$/1K` in/out and feeds the budget guard.

Selector forms: `cloud` (manifest-auto or pool top), `cloud/<id>` (unique
id/alias match → else manifest passthrough; multi-match is a 4xx naming the
providers), `cloud/<provider>/<model>` (qualified; only when the first segment
names an enabled provider — otherwise the remainder cascades). A `cloud/`
prefix never reaches llama-swap.

### LAN federation (opt-in; second listener)

Default off: no listener exists and the loopback surface is byte-identical to
v1. To enable, add to `brainrouter.yaml` and restart:

```yaml
lan:
  enabled: true
  bind: "0.0.0.0:9099"         # second listener, dedicated to inference
  token_credential: lan-token  # credential id in the vault
  expose: [qwen3-8b, cloud/deepinfra/deepseek-ai/DeepSeek-V4-Flash]
  rate: {burst: 20, per_sec: 4}
```

```sh
brainrouter cli credential add lan-token --kind store --value "$(openssl rand -hex 32)"
```

Boot logs a plaintext-LAN risk marker (accepted risk: bearer over HTTP on a
trusted LAN; `peer-info` exposes `host_id`, brainrouter `version`, and the
exposed model/tier/state menu to any token holder on that LAN). The LAN route
table is only `/v1/chat/completions`, `/v1/messages`, `/v1/models` (filtered),
`/api/peer-info`; admin/dashboard/roles/credentials return 404 at listener
level.

Expose is enforced on the RAW requested string AND on every model the route
can actually dispatch (managed `local` ⇒ `fallback_model`, `subs` ⇒
`subs_model`, `auto` ⇒ the decision chain/pool, `cloud` ⇒ pool top plus the
managed fallback), including alias-resolved values. Expose the fallback and
pool models too, or managed selectors will be refused.

- **Rotation**: `brainrouter cli credential add lan-token --kind store --value
  '<new>'` — the old token stops working on the next request (constant-time
  compare against the resolved vault value, file-first; no restart). Provider
  credentials resolve the same way.
- **Expose double-enforcement**: the RAW requested string AND the RESOLVED
  model must both be listed. An exposed alias resolving to a hidden model is
  refused (no cloud-spend escape).
- **Peer provider** (federation): register the peer's `http://host:9099/v1`
  with `lan_peer: true` and the peer's LAN token as its credential. Only peers
  receive the forwarded `x-brainrouter-role` header, so the peer applies its
  own Review/Main/Subagent cards to the named role.
- **Budget**: set `monthly_budget_usd` + `budget_mode` per provider.
  `warn` badges + events; `stop` refuses as a primary failure (the managed
  `fallback_model` hop fires; a review escalates) and excludes the provider
  from the auto pool. The ledger rolls over on the calendar month and survives
  restarts.
- **Revert path**: `brainrouter credential export --env-line
  /etc/brainrouter/env --force` writes `NAME=VALUE` lines for `store`
  credentials (0600, atomic; `env`/`file` references become comments) so a
  reverted v1 can consume them via `EnvironmentFile=`.

### fedora hosts

The Manifest binary is untouched; brainrouter simply no longer requires it.
On fedora1/fedora2 keep `manifest.base_url` pointing at the local gateway
during migration, or register the cloud provider directly and drop Manifest.

## v2.0-p4 (L4): swap catalog + platform probe

L4 introduces two owner-visible state files under
`$XDG_STATE_HOME/brainrouter/`:

- `swap_catalog.json` — the model registry (entries with resolved launch
  specs, sized pins, llama-swap groups metadata);
- `budget.json` — the append-only budget ledger (admit/deny/evict/pin/spawn/
  stop/adopt entries);
- plus `adoption/<model>.json` markers (one per native child) and
  `swap_logs/<model>.log` engine logs.

On the first boot of a v2 L4 daemon the `v2.0-p4` migration imports the live
llama-swap `~/.config/llama-swap/config.yaml` into `swap_catalog.json`:

1. A stable snapshot directory
   `$XDG_STATE_HOME/brainrouter/migrations/v2.0-p4/` captures the
   pre-migration catalog and budget ledger (0600, atomic).
2. The macro chain is expanded (`${ls}`, `${common}`, per-model macro
   overrides) but runtime placeholders (`${PORT}`) are preserved; groups are
   recorded as metadata; existing CLI-added entries are never overwritten.
3. The catalog is marked with the phase; later boots are no-ops. Re-running
   the import is idempotent (`brainrouter swap import-llamaswap`), and
   `--dry-run` prints the plan without touching state.

Preview before booting:

```sh
brainrouter serve --dry-run            # includes the v2.0-p4 preview
brainrouter swap import-llamaswap --dry-run
```

Rollback: `brainrouter migration restore v2.0-p4` restores the pre-import
catalog/budget (or removes them if they did not exist), then revert the
phase merge. Pins and budget reservations are state, not config — the
snapshot is what returns them.

Operational notes:

- Native spawns go through a transient systemd user scope
  (`systemd-run --user --scope`) so engines survive daemon restarts and can
  be adopted. The daemon tracks the ENGINE pid (resolved from the scope's
  `cgroup.procs`), never the `systemd-run` wrapper, and daemon shutdown never
  kills engines. Without systemd the daemon falls back to `setsid`; adoption
  is best-effort and the Engines tab says so. Do not run the daemon with
  `KillMode=control-group` expecting native children to die with it — that
  is the point of the scope.
- Cancelling a queue row aborts a ROUTED request's wait (the harness request
  carrying an in-flight row). A direct `POST /api/engines/ensure` caller owns
  its request: dropping the client connection is its cancel, and the queue
  row is informational.
- A running engine is never killed by an upgrade. If the resolved launch spec
  changed, it is adopted and badged `spec_stale`; `swap ensure <model>`
  converges it explicitly.
- The swap queue is in-memory and resets on daemon restart (clients retry;
  adopted loads keep serving).
- `budget.json` is append-only but bounded: it rotates at write, keeping the
  newest 5000 rows. An LRU pass records one aggregate row (a single victim
  keeps its model id).
- `BRAINROUTER_BUDGET_GIB` overrides the probe-reported budget (GiB);
  `BRAINROUTER_SWAP_QUEUE_CAP`, `BRAINROUTER_SWAP_WAIT_MS`,
  `BRAINROUTER_SWAP_STOP_GRACE_MS`, and `BRAINROUTER_SWAP_NO_SYSTEMD=1` are
  operational knobs.

## v2.0-p3 (L3): role cards + decision slot

v1 configured models in four places: the `routing:` profile block
(+ presets), `project_pins.json`, `review.forced_mode`/`forced_model`, and
the `bonsai:` classifier section. v2 L3 replaces all of them with
**role cards** (`$XDG_STATE_HOME/brainrouter/role_cards.json`) + the
**decision slot** (`decision_model:` in YAML). Booting a v2 L3 daemon with a
pre-L3 config runs the `v2.0-p3` migration automatically:

1. Everything consumed is snapshotted — the config YAML, the credentials
   store, `project_pins.json`, `routing_state.json`, and (from Step 9) the
   `role_cards.json` store — under the stable directory
   `$XDG_STATE_HOME/brainrouter/migrations/v2.0-p3/` (one snapshot per
   migration phase; a one-shot migration has one pre-state).
2. Cards are written via the exact table:

   | v1 state | card |
   |---|---|
   | `Auto` | `auto` |
   | `Local{None}` (the default with no `routing:` block) | `local` |
   | `Local{Some(id)}` | `<bare-key>` |
   | `Cloud{None}` | `cloud` |
   | `Cloud{Some(id)}` | `cloud/<id>` |
   | presets | expanded over `main`/`reviewer` via the same table |
   | project pins (per scope, incl. `subagent_model`/subagent pin) | per-project cards |
   | `review.forced_mode`/`forced_model` | global `review` card (the routing block's reviewer wins, v1 precedence) |
   | `review.max_iterations`/`hankndory_integration`/`design_doc_path` | review-card extras |
   | `bonsai.*` | `decision_model:` (kind `openai-compatible`, `prompt_profile: bonsai`, launch backend under `decision_model.launch`) |

3. The consumed YAML keys (`routing:`, `review.forced_mode/forced_model`,
   `bonsai:`) are stripped; a `.bak` copy of the original is written next to
   the config. The `v2.0-p3` phase runs BEFORE `v2.0-p2`, so restoring it
   returns the pristine v1 YAML.

Preview without writing: `brainrouter serve --config <path> --dry-run`.
Rollback:

```bash
brainrouter migration --config brainrouter.yaml restore v2.0-p3
# restores the pre-migration YAML byte-for-byte + the v1 pins/profile state
# files, and the role_cards.json captured in the snapshot (or removes it
# when the snapshot predates the cards store)
```

The v1 selector surface on the proxy is unchanged (`auto`, `local`, `cloud`,
`cloud/<id>`, `subs`, `brainrouter/<id>`); a new `x-brainrouter-role` header
(labels `main|review|subagent`) tags events, and the `subs` selector now
NAMES the subagent role and resolves through its card. `sync-omp` and
`routing-profile` are gone: `brainrouter roles get|set|remove`,
`brainrouter decision show|test`, and `brainrouter credential …` replace
them, and install templates carry exactly one `auto` entry per harness.

## v2.0-p2 (L2): the bridge becomes a plugin

v1 ran Discord/Signal inside the daemon behind cargo features. v2 L2 moves
them to the supervised `bridge` plugin; the core binary no longer contains
any transport code (serenity is gone from the dependency tree).

### What happens automatically on first boot

Booting a v2 daemon with a v1 `brainrouter.yaml` runs the `v2.0-p2`
migration before validation:

1. The pre-migration config YAML + credentials store are snapshotted to
   `$XDG_STATE_HOME/brainrouter/migrations/v2.0-p2/` (one snapshot per
   phase; owner-only, mode 0700).
2. The plaintext `bridge.discord.token` value moves into
   `$XDG_STATE_HOME/brainrouter/credentials.json` (0600, atomic writes) as
   credential `discord_token`, granted to plugin `bridge`. The value is
   masked in every API/log/`config show` surface afterwards.
3. The key is removed from the YAML; the original file is kept as
   `<config>.bak`.
4. The migration is logged (token value never).

`brainrouter serve --dry-run` prints this preview and exits 0 without
touching any state (R9 contract).

### Installing the bridge plugin

```bash
cargo build --manifest-path plugins/bridge/Cargo.toml --release
mkdir -p bridge-dist && cp plugins/bridge/plugin.toml bridge-dist/ \
  && cp plugins/bridge/target/release/brainrouter-bridge bridge-dist/
brainrouter plugin install bridge-dist
brainrouter plugin config set bridge \
  discord.enabled=true signal.enabled=true \
  discord.prefix='!' signal.llama_swap_url=http://localhost:8081
brainrouter plugin enable bridge
```

(Install from a directory containing ONLY `plugin.toml` + the binary; the
plugin source tree's `target/` would be copied verbatim.)

Effect:

- The daemon mints a per-plugin token (`$XDG_STATE_HOME/brainrouter/
  plugins/bridge/token`, 0600) and passes it to the child ONLY via the
  `BRAINROUTER_PLUGIN_TOKEN` environment variable.
- The plugin's state dir `$XDG_STATE_HOME/brainrouter/plugins/bridge/`
  (0700) holds its config, kv, logs, and persistence.
- On the plugin's first start it copies the six v1 persistence files
  (`{discord,signal}-{sessions,channel-models,work-dirs}.json`) from
  `~/.local/share/omp-bridge/` into the plugin state dir. The v1 originals
  are left in place — nothing is deleted, so a revert keeps working state.
- The Discord token reaches the plugin only through the grant-scoped
  `secrets.get` RPC; it is not present in any plugin config file.
- Pause/resume per transport (the v1 dashboard toggles):
  `brainrouter plugin config set bridge discord.enabled=false` (also from
  the Plugins tab).

### Reverting

```bash
brainrouter migration restore v2.0-p2   # newest snapshot: YAML keys + credentials
```

Then check out the `v1` branch (or `v1-classic`) per the global rollback
contract. Reverting code alone is never the whole restore — run the
migration restore first so the config/YAML keys and credential grants are
the pre-migration ones.
