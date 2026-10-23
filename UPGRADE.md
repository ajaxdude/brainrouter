# UPGRADE notes

Per-phase upgrade/rollback procedures. Phase tags + the per-leg restore
contract live in `docs/design/v2-streamlining.md` ("Rollout, migration, and
rollback"); the change list is in `CHANGELOG.md`.

## v2.0-p3 (L3): role cards + decision slot

v1 configured models in four places: the `routing:` profile block
(+ presets), `project_pins.json`, `review.forced_mode`/`forced_model`, and
the `bonsai:` classifier section. v2 L3 replaces all of them with
**role cards** (`$XDG_STATE_HOME/brainrouter/role_cards.json`) + the
**decision slot** (`decision_model:` in YAML). Booting a v2 L3 daemon with a
pre-L3 config runs the `v2.0-p3` migration automatically:

1. Everything consumed is snapshotted — the config YAML, the credentials
   store, `project_pins.json`, and `routing_state.json` — under
   `$XDG_STATE_HOME/brainrouter/migrations/v2.0-p3/<UTC timestamp>/`.
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
# files and removes role_cards.json
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
   `$XDG_STATE_HOME/brainrouter/migrations/v2.0-p2/<UTC timestamp>/`
   (owner-only, mode 0700).
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
