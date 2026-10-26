#!/usr/bin/env bash
# v2 L7 hygiene gate (suite tier: v2-streamlining.L7.md, "the deletion pass").
#
# After `v2.0-p7`, the config section spells `swap:`. The pre-`p7` spelling
# `llama_swap:` is legal in exactly two roles:
#   1. the COMPATIBILITY PARSE (`#[serde(alias)]`) and its fixtures/tests, and
#   2. the MIGRATION itself (p7 reads the old section and must name it).
# Anywhere else, `llama_swap:` used as a CONFIG KEY means code is still
# written against the retired spelling. This script fails on that.
#
# Scope note: the bare text `llama_swap:` also occurs as Rust struct-field
# syntax (`llama_swap: Arc<...>`). That is an identifier, not a config key,
# and is out of scope — the gate only polices YAML spellings, i.e. the key
# inside STRING LITERALS (fixture configs, previews, generated YAML), plus
# tracked YAML files.
set -euo pipefail
cd "$(dirname "$0")/.."

fail=0

# 1) `llama_swap:` as a YAML key inside string literals in shipped code.
#    Matched: a quote opening the literal with the key appearing in it
#    (covers "llama_swap:", `{"llama_swap": ...}` JSON, and raw-string YAML
#    headers on the same line). Comments are exempt — they describe history.
while IFS=: read -r file line text; do
    case "$file" in
        src/config.rs|src/migration.rs) continue ;;   # parse + migration
    esac
    if printf '%s' "$text" | grep -qE '^[[:space:]]*(//|///|#|\*)' ; then
        continue
    fi
    echo "$file:$line: $text"
    fail=1
done < <(grep -rnE '"[^"]*"?llama_swap:|llama_swap: *\\?"' src/ --include='*.rs' \
        --exclude=config.rs --exclude=migration.rs \
        | grep -vE '^[^:]+:[0-9]+:[[:space:]]*(//|///)' \
        | grep -vE '(pub )?(llama_swap): *(Arc|bool|String|Option|&)' || true)

# 2) Tracked YAML files must use the new spelling.
while IFS=: read -r file line text; do
    echo "$file:$line: $text"
    fail=1
done < <(grep -rn '^[[:space:]]*llama_swap:' --include='*.yaml' --include='*.yml' . \
        --exclude-dir=target --exclude-dir=.git 2>/dev/null \
        | grep -vE 'tests?/|fixtures?/' || true)

if [ "$fail" -ne 0 ]; then
    echo
    echo "FAIL: the retired \`llama_swap:\` config spelling is live outside the"
    echo "compatibility parse / migration. Use \`swap:\` (v2.0-p7)."
    exit 1
fi
echo "check-v2-p7-clean: OK (no live \`llama_swap:\` YAML key outside config.rs / migration.rs / tests)"
