#!/usr/bin/env bash
# check-html-js — syntax-check every <script> block embedded in the dashboard
# templates.  cargo test never executes the HTML (it's a compile-time string),
# so a JS SyntaxError ships silently and kills ALL dashboard JS — the one class
# of dashboard bug the Rust suite cannot catch.
#
# Usage: scripts/check-html-js.sh [file.html ...]
#        (default: the two templates embedded via include_str!)
set -euo pipefail

files=("$@")
if [[ ${#files[@]} -eq 0 ]]; then
    files=(src/escalation/templates/main_dashboard.html src/escalation/templates/session.html)
fi

node_bin="$(command -v node || command -v bun || true)"
if [[ -z "$node_bin" ]]; then
    echo "check-html-js: node/bun not found; skipping" >&2
    exit 0
fi

fail=0
for f in "${files[@]}"; do
    [[ -f "$f" ]] || { echo "check-html-js: $f not found" >&2; exit 1; }
    tmpdir=$(mktemp -d /tmp/brainrouter-html-js.XXXXXX)
    # One temp file per <script>…</script> body.
    "$node_bin" -e '
        const fs = require("fs");
        const [srcFile, outPrefix] = process.argv.slice(1);
        const src = fs.readFileSync(srcFile, "utf8");
        const blocks = [...src.matchAll(/<script>([\s\S]*?)<\/script>/g)];
        blocks.forEach((b, i) => fs.writeFileSync(`${outPrefix}.${i}.js`, b[1]));
        process.stdout.write(String(blocks.length));
    ' "$f" "$tmpdir/$(basename "$f")" > "$tmpdir/count"
    n=$(cat "$tmpdir/count")
    i=0
    while [[ $i -lt $n ]]; do
        if ! "$node_bin" --check "$tmpdir/$(basename "$f").$i.js" 2>"$tmpdir/err"; then
            echo "check-html-js: SYNTAX ERROR in $f <script> #$((i+1)):" >&2
            sed 's/^/    /' "$tmpdir/err" >&2
            fail=1
        fi
        i=$((i + 1))
    done
    rm -rf "$tmpdir"
    if [[ $fail -eq 0 ]]; then
        echo "check-html-js: $f OK ($n script blocks)"
    fi
done
exit $fail
