#!/usr/bin/env bash
# sync-toolbox-catalog — refresh the vendored ai-toolbox-cockpit catalog
# snapshot in assets/cockpit-catalog/ from a pinned upstream commit/ref.
#
# What this does NOT do: auto-commit. It only ever rewrites files in the
# working tree; review and commit the diff like any other dependency bump
# (see docs/design/ai-toolbox-cockpit-integration.md §3). The scheduled CI
# workflow (.github/workflows/sync-toolbox-catalog.yml) runs this in apply
# mode and opens a PR only if the working tree actually changed.
#
# Usage:
#   scripts/sync-toolbox-catalog.sh [--ref <branch-or-sha>] [--check-only]
#
#   --ref <branch-or-sha>   Upstream ref to fetch. Defaults to the upstream
#                           repo's default branch HEAD (i.e. "is there
#                           anything new at all").
#   --check-only            Fetch + structurally validate + diff, but never
#                           overwrite assets/cockpit-catalog/ or SOURCE.
#                           Exit code 0 = no change, 3 = change available,
#                           1 = fetch/validation error.
#
# Requires: git, cargo (to build/run the Rust structural validator).
set -euo pipefail

UPSTREAM_URL="https://github.com/kyuz0/ai-toolbox-cockpit.git"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CATALOG_DIR="$REPO_ROOT/assets/cockpit-catalog"
SOURCE_FILE="$CATALOG_DIR/SOURCE"

ref=""
check_only=0
while [[ $# -gt 0 ]]; do
    case "$1" in
        --ref)
            ref="${2:?--ref requires a value}"
            shift 2
            ;;
        --check-only)
            check_only=1
            shift
            ;;
        -h|--help)
            sed -n '2,20p' "${BASH_SOURCE[0]}"
            exit 0
            ;;
        *)
            echo "sync-toolbox-catalog: unknown argument: $1" >&2
            exit 2
            ;;
    esac
done

tmpdir="$(mktemp -d /tmp/brainrouter-catalog-sync.XXXXXX)"
cleanup() { rm -rf "$tmpdir"; }
trap cleanup EXIT

echo "sync-toolbox-catalog: fetching ${ref:-<default branch HEAD>} from $UPSTREAM_URL" >&2
git init --quiet "$tmpdir"
git -C "$tmpdir" remote add origin "$UPSTREAM_URL"
if [[ -n "$ref" ]]; then
    git -C "$tmpdir" fetch --quiet --depth 1 origin "$ref"
else
    git -C "$tmpdir" fetch --quiet --depth 1 origin HEAD
fi
git -C "$tmpdir" checkout --quiet FETCH_HEAD

fetched_sha="$(git -C "$tmpdir" rev-parse HEAD)"
fetched_date="$(git -C "$tmpdir" log -1 --format=%cI HEAD)"

candidate_toolboxes="$tmpdir/ai_toolbox_cockpit/assets/toolboxes.json"
candidate_models="$tmpdir/ai_toolbox_cockpit/assets/models.json"
for f in "$candidate_toolboxes" "$candidate_models"; do
    if [[ ! -f "$f" ]]; then
        echo "sync-toolbox-catalog: expected file not found in upstream checkout: $f" >&2
        echo "sync-toolbox-catalog: upstream's asset layout may have changed; update this script" >&2
        exit 1
    fi
done

echo "sync-toolbox-catalog: structurally validating candidate catalog (commit $fetched_sha)" >&2
if ! (cd "$REPO_ROOT" && cargo run --quiet --bin toolbox_catalog_check -- "$candidate_toolboxes" "$candidate_models"); then
    echo "sync-toolbox-catalog: candidate catalog failed structural validation; not updating vendored files" >&2
    exit 1
fi

vendored_toolboxes="$CATALOG_DIR/toolboxes.json"
vendored_models="$CATALOG_DIR/models.json"

if [[ -f "$vendored_toolboxes" ]] && [[ -f "$vendored_models" ]] \
    && cmp -s "$candidate_toolboxes" "$vendored_toolboxes" \
    && cmp -s "$candidate_models" "$vendored_models"; then
    echo "sync-toolbox-catalog: vendored catalog is already up to date (commit $fetched_sha content-identical)" >&2
    exit 0
fi

echo "sync-toolbox-catalog: candidate catalog differs from vendored copy:" >&2
diff -u "$vendored_toolboxes" "$candidate_toolboxes" || true
diff -u "$vendored_models" "$candidate_models" || true

if [[ "$check_only" -eq 1 ]]; then
    echo "sync-toolbox-catalog: --check-only set; not writing changes" >&2
    exit 3
fi

cp "$candidate_toolboxes" "$vendored_toolboxes"
cp "$candidate_models" "$vendored_models"

# Rewrite just the two pin-tracking lines in SOURCE; everything else in the
# file (the human-readable explanation) is left untouched.
sed -i.bak \
    -e "s|^- Pinned commit: .*|- Pinned commit: \`$fetched_sha\`|" \
    -e "s|^- Commit date: .*|- Commit date: $fetched_date|" \
    "$SOURCE_FILE"
rm -f "$SOURCE_FILE.bak"

echo "sync-toolbox-catalog: updated assets/cockpit-catalog/ to commit $fetched_sha" >&2
echo "sync-toolbox-catalog: review the diff and commit like any other dependency bump" >&2
