#!/usr/bin/env bash
# Publish website/ to the gh-pages branch (GitHub Pages, no Actions needed).
#   bash scripts/publish_site.sh            # push website/ as the root of gh-pages
#   bash scripts/publish_site.sh --dry-run
# Then enable Pages once: repo Settings -> Pages -> Source: "Deploy from a branch" -> gh-pages / (root).
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$ROOT"
DRY=0; [[ ${1:-} == --dry-run ]] && DRY=1
[[ -f website/index.html ]] || { echo "website/index.html missing" >&2; exit 1; }
python3 -c "import json,sys; [json.load(open(p)) for p in ('website/data/results.json','website/data/gallery.json')]; print('json ok')"
tmp="$(mktemp -d)"
cp -R website/. "$tmp"/ && touch "$tmp/.nojekyll"
commit="$(git rev-parse --short HEAD)"
echo "publishing website/ @ $commit -> gh-pages"
if [[ $DRY -eq 1 ]]; then ls -aR "$tmp" | head -30; rm -rf "$tmp"; exit 0; fi
git fetch -q origin gh-pages 2>/dev/null || true
wt="$(mktemp -d)"
if git show-ref --quiet refs/remotes/origin/gh-pages; then git worktree add -q "$wt" origin/gh-pages; else git worktree add -q --detach "$wt"; (cd "$wt" && git checkout -q --orphan gh-pages && git rm -rfq . >/dev/null 2>&1 || true); fi
(cd "$wt" && find . -mindepth 1 -maxdepth 1 ! -name .git -exec rm -rf {} + && cp -R "$tmp"/. . && git add -A && git -c user.name="${GIT_AUTHOR_NAME:-$(git config user.name)}" -c user.email="${GIT_AUTHOR_EMAIL:-$(git config user.email)}" commit -q -m "site: publish website/ @ $commit" && git push -q origin HEAD:gh-pages)
git worktree remove --force "$wt"; rm -rf "$tmp"
echo "done: https://<owner>.github.io/open-vision-banana/"
