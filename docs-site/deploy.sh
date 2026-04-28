#!/usr/bin/env bash
# Build mdbook locally and publish to the gh-pages branch.
# Use this when GitHub Actions is unavailable; Pages serves from gh-pages.
#
# Requires: mdbook, mdbook-mermaid (cargo install --locked mdbook mdbook-mermaid).
# Run from repo root or docs-site/.
set -euo pipefail

repo_root=$(git rev-parse --show-toplevel)
docs_dir="$repo_root/docs-site"
worktree="$(mktemp -d -t kinetic-gh-pages.XXXX)"

cd "$docs_dir"
mdbook build

git -C "$repo_root" fetch origin gh-pages 2>/dev/null || true
if git -C "$repo_root" show-ref --verify --quiet refs/remotes/origin/gh-pages; then
    git -C "$repo_root" worktree add "$worktree" gh-pages
else
    git -C "$repo_root" worktree add --orphan -b gh-pages "$worktree"
fi

# Replace contents (preserve .git)
find "$worktree" -mindepth 1 -maxdepth 1 ! -name '.git' -exec rm -rf {} +
cp -r "$docs_dir/book/." "$worktree/"
touch "$worktree/.nojekyll"

cd "$worktree"
git add -A
if git diff --cached --quiet; then
    echo "no changes to deploy"
else
    git commit -m "deploy: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
    git push origin gh-pages
fi

cd "$repo_root"
git worktree remove --force "$worktree"
echo "deployed: https://softmata.github.io/kinetic/"
