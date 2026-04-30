#!/usr/bin/env bash
set -euo pipefail

echo "==> Step 1: Undo last commit but keep changes staged"
git reset --soft HEAD~1

echo "==> Step 2: Unstage everything"
git reset

echo "==> Step 3: Purge Git index cache"
git rm -r --cached .

echo "==> Step 4: Re-add files respecting .gitignore"
git add .

echo "==> Step 5: Verify tracked changes are lightweight"
git status

echo
echo "Check above output: no *.npz, *.msh, or *.log should appear under 'Changes to be committed'."
