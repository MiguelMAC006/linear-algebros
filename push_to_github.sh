#!/bin/bash
# ============================================================
# Run this script once after creating the GitHub repo.
# Usage: bash push_to_github.sh https://github.com/YOUR_USERNAME/linear-algebros.git
# ============================================================

REPO_URL=$1

if [ -z "$REPO_URL" ]; then
  echo "Usage: bash push_to_github.sh <github-repo-url>"
  exit 1
fi

cd linear-algebros

git init
git add .
git commit -m "Initial commit: project structure, data, all starter files"
git branch -M main
git remote add origin "$REPO_URL"
git push -u origin main

echo ""
echo "Done! Your repo is live at: $REPO_URL"
echo ""
echo "To add teammates as collaborators:"
echo "  GitHub repo → Settings → Collaborators → Add people"
