#!/bin/bash
# ══════════════════════════════════════════════════════════════════
#  BloodSense AI — Push to GitHub
#  Run this script in the project root after cloning the repo
# ══════════════════════════════════════════════════════════════════

echo "🩸 BloodSense AI — GitHub Push Script"
echo "════════════════════════════════════════"

# Configure git (only first time)
git config user.email "rahulpatelanuppur@gmail.com"
git config user.name "Rahul Patel"

# Stage all new/changed files
git add app.py
git add model_v2.py
git add requirements.txt
git add README.md
git add templates/index.html
git add templates/result.html
git add templates/history.html

# Commit
git commit -m "feat: BloodSense AI v2 - animated dashboard, patient history, EfficientNetV2S model (99%+)"

# Push to main
git push origin main

echo ""
echo "✅ Done! Visit: https://github.com/RAHULPATEL2002/blood-group-detection"
