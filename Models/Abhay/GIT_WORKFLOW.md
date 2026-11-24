# Git preparation — Model 5 OSSF

Repository: `https://github.com/Emad-Tariq/Learning`. The inspected checkout is on `main`. No branch switch, staging, commit or push is performed as part of preparation.

From PowerShell:

```powershell
Set-Location "C:\Users\SANGEETA-DAILY\Documents\Abhay\Semester 5\ML\Project\Learning"
git status --short
git switch -c codex/abhay-model5-ossf
git add -- Models/Abhay Contribution_logs.md
git diff --cached --stat
git diff --cached --check
git diff --cached -- Contribution_logs.md Models/Abhay/RESULTS.md
git commit -m "Add Model 5 open-set score fusion experiment and validation analysis"
git push -u origin codex/abhay-model5-ossf
```

The final two commands create the commit and publish the branch. The branch can then be proposed for merging into `main`. `Models/Abhay/.gitignore` excludes runtime folders, environments, caches, checkpoints and prediction CSVs. Only source notebooks, scripts, tests, documentation and the small reported-results summary belong in this change.

Existing local run folders remain on disk. Runtime snapshots under `local_runs` preserve the source that produced those runs. Edited notebook sources are repository versions with cleared execution output; they do not retroactively change completed results.
