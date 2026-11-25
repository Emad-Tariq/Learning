# Push the EDA to main

Use the existing Learning checkout and push directly to main. Copy the packaged EDA from the Animal-ReID workspace into that checkout first. Copying files individually also works when the destination EDA directory already exists and excludes runtime subdirectories.

Run in PowerShell:

```powershell
Set-Location "C:\Users\SANGEETA-DAILY\Documents\Abhay\Semester 5\ML\Project\Learning"
git switch main
git pull --ff-only origin main
New-Item -ItemType Directory -Force .\Models\Abhay\EDA | Out-Null
Get-ChildItem -LiteralPath "C:\Users\SANGEETA-DAILY\Documents\ChatGPT\Animal-ReID\Models\Abhay\EDA" -File -Force | Copy-Item -Destination .\Models\Abhay\EDA -Force
git add -- Models/Abhay/EDA
git diff --cached --stat
git diff --cached --check
git commit -m "Add Animal ReID exploratory analysis, plots and diagnostics" -- Models/Abhay/EDA
git push origin main
```

The commit command includes only the EDA path. No commit or push was performed while updating these instructions.
