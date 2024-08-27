@echo off
setlocal enabledelayedexpansion

REM Set the directory containing the config files
set "CONFIG_DIR=config"

REM Loop through all .yml files in the directory
for %%f in (%CONFIG_DIR%\*.yml) do (
    echo Processing %%f
    venv\Scripts\python.exe experiments\hyperparameter_search.py "%%f"
)

echo Done!
endlocal