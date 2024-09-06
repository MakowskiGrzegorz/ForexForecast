@echo off
setlocal enabledelayedexpansion

REM Set the directory containing the config files
set "CONFIG_DIR=config"

REM Loop through all .yml files in the directory
for %%f in (%CONFIG_DIR%\*.yml) do (
   echo Processing %%f
   venv\Scripts\python.exe src\main.py --config="%%f" --train
)
REM for %%f in (%CONFIG_DIR%\*.yml) do (
REM    echo Processing %%f
REM    venv\Scripts\python.exe src\main.py --config="%%f" --test
REM )
for %%f in (%CONFIG_DIR%\*.yml) do (
    echo Processing %%f
    venv\Scripts\python.exe src\main.py --config="%%f" --bot
)
echo Done!
endlocal