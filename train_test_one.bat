set "CONFIG_DIR=config"

venv\Scripts\python.exe src\main.py --config="%CONFIG_DIR%\%1" --train
venv\Scripts\python.exe src\main.py --config="%CONFIG_DIR%\%1" --test
venv\Scripts\python.exe src\main.py --config="%CONFIG_DIR%\%1" --bot