@echo off
echo Starting 3D Maze RL Backend Server (ws://localhost:8765)...
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
    echo Creating .venv and installing maze_runner_rl\requirements.txt ...
    py -3.13 -m venv .venv || goto :error
    ".venv\Scripts\python.exe" -m pip install -r maze_runner_rl\requirements.txt || goto :error
)
cd /d "%~dp0maze_runner_rl"
"..\.venv\Scripts\python.exe" main.py
pause
exit /b 0

:error
echo Setup failed. See the messages above.
pause
exit /b 1
