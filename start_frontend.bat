@echo off
echo Starting 3D Maze RL Frontend (http://localhost:3001)...
cd /d "%~dp0maze-3d-frontend"
if not exist "node_modules" (
    echo Installing frontend dependencies from package-lock.json ...
    call npm ci || goto :error
)
call npm start
pause
exit /b 0

:error
echo npm ci failed. See the messages above.
pause
exit /b 1
