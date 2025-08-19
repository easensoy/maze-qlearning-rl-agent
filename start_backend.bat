@echo off
echo Starting 3D Maze RL Backend Server...
cd /d "C:\Users\alper\OneDrive\Documents\Maze Runner RL"
call venv\Scripts\activate
python maze_runner_rl\backend\websocket_server.py
pause