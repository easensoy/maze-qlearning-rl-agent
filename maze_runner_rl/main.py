import asyncio
from websocket.websocket_server import MazeRLServer

server = MazeRLServer()
asyncio.run(server.start_server())