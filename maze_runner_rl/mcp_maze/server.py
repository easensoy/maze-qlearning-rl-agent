"""A minimal MCP server for seeded maze sessions: reset, observe, step, episode_status.

Run from maze_runner_rl/ (stdio transport, for an MCP client that launches it):
    python -m mcp_maze.server

The rules are in mcp_maze/sessions.py. A refused call (unknown session, bad settings, a
move after the episode ended, a direction outside the enum) comes back as a tool error
with its reason, so a client can count it.
"""
from typing import Literal

from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError

from mcp_maze.sessions import SessionError, SessionManager

INSTRUCTIONS = (
    "Seeded maze sessions. reset starts an isolated episode and returns its session_id. "
    "observe and episode_status cost nothing; step moves one cell and costs one move, even into a wall. "
    "Coordinates are (x, y, level): north is -y, south is +y, west is -x, east is +x, up is +level, down is -level."
)

Direction = Literal["north", "south", "east", "west", "up", "down"]


def build_server(manager=None):
    manager = manager or SessionManager()
    server = MCPServer(name="maze-sessions", instructions=INSTRUCTIONS)

    def run(call):
        try:
            return call()
        except SessionError as error:
            raise ToolError(str(error)) from None

    @server.tool(description="Start a new, isolated maze episode and return its session_id and first observation. "
                             "The same seed and size always give the same maze. width and height must be odd "
                             "(5-41), levels 1-5, max_moves 1-10000.")
    def reset(seed: int, width: int = 11, height: int = 11, levels: int = 1, max_moves: int = 120) -> dict:
        return run(lambda: manager.reset(seed, width, height, levels, max_moves).observe())

    @server.tool(description="Your position, the goal's position, the open directions from your cell, and the "
                             "moves used and left. Costs no move.")
    def observe(session_id: str) -> dict:
        return run(lambda: manager.get(session_id).observe())

    @server.tool(description="Move one cell in a direction. Costs one move; a move into a wall leaves you in place "
                             "and still costs the move. Returns whether you moved and the new observation.")
    def step(session_id: str, direction: Direction) -> dict:
        return run(lambda: manager.get(session_id).step(direction))

    @server.tool(description="Whether the goal was reached, whether the episode is over, moves used and left, wall "
                             "bumps and refused calls. Costs no move.")
    def episode_status(session_id: str) -> dict:
        return run(lambda: manager.get(session_id).status())

    return server, manager


server, manager = build_server()


def main():
    server.run("stdio")


if __name__ == "__main__":
    main()
