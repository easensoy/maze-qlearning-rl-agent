# Claude navigating mazes through MCP tools

[README](../../README.md) · [Value-propagation experiment](../value-propagation/README.md) · [Evaluation suite](../evaluation/README.md)

**Status: the harness, the MCP server and the deterministic references are complete and tested; the Claude run itself has not been executed.** This machine has no Claude API key configured, and no spending limit has been approved for this experiment. No Claude result is reported anywhere in this repository until the run below has been made with real API responses.

This is a zero-shot tool-use task, not a contest with the reinforcement-learning agents: the agents differ in what they are told and in how much experience they get (see [What is and is not compared](#what-is-and-is-not-compared)).

## The task

- **Mazes:** 10 held-out mazes, 11 × 11 cells on one level, built by the repository's 3D generator with the live viewer's density settings, maze seeds 30000–30009. No other experiment in this repository uses these seeds.
- **Goal:** start in one corner and reach the goal in the opposite corner within **120 moves**.
- **Trials:** 3 per maze for Claude, so 30 episodes.

## Tools, served over MCP

[`mcp_maze/server.py`](../../maze_runner_rl/mcp_maze/server.py) is a minimal [Model Context Protocol](https://modelcontextprotocol.io/) server (Python `mcp` SDK, stdio transport) over isolated, seeded sessions ([`mcp_maze/sessions.py`](../../maze_runner_rl/mcp_maze/sessions.py)):

| Tool | Does | Costs a move |
|---|---|---|
| `reset(seed, width, height, levels, max_moves)` | Starts a new, isolated episode; returns its `session_id` and first observation | – |
| `observe(session_id)` | Position, goal position, open directions from the current cell, moves used and left | no |
| `step(session_id, direction)` | Moves one cell; `direction` is an enum of north, south, east, west, up, down | yes, also into a wall |
| `episode_status(session_id)` | Reached the goal, episode over, moves used and left, wall bumps, refused calls | no |

A refused call (an unknown session, a move after the episode ended, a direction outside the enum) comes back as an MCP tool error with its reason. The schemas Claude receives are the server's own listing, recorded in [`task.json`](task.json).

## What the agent may know and do

- **Observation:** its own position and the goal's, as (x, y, level), and which moves are open from its current cell. It never sees the map: anything beyond the current cell it has to remember.
- **Access:** Claude gets `observe`, `step` and `episode_status`. The harness alone calls `reset`, so Claude cannot restart an episode or choose a maze.
- **Memory:** its own conversation. Nothing is summarised or edited between calls.
- **The episode ends** when the goal is reached, when 120 moves are used, when Claude stops calling tools, on a refusal, after 200 API calls, or at the spending cap.

## Agents and references

| Agent | Knows | Experience before the episode |
|---|---|---|
| **Claude** (Messages API, tool use) | The observations above; its context as memory | None (zero-shot) |
| **Depth-first explorer** | The same observations; perfect memory of visited cells; moves to the open, unvisited neighbour nearest the goal, backtracks from dead ends | None |
| **Breadth-first search** | The whole map | None; an oracle with information Claude does not have |
| **Random walk** | Nothing; uniform over north, south, east, west | 1,000 seeded walks per maze |
| **Q-learning, Dyna-Q** (context only) | Their cell index; rewards; no coordinates, no goal position | Trained on each maze from scratch, 1,000 episodes |

The depth-first explorer is the fair planning reference: it gets exactly Claude's information and a perfect memory. Breadth-first search shows what full information buys.

## The references, measured

From `python -m experiments.claude_navigation references` ([`results/references.csv`](results/references.csv), [`results/rl_per_maze.csv`](results/rl_per_maze.csv)):

| Agent | Reached the goal within 120 moves | Moves (median, range) | Excess over the shortest route |
|---|---|---|---|
| Breadth-first search, whole map | 10 of 10 mazes | 33 (24–46) | 0 |
| Depth-first explorer, Claude's information | 10 of 10 mazes | 47 (24–118) | median 4%; shortest on 4 mazes; worst 3.7× (118 moves for 32) |
| Random walk | 2 of 10,000 walks | – | – |
| **Claude** | **not yet run** | – | – |

Per-maze reinforcement learning, for context only (training budgets, not zero-shot scores):

- **Q-learning:** first reaches the goal after a median 526 real moves, and its greedy route converges after a median 23.7k.
- **Dyna-Q:** with the value-propagation experiment's frozen setting, 50 planning updates per move, those medians are 621 and 2.1k.

Both converged on all 10 mazes.

## Measured per episode

Reached the goal; moves used; excess path length (moves ÷ shortest route − 1, for episodes that reached the goal); wall bumps and refused calls (invalid actions); tool calls; API calls; latency per API call and per episode; input, output, cache-write and cache-read tokens; and the estimated cost from those tokens. Each episode's full transcript is kept: the system prompt, the tool schemas, every response's content blocks (including summarised thinking), every tool result, the usage, the latency and the request id.

## How Claude is called

[`experiments/claude_navigation.py`](../../maze_runner_rl/experiments/claude_navigation.py) runs a manual tool-use loop on the Messages API with the official Python SDK. It owns the loop so the environment, not the model, decides when an episode ends, and so every call is accounted for.

- **Model:** `claude-opus-5-5` by default (`--model` to change it). Effort `medium`, set explicitly. Adaptive thinking with summarised display, so the transcripts show the reasoning summaries. `max_tokens` 16,000.
- **Tools:** the three MCP tools, sorted by name, so the prompt prefix never changes. Parallel tool calls are allowed; results go back in one message.
- **Prompt caching:** automatic, through top-level `cache_control`. The conversation grows every call and the cached prefix makes that affordable.
- **Model fallbacks:** off. A refusal ends the episode and is recorded, so every result comes from the model named. The served model is recorded for every call.
- **Endpoint:** `https://api.anthropic.com` unless `--base-url` says otherwise, so an `ANTHROPIC_BASE_URL` inherited from another tool is not used by accident.

The system prompt is fixed and stored, with its SHA-256, in [`task.json`](task.json).

## Cost and budget

Cost is estimated from each response's token usage and list prices: $4 / $20 per million input / output tokens for `claude-opus-5-5`, cache writes at 1.25× input and cache reads at $0.20 per million. These are the prices in the Claude API reference bundled with Claude Code, cached 2026-10-06.

A hard cap (`--max-usd`, default $20) is checked before every API call. Spend is kept on disk, so an interrupted run resumes against the same cap and skips finished episodes. A rough estimate for this task is $0.3 to $1 per episode, about $10 to $30 for all 30 episodes on `claude-opus-5-5`. `claude-sonnet-5-5` is about half that and `claude-haiku-5-5` a small fraction; `--model` selects either.

## Run it

From `maze_runner_rl/`, with the repository's virtual environment:

```bash
python -m pip install -r requirements-claude.txt
python -m experiments.claude_navigation references                 # no API calls: the references and task.json
python -m experiments.claude_navigation claude --max-usd 20        # needs ANTHROPIC_API_KEY; writes results/claude/<model>/
python -m experiments.claude_navigation summary
```

`python -m mcp_maze.server` starts the MCP server on its own (stdio), for any MCP client.

## What is and is not compared

- **Claude and the depth-first explorer** get the same observations, so their success rates and excess path lengths compare directly. The explorer has perfect memory by construction; Claude has its context.
- **Breadth-first search** has the whole map; it bounds what any agent could do.
- **Q-learning and Dyna-Q** learn one maze from rewards over hundreds of episodes and cannot use coordinates or the goal's position. Their numbers are a training budget on that maze, not a zero-shot score. They are reported for context, not ranked against Claude.

## Tests

[`tests/test_mcp_maze.py`](../../maze_runner_rl/tests/test_mcp_maze.py) covers:

- **Session rules:** seeded and isolated sessions, observations that show only the local cell and the goal, bumps, the end of an episode, refused calls, bad settings.
- **MCP server:** its four tools, the direction enum, and that every result matches the session rules.
- **Harness:** driven by a scripted, Messages-API-shaped fake client. This covers token and cost accounting, transcripts, stopping when the session ends, the spending cap and resuming.

No test calls the API.
