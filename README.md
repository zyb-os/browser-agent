# Browser Agent

An AI-powered browser agent that takes natural-language tasks, opens a real Chromium browser, and navigates the web like a human. It can run standalone in an interactive REPL or connect to an [agent-orchestrator](https://github.com/zyb-os/agent-orchestrator) to receive tasks over WebSocket.

## Requirements

- Python 3.10+
- An [Anthropic API key](https://console.anthropic.com)

## Setup

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Install the Chromium browser
python -m playwright install chromium

# 3. Set your API key
export ANTHROPIC_API_KEY=your_api_key_here
```

## Running

### Orchestrator mode (default)

Registers with the agent orchestrator and waits for tasks over WebSocket:

```bash
python main.py
```

Override the orchestrator URL:

```bash
python main.py --orchestrator-url http://my-server:8000
# or
export ORCHESTRATOR_URL=http://my-server:8000
python main.py
```

### Interactive mode

Opens a local REPL — type tasks directly in the terminal without an orchestrator:

```bash
python main.py -i
```

```
You: find me the best powerbank under $50 for hiking
You: search amazon for noise-cancelling headphones and compare the top 3
You: look up reviews for the latest MacBook Air
```

#### Interactive commands

| Command    | Description                    |
|------------|--------------------------------|
| `/profile` | View your learned preferences  |
| `/clear`   | Clear conversation history     |
| `/quit`    | Exit and close the browser     |

## How it works

1. **Vision loop** — The agent takes a screenshot, sends it to Claude, and receives a browser action (click, type, scroll, navigate). This repeats until the task is done.
2. **Self-learning fast path** — Successful sequences of clicks and key presses are cached per URL in `data/site_knowledge.json`. On repeat visits the cached steps are replayed without sending a screenshot to the LLM, reducing token cost. If a cached step fails, the agent falls back to full vision inference automatically.
3. **Profile learning** — Whenever you mention a budget, brand preference, or use case, it's saved to `data/profile.json` and reused in future sessions.
4. **Adaptive thinking** — Claude reasons step-by-step before each action, visible in the terminal output.

## Orchestrator integration

The agent implements the full [AGENT_MANIFEST.md](https://github.com/zyb-os/agent-orchestrator/blob/main/AGENT_MANIFEST.md) protocol:

- Registers the `browse_web` capability with input/output schemas
- Maintains a persistent WebSocket connection with heartbeats every 15 s
- Reconnects automatically with exponential backoff (up to 60 s)
- Reports status transitions: `starting → available → busy → draining → offline`
- Streams per-task metrics (completed, failed, avg response time, uptime)
- Reads the Anthropic API key from orchestrator settings; falls back to `ANTHROPIC_API_KEY` env var
- Uses a stable agent ID (`.agent_id`) so re-registrations upsert rather than duplicate the record
- For `ask_user` tool prompts, returns a structured `output_data.followup_request` in `task_response`
  so the planner can either resolve from Cortex context or route the question to the user.

## Model

Set in `agent.py`:

```python
MODEL = "claude-sonnet-4-6"   # faster, cheaper
# MODEL = "claude-opus-4-6"   # more capable, higher cost
```

## Project structure

```
browser-agent/
├── main.py                  # Entry point (orchestrator by default, -i for REPL)
├── agent.py                 # Claude vision + agentic tool loop
├── browser.py               # Playwright Chromium controller
├── orchestrator_client.py   # Agent-orchestrator WebSocket client
├── site_knowledge.py        # Per-URL action cache (self-learning fast path)
├── profile.py               # User profile manager
├── tools.py                 # Tool definitions and executor
├── data/
│   └── profile.json         # Persisted user preferences
└── requirements.txt
```
