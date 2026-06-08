# Nadia — HBL Microfinance Bank Complaint Agent (LiveKit, VPS)

Production LiveKit voice agent that handles HBL MFB complaints in Urdu, web-callable.
Cloud APIs only — **no GPU required**, runs on any small VPS.

- **LLM:** OpenAI (`gpt-4o`)
- **STT:** Deepgram nova-3 (Urdu)
- **TTS:** Uplift AI (Urdu-native) — *not* ElevenLabs
- **Media:** LiveKit server (self-hosted on the VPS, or LiveKit Cloud)

## What it does
Greets → routes (fraud fast-path / existing complaint / new complaint / query) → verifies caller →
categorizes (loan / account / staff / digital / fraud / branch / other) → collects details with
empathy → assigns priority → **registers the complaint** → gives a reference number + SLA →
summarizes → handles additional complaints → closes. Post-call it runs analytics and POSTs
complaint + call data to your dashboard/CRM.

## Improvements over the original Retell flow
- **Deterministic reference numbers** — generated in code (`register_complaint` tool), never
  hallucinated by the LLM. Each complaint gets a unique `MFB-####-X`.
- **Consistent priority → SLA mapping** in code, so the spoken timeline always matches the priority.
- **Structured logging** — every complaint + a full call summary is POSTed to your backend.
- **Fixed, correctly-pronounced opening** (not LLM-generated → identical every call).
- **Robust fallbacks** — OpenAI model fallback, OpenAI TTS as last resort if Uplift is down.
- **Single maintainable prompt** instead of 18 hand-wired nodes — easier to tune.

## Setup (on the VPS)
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env && nano .env        # fill in all keys
python agent.py dev                       # connects to your LiveKit server
```

> **Important:** the Uplift plugin needs `python-socketio` (in requirements.txt). Without it the
> plugin silently fails to import and the agent falls back to OpenAI TTS. Verify with:
> `python -c "from livekit.plugins import upliftai; print('OK')"`

## Web calling
Point your token endpoint / frontend at the same `LIVEKIT_URL` + keys, and dispatch the agent by
name **`nadia`** (set in `WorkerOptions` / `AGENT_NAME`). Any LiveKit web client (or the LiveKit
Agents Playground) can then call it.

## Run it as a service (stays up)
```bash
# simplest: tmux
tmux new -s nadia -d 'cd /path/to/VPS-agents && source .venv/bin/activate && python agent.py dev 2>&1 | tee agent.log'
# production: a systemd unit or Docker — run `python agent.py start`
```

## Files
- `agent.py` — the agent (system prompt, flow, tools, builders, analytics)
- `requirements.txt` — dependencies (incl. `python-socketio` for Uplift)
- `.env.example` — config template

## Tuning notes
- **Voice:** Nadia is female — set `UPLIFT_VOICE_ID` to a professional female Urdu voice from
  platform.upliftai.org. Test it on a call.
- **Roman Urdu vs script:** the prompt speaks Roman Urdu (as in the original). Uplift is Urdu-native;
  if any words sound off, test feeding Urdu script for those terms and adjust the prompt.
- **Reference number readout:** the agent spells it letter-by-letter for clarity on the phone.
