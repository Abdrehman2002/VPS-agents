# Deploy Nadia on your VPS — step by step

Two paths. **Pick A if your VPS already runs a LiveKit server** (your Hostinger box already does:
`livekit-server` + `redis` + `caddy` on `livekit.vextriaai.com`). Pick B for a brand-new VPS.

The agent is **CPU-only** (all cloud APIs) — it runs fine on a small VPS next to LiveKit.

---

## Files in this folder
| File | Purpose |
|---|---|
| `agent.py` | The Nadia complaint agent |
| `Dockerfile` | Builds the agent image |
| `docker-compose.yml` | Redis + LiveKit + Caddy + agent (full stack) |
| `livekit.yaml` | LiveKit server config (Path B) |
| `Caddyfile` | TLS reverse proxy (Path B) |
| `requirements.txt`, `.env.example` | deps + config template |

---

## 0. Get the files onto the VPS
From your laptop (PowerShell), copy the folder up (replace IP):
```powershell
scp -r "C:\Users\Thinkbook 16 G6\Desktop\VPS agents" root@187.77.117.11:/root/nadia
```
Then SSH in: `ssh root@187.77.117.11` and `cd /root/nadia`.
*(Or push this folder to a git repo and `git clone` it on the VPS.)*

Make sure Docker is installed: `docker --version` (your Hostinger box already has it).

---

## PATH A — VPS already has LiveKit (recommended for your Hostinger box)

Your existing LiveKit server is on `localhost:7880` with keys in `/opt/livekit/livekit.yaml`
(`42cd3baa67acb1a56fb272e8199ea07714e095aa` / `c3f8…`). The agent just connects to it locally —
**no second LiveKit, no cross-network blocking** (the agent is on the same box).

**1. Create `.env`** (agent talks to the local LiveKit over plain ws):
```bash
cat > /root/nadia/.env << 'EOF'
LIVEKIT_URL=ws://localhost:7880
LIVEKIT_API_KEY=42cd3baa67acb1a56fb272e8199ea07714e095aa
LIVEKIT_API_SECRET=c3f83951a178581c221fb9b7d203a89686194d651c9ccf5cec6cd659ebc3869d

OPENAI_API_KEY=sk-proj-...your key...
LLM_MODEL=gpt-4o
DEEPGRAM_API_KEY=...your key...

UPLIFTAI_API_KEY=sk_api_ffdf9436d057b648d56c7dc0c092ac2cd0de96910cdc4f748f6f635fbb3b4a00
UPLIFT_VOICE_ID=helpdesk-agent
UPLIFT_OUTPUT_FORMAT=MP3_22050_128

AGENT_NAME=nadia
HELPLINE=111-42-5000
DASHBOARD_URL=https://vextriadashboard.vercel.app
EOF
nano /root/nadia/.env    # paste your real OpenAI + Deepgram keys
```

**2. Build & run ONLY the agent** (don't touch the existing LiveKit/Caddy):
```bash
cd /root/nadia && docker compose up -d --build --no-deps nadia
```

**3. Verify it registered:**
```bash
docker compose logs -f nadia | grep -iE "registered worker|TTS: Uplift|error"
```
You want `registered worker`. Done — skip to **Test it** below.

---

## PATH B — fresh VPS (full stack: LiveKit + Caddy + Redis + agent)

**1. DNS:** point `livekit.yourdomain.com` (A record) at the VPS public IP.

**2. Open firewall ports** (cloud panel and/or ufw): TCP **80, 443, 7881**, UDP **50000-60000**.
```bash
ufw allow 80,443,7881/tcp && ufw allow 50000:60000/udp
```

**3. Generate LiveKit keys:**
```bash
docker run --rm livekit/livekit-server generate-keys
```
Put the key+secret into **`livekit.yaml`** (the `keys:` line) AND into `.env`
(`LIVEKIT_API_KEY` / `LIVEKIT_API_SECRET`).

**4. Edit `Caddyfile`** — replace `livekit.yourdomain.com` with your real domain.

**5. Create `.env`** (same as Path A, but):
```
LIVEKIT_URL=ws://localhost:7880
LIVEKIT_API_KEY=<from generate-keys>
LIVEKIT_API_SECRET=<from generate-keys>
```
plus your OpenAI / Deepgram / Uplift / Dashboard values.

**6. Launch everything:**
```bash
cd /root/nadia && docker compose up -d --build
```

**7. Verify:**
```bash
docker compose ps
docker compose logs livekit | tail -20      # LiveKit started
curl -s https://livekit.yourdomain.com/      # should say OK (Caddy+TLS working)
docker compose logs -f nadia | grep -i "registered worker"
```

---

## Test it (both paths)

**Quickest:** open the **LiveKit Agents Playground** → https://agents-playground.livekit.io →
enter your `LIVEKIT_URL` (the public `wss://livekit.yourdomain.com`) + a token, connect, and talk.

**Generate a test token** on the VPS:
```bash
docker run --rm livekit/livekit-cli token create \
  --api-key <KEY> --api-secret <SECRET> \
  --identity tester --room test-room --join --valid-for 24h
```
Join `test-room` in the Playground (with the public wss URL) — Nadia should greet you in Urdu,
and complaints will POST to your dashboard.

**From your CRM:** point its LiveKit web client at `wss://livekit.yourdomain.com`, mint tokens with
the same key/secret server-side, and dispatch the agent named **`nadia`**.

---

## Operate
```bash
docker compose logs -f nadia          # live logs
docker compose restart nadia          # restart after editing .env
docker compose up -d --build nadia    # rebuild after editing agent.py
docker compose down                   # stop the stack
```

## Verify Uplift loaded (the earlier gotcha)
```bash
docker compose exec nadia python -c "from livekit.plugins import upliftai; print('uplift OK')"
```
If that errors, `python-socketio` is missing — it's already in `requirements.txt`, so a clean
`--build` fixes it.

## Data → dashboard
Each complaint POSTs to `DASHBOARD_URL/api/complaints` and a full call summary to
`DASHBOARD_URL/api/calls` at call end. Make sure those endpoints exist on your dashboard
(or set `CRM_WEBHOOK_URL` to any webhook that accepts JSON).
