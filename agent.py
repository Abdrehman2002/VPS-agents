"""
Nadia — HBL Microfinance Bank · Complaint & Resolution voice agent (LiveKit)
============================================================================
Production LiveKit agent, designed to run on a VPS behind a self-hosted (or Cloud)
LiveKit server, web-callable.

Stack (all cloud APIs — no GPU needed):
  • LLM : OpenAI  (gpt-4o / gpt-4.1)            — reasoning + flow
  • STT : Deepgram nova-3 (Urdu)                — speech to text
  • TTS : Uplift AI (Urdu-native)              — text to speech  ← NOT ElevenLabs

Design: one comprehensive system prompt (global prompt + node flow folded in)
driven by the LLM, plus deterministic function tools for the things that must
NOT be hallucinated — reference numbers, priority→SLA mapping, complaint logging.

Run:  python agent.py dev      (connects to the LiveKit server in .env)
"""
import os
import json
import random
import string
import logging
import uuid
from datetime import datetime, timezone, timedelta
from typing import Annotated

import aiohttp
from dotenv import load_dotenv

from livekit import agents, rtc
from livekit.agents import (
    Agent,
    AgentSession,
    JobContext,
    WorkerOptions,
    function_tool,
    RoomInputOptions,
    AutoSubscribe,
)
from livekit.agents import llm, stt, tts
from livekit.plugins import openai as lk_openai
from livekit.plugins import deepgram, silero

# Uplift AI TTS plugin — optional import so a missing dep never crashes startup
try:
    from livekit.plugins import upliftai
    HAS_UPLIFTAI = True
except Exception:
    HAS_UPLIFTAI = False

# Multilingual turn detector (optional)
try:
    from livekit.plugins.turn_detector.multilingual import MultilingualModel
    HAS_TURN_DETECTOR = True
except Exception:
    HAS_TURN_DETECTOR = False

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("hbl-nadia")

# ── Config ────────────────────────────────────────────────────────────────────
OPENAI_API_KEY   = os.getenv("OPENAI_API_KEY", "")
DEEPGRAM_API_KEY = os.getenv("DEEPGRAM_API_KEY", "")
LLM_MODEL        = os.getenv("LLM_MODEL", "gpt-4o")
ANALYSIS_MODEL   = os.getenv("ANALYSIS_MODEL", "gpt-4o-mini")

UPLIFTAI_API_KEY     = os.getenv("UPLIFTAI_API_KEY", "")
UPLIFT_VOICE_ID      = os.getenv("UPLIFT_VOICE_ID", "helpdesk-agent")  # Uplift Urdu voice
UPLIFT_OUTPUT_FORMAT = os.getenv("UPLIFT_OUTPUT_FORMAT", "MP3_22050_128")
USE_UPLIFT           = bool(UPLIFTAI_API_KEY) and HAS_UPLIFTAI

DASHBOARD_URL    = os.getenv("DASHBOARD_URL", "")
CRM_WEBHOOK_URL  = os.getenv("CRM_WEBHOOK_URL", "")
# CRM API (Fastify backend) — Nadia files complaints via the voice-bot LiveKit endpoint
CRM_API_URL       = os.getenv("CRM_API_URL", "")             # e.g. http://127.0.0.1:3000
CRM_INGEST_SECRET = os.getenv("CRM_INGEST_SECRET", "")       # must match API's LIVEKIT_INGEST_SECRET
CRM_TENANT_ID     = os.getenv("CRM_TENANT_ID", "")           # tenant UUID
# Supabase (legacy/optional — no longer used for complaint creation)
SUPABASE_URL         = os.getenv("SUPABASE_URL", "")
SUPABASE_SERVICE_KEY = os.getenv("SUPABASE_SERVICE_KEY", "")
AGENT_NAME       = os.getenv("AGENT_NAME", "nadia")
HELPLINE        = os.getenv("HELPLINE", "111-42-5000")

# Priority → SLA spoken commitment (Urdu script) + working-day count for dates
SLA_TEXT = {
    "P1": "چوبیس گھنٹے کے اندر",
    "P2": "تین ورکنگ ڈیز میں",
    "P3": "سات ورکنگ ڈیز میں",
    "P4": "دس سے پندرہ ورکنگ ڈیز میں",
}
SLA_DAYS = {"P1": 1, "P2": 3, "P3": 7, "P4": 15}

VALID_CATEGORIES = {
    "loan_issue", "account_issue", "staff_complaint",
    "digital_banking", "fraud", "branch_service", "other",
}


def _norm_priority(p: str) -> str:
    """Accept 'P1', 'P1_critical', 'critical', etc. → 'P1'..'P4'."""
    p = (p or "").strip().lower()
    if p.startswith("p1") or "critical" in p:
        return "P1"
    if p.startswith("p2") or "high" in p:
        return "P2"
    if p.startswith("p4") or "low" in p:
        return "P4"
    return "P3"  # sensible default (medium)


def _gen_reference() -> str:
    """MFB-<4 digits>-<1 capital letter>, e.g. MFB-4271-R."""
    digits = "".join(random.choices(string.digits, k=4))
    letter = random.choice(string.ascii_uppercase)
    return f"MFB-{digits}-{letter}"


# ── System prompt (global prompt + node flow folded into one) ──────────────────
def build_system_prompt() -> str:
    return f"""Tum Nadia ho — HBL Microfinance Bank ki Complaint aur Resolution specialist. Tumhara kaam complaints professionally sunna, register karna, reference number dena, aur clear resolution timeline commit karna hai. HAMESHA empathetic raho — customer frustrated hai, usse pehle validate karo, phir solution do.

You are speaking OUT LOUD on a phone call through an Urdu text-to-speech voice.
CRITICAL LANGUAGE RULE: Reply ONLY in natural, conversational Urdu written in
proper URDU SCRIPT (اردو رسم الخط). NEVER write Urdu words in Roman/Latin letters —
Roman text is pronounced with an English accent and sounds wrong. For example write
«جی بالکل، میں آپ کی بات سمجھ رہی ہوں» — never "ji bilkul, main aap ki baat...".

PRONUNCIATION & SPEAKING RULES (CRITICAL — this is voice):
- بول چال والی نرم اردو استعمال کریں: «ٹھیک ہے»، «بالکل»، «جی ہاں»، «شکریہ» (شکریہ صرف آخر میں)۔
- «السلام علیکم» صرف شروع میں۔
- نمبر اور رقم آہستہ اور واضح بولیں؛ حساس نمبر اور ریفرنس نمبر حرف بہ حرف اردو میں دہرائیں۔
- آہستہ، سکون اور ٹھہراؤ کے ساتھ بولیں، جملوں کے درمیان مختصر وقفہ دیں — کبھی جلدی میں نہ بولیں۔
- کبھی hyphen، bullet، asterisk یا markdown استعمال نہ کریں — یہ آواز خراب کرتے ہیں۔ صرف بہتے ہوئے جملے۔
- ایک وقت میں صرف ایک سوال پوچھیں، جواب کا انتظار کریں، کسٹمر کو جلدی نہ کرائیں۔
- ہر جواب مختصر اور فطری رکھیں — جیسے ایک حقیقی انسان بات کرتا ہے۔

PACING (CRITICAL — your voice sounds too fast otherwise):
- بہت آہستہ بولیں۔ ہر چھوٹے جملے کے بعد فُل سٹاپ «۔» لگائیں۔
- لمبے جملے کبھی نہ بولیں — ہر سانس میں صرف چھ سے دس الفاظ۔ پھر رک کر اگلا جملہ شروع کریں۔
- جملے کے درمیان کوما «،» بکثرت لگائیں، ہر دو تین الفاظ کے بعد، تاکہ بولنے میں ٹھہراؤ آئے۔
- مثال (تیز اور غلط): «جی ٹھیک ہے میں آپ کی بات سمجھ گئی ہوں اور آپ کی شکایت درج کر رہی ہوں۔»
- مثال (سست اور صحیح): «جی۔ ٹھیک ہے۔ میں، آپ کی بات، سمجھ گئی ہوں۔ آپ کی شکایت، درج کر رہی ہوں۔»
- نمبر اور رقم کے ہر ہندسے کے بعد چھوٹا وقفہ — مثلاً «پانچ۔ ہزار۔ روپے۔»

EMPATHY (freely use, especially with frustrated callers):
- "Main samajhti hoon yeh kitna mushkil hai."
- "Aap ki takleef ke liye main maafi chahti hoon."
- "Yeh bilkul theek nahi tha — hum ise seriously le rahe hain."
- "Aap ne bilkul sahi kiya call karke."

═══════════════════════════════════════════════════════════════════════════════
CONVERSATION FLOW (follow this order; adapt naturally, don't read it robotically):

1. OPENING: "Assalam-u-Alaikum, main Nadia hoon, HBL Microfinance Bank Complaint Resolution se. Main aap ki madad ke liye hoon — please batayein kya masla hai?" Phir SUNO.

2. ROUTE based on what they say:
   - Agar FRAUD / unauthorized transaction / account hack → go straight to FRAUD PROTOCOL (urgent).
   - Agar woh kehte hain pehle se complaint ki thi ya reference number hai → EXISTING COMPLAINT.
   - Agar naya masla/complaint → VERIFY then categorize.
   - Agar sirf ek sawaal (rates, process, policy) → jawab do, phir poochho koi complaint bhi hai.

3. VERIFY (ek ek sawaal, frustrated caller ke liye fast):
   a. "Aap ka poora naam kya hai?"
   b. "Aap ka account number ya CNIC — jo available ho?" (na ho toh: "CNIC se bhi chalega"; woh bhi na ho toh naam aur city se proceed karo)
   c. "Aap kaunse sheher mein hain?"
   Incomplete info pe bhi aage barho — register karna zyada zaroori hai.

4. CATEGORIZE & COLLECT DETAILS (empathy pehle, phir relevant detail — ek ek sawaal):
   • LOAN_ISSUE: kaunsa loan, amount, disbursement delay/terms/recovery-agent harassment/forced insurance? kab se? officer ka naam? document?
     - Recovery agent harassment ya physical threat → P1.
   • ACCOUNT_ISSUE: account blocked/frozen, wrong deduction (kitna amount? kab? transaction ID?), balance discrepancy, ATM. Notification mila tha?
     - Deduction > five thousand → P2; < five thousand → P3.
   • STAFF_COMPLAINT (sensitive): branch + city, date/time, staff naam/designation, exactly kya hua, witness, document. Bribery/corruption maanga gaya? → "Yeh serious hai, hamari integrity committee handle karegi." Staff misconduct = P2 minimum.
   • DIGITAL_BANKING: PEHLE basic troubleshoot try karo (login: internet/app-update/restart/Forgot-PIN; OTP: network/registered-number/wait). LEKIN agar "transaction failed, paise kat gaye" → troubleshoot mat karo, yeh complaint hai (amount/date/transaction-ID). Resolved by troubleshoot → no complaint needed.
   • BRANCH_SERVICE: branch location, date/time, kya hua (wait/rudeness/wrong info), staff naam. = P3.
   • OTHER: dhyan se suno, masla + kab se + pehle koi action liya + proof.

5. FRAUD PROTOCOL (P1 — calm aur fast):
   a. Calm karo: "Ghabrayein nahi — main abhi help karti hoon. Aap ne bilkul sahi kiya call karke."
   b. Immediate action: "Pehle ABHI yeh karein: HBL Mobile app kholein, Settings, phir Block Card ya Account — ya nearest branch jaayein CNIC le ke."
   c. Details: unauthorized transaction ka amount/date? koi suspicious call/SMS? kisi ne OTP/PIN/card details maange? kab notice kiya?
   d. Warn: "Hamari bank kabhi OTP, PIN, ya password phone pe nahi maangti. Agar koi maange toh woh fraud hai. Kisi se share mat karein."
   e. "Aap local police station pe FIR bhi file kar sakte hain — reference number helpful hoga."

6. After details for ANY complaint → assign priority INTERNALLY (caller ko P-label mat do, sirf clear timeline batao) using the PRIORITY MATRIX, then CALL THE register_complaint TOOL. The tool returns the official reference number and SLA — use EXACTLY what the tool returns.

7. After the tool returns: reference number letter-by-letter spell karo, ek baar repeat karo ("note kar liya?"), phir exact SLA commitment do.

8. SUMMARY: "Main summary repeat karti hoon: [naam] ji, aap ki [category] complaint register ho gayi. Reference number [number]. [SLA] mein hamari team rabta karegi. Sab theek hai?"

9. "Koi aur masla hai?" — agar haan, fresh details lo aur DOBARA register_complaint call karo (alag reference number milega). Agar nahi → CLOSE.

10. CLOSE: "[Naam] ji, aap ka time aur hum par trust karne ka shukriya. Hamari team aap ke reference number pe jald rabta karegi. Urgent ho toh {HELPLINE} pe call karein — Monday se Friday, nau baje subah se chhe baje shaam. Shukriya, Allah Hafiz!"

═══════════════════════════════════════════════════════════════════════════════
PRIORITY MATRIX (decide internally, pass to the tool as P1/P2/P3/P4):
  P1 (24 hours): fraud / unauthorized transaction / account hacked; account wrongly blocked causing active financial loss; recovery agent physical threats or harassment.
  P2 (3 working days): loan disbursement delayed >7 days after approval; wrong deduction above five thousand; staff misconduct or bribery; account frozen without notification.
  P3 (7 working days): app technical issues (not fixed by troubleshoot); wrong deduction below five thousand; branch service complaint; account opening delay; ATM issues.
  P4 (10–15 working days): general policy dissatisfaction; minor profit rate dispute; document return delay; general feedback.

SLA WORDING (the tool gives you the right one — say it as returned):
  P1 → "24 ghante ke andar hamari specialized team aap se rabta karegi."
  P2 → "3 working days mein resolution ya update milegi."
  P3 → "7 working days mein hamari team jawab degi."
  P4 → "10 se 15 working days mein response milega."

EXISTING COMPLAINT: "Aap ka reference number kya hai?" → "Note kar liya. Existing complaint ka live status main access nahi kar sakti — {HELPLINE} pe call karein ya branch jaayein reference number le ke. Kya aap ise escalate karna chahte hain ya nayi update?" Agar escalate → naya complaint register karo (verify → category → tool).

CUSTOMER RIGHTS: Har customer ko complaint register karne aur reference number ka haq hai. Agar bank 45 din mein resolve na kare toh State Bank Banking Mohtasib se bhi shikayat ho sakti hai.

HARD RULES:
- Reference number HAMESHA register_complaint tool se aata hai — KABHI khud invent mat karo.
- Caller ki gender maloom nahi — "aap" use karo, "bhai/behen/sahib/madam" nahi.
- Tum aurat ho (Nadia) — apne liye feminine verbs ("karti hoon", "samajhti hoon").
- OTP/PIN/password kabhi mat maango.
- Pareshan caller → slow down, pehle feelings acknowledge karo, phir badho."""


# ── Complaint registration tool (deterministic) ────────────────────────────────
class NadiaAgent(Agent):
    def __init__(self, system_prompt: str, caller_phone: str | None = None):
        super().__init__(instructions=system_prompt)
        self.caller_phone = caller_phone
        self.complaints: list[dict] = []       # all complaints filed this call (for analytics/CRM)
        self._voice_call_ids: list[str] = []   # CRM voice_bot_call ids → updated with transcript at call end

    async def on_enter(self) -> None:
        # Fixed, correctly-pronounced opening (not LLM-generated) so it's identical every call.
        self.session.say(
            "السلام علیکم، میں نادیہ ہوں، ایچ بی ایل مائیکرو فنانس بینک کمپلینٹ "
            "ریزولوشن سے۔ میں آپ کی مدد کے لیے حاضر ہوں۔ برائے مہربانی بتائیے، آپ کو کیا مسئلہ درپیش ہے؟",
            allow_interruptions=True,
        )

    @function_tool
    async def register_complaint(
        self,
        caller_name: Annotated[str, "Caller's name in Roman script. If unknown, 'Not provided'."],
        complaint_category: Annotated[
            str,
            "One of: loan_issue, account_issue, staff_complaint, digital_banking, fraud, branch_service, other",
        ],
        priority: Annotated[str, "One of: P1, P2, P3, P4 (decide via the priority matrix)."],
        description: Annotated[str, "1–2 sentence summary of the complaint in English."],
        account_or_cnic: Annotated[str, "Account number or CNIC if given, else 'Not provided'."] = "Not provided",
        caller_city: Annotated[str, "Caller's city if given, else 'Not provided'."] = "Not provided",
        fraud_amount: Annotated[str, "Amount involved for fraud cases, else 'Not applicable'."] = "Not applicable",
    ) -> str:
        """Register the complaint. Call this ONLY after collecting the details and deciding the
        priority. Generates the official reference number and SLA. Returns the reference number
        and the exact SLA wording to read back to the caller."""
        cat = complaint_category.strip().lower()
        if cat not in VALID_CATEGORIES:
            cat = "other"
        pri = _norm_priority(priority)
        sla = SLA_TEXT[pri]

        # Create the ticket in the Itqan CRM and read back the official TKT number.
        ref = await self._create_crm_ticket(caller_name, cat, pri, description, fraud_amount)

        record = {
            "reference_number": ref,
            "caller_name": caller_name,
            "account_or_cnic": account_or_cnic,
            "caller_city": caller_city,
            "complaint_category": cat,
            "complaint_priority": pri,
            "complaint_description": description,
            "fraud_amount": fraud_amount,
            "sla_committed": sla,
            "caller_phone": self.caller_phone,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        self.complaints.append(record)
        logger.info(f"Complaint registered: {ref} [{cat}/{pri}] {description!r}")

        return (
            f"COMPLAINT REGISTERED. Reference number: {ref}. Priority: {pri}. "
            f"SLA to tell the caller: '{sla}'. "
            f"Now: read the reference number to the caller clearly in Urdu, character by character "
            f"(مثلاً «ٹی کے ٹی، صفر صفر صفر صفر ایک»), repeat it once, then state the SLA exactly in Urdu."
        )

    async def _create_crm_ticket(self, name: str, category: str, priority: str,
                                 description: str, fraud_amount: str) -> str:
        """Create the complaint via the CRM API's voice-bot LiveKit endpoint and return
        its TKT number. Falls back to a local reference if the CRM is unreachable."""
        if CRM_API_URL and CRM_TENANT_ID:
            url = (f"{CRM_API_URL.rstrip('/')}/api/v1/voice-bot/livekit/complaint"
                   f"?tenantId={CRM_TENANT_ID}")
            headers = {"Content-Type": "application/json"}
            if CRM_INGEST_SECRET:
                headers["Authorization"] = f"Bearer {CRM_INGEST_SECRET}"
            payload = {
                "reporterName":  name,
                "reporterPhone": self.caller_phone,
                "category":      category,
                "priority":      priority,
                "subject":       description[:120],
                "description":   description,
                "fraudAmount":   fraud_amount,
                "callId":        getattr(self, "room_name", None),
            }
            try:
                async with aiohttp.ClientSession() as s:
                    async with s.post(url, json=payload, headers=headers,
                                      timeout=aiohttp.ClientTimeout(total=15)) as r:
                        data = await r.json()
                        if isinstance(data, dict) and data.get("ticketNumber"):
                            if data.get("voiceCallId"):
                                self._voice_call_ids.append(data["voiceCallId"])
                            return data["ticketNumber"]
                        logger.error(f"CRM complaint create returned: {data}")
            except Exception as e:
                logger.error(f"CRM complaint create failed: {e}")
        return _gen_reference()  # fallback if CRM unreachable


# ── Backend posting (dashboard + CRM) ──────────────────────────────────────────
async def _post_complaint(record: dict) -> None:
    targets = []
    if DASHBOARD_URL:
        targets.append(f"{DASHBOARD_URL.rstrip('/')}/api/complaints")
    if CRM_WEBHOOK_URL:
        targets.append(CRM_WEBHOOK_URL)
    if not targets:
        return
    async with aiohttp.ClientSession() as http:
        for url in targets:
            try:
                async with http.post(url, json=record, timeout=aiohttp.ClientTimeout(total=10)) as r:
                    logger.info(f"POST {url} -> HTTP {r.status}")
            except Exception as e:
                logger.error(f"POST {url} failed: {e}")


# ── Post-call analysis (matches the Retell post_call_analysis_data fields) ──────
ANALYSIS_PROMPT = """You are a call analytics engine for HBL Microfinance Bank complaint calls.
Analyze the transcript and return a JSON object with EXACTLY these fields:
- caller_name: string or "Not provided"
- account_number: string (account number or CNIC) or "Not provided"
- caller_city: string or "Not provided"
- complaint_category: one of loan_issue, account_issue, staff_complaint, digital_banking, fraud, branch_service, other
- complaint_priority: one of P1_critical, P2_high, P3_medium, P4_low
- complaint_description: 1-2 sentence summary
- reference_number: e.g. "MFB-4271-R" or "Not generated"
- sla_committed: e.g. "24 hours", "3 working days"
- fraud_amount: amount for fraud cases, else "Not applicable"
- caller_sentiment: one of calm, frustrated, angry, satisfied
- call_summary: 2-3 sentences (complaint, priority, reference number, next step)"""


async def analyze_call(transcript: str) -> dict:
    if not OPENAI_API_KEY or not transcript.strip():
        return {}
    try:
        import openai as _openai
        client = _openai.AsyncOpenAI(api_key=OPENAI_API_KEY)
        resp = await client.chat.completions.create(
            model=ANALYSIS_MODEL,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": ANALYSIS_PROMPT},
                {"role": "user", "content": f"Analyze this call transcript:\n\n{transcript}"},
            ],
            max_tokens=500,
            temperature=0,
        )
        return json.loads(resp.choices[0].message.content)
    except Exception as e:
        logger.error(f"Post-call analysis failed: {e}")
        return {}


# ── Pipeline builders (cloud APIs, key-guarded) ────────────────────────────────
def build_llm():
    if not OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY is required for the LLM.")
    # Primary model + a cheaper fallback, both OpenAI.
    return llm.FallbackAdapter([
        lk_openai.LLM(model=LLM_MODEL),
        lk_openai.LLM(model="gpt-4o-mini"),
    ])


def build_stt():
    if not DEEPGRAM_API_KEY:
        raise RuntimeError("DEEPGRAM_API_KEY is required for STT.")
    return deepgram.STT(
        model="nova-3",
        language="ur",          # Urdu
        punctuate=True,
        interim_results=True,
        smart_format=True,
    )


def build_tts():
    engines = []
    if USE_UPLIFT:
        logger.info(f"TTS: Uplift AI (voice={UPLIFT_VOICE_ID}, {UPLIFT_OUTPUT_FORMAT})")
        engines.append(upliftai.TTS(
            voice_id=UPLIFT_VOICE_ID,
            output_format=UPLIFT_OUTPUT_FORMAT,
        ))
    elif not HAS_UPLIFTAI:
        logger.warning("Uplift plugin not importable (pip install livekit-plugins-upliftai python-socketio).")
    # Last-resort fallback so the agent is never voiceless (OpenAI TTS). NOT ElevenLabs.
    if OPENAI_API_KEY:
        engines.append(lk_openai.TTS(model="tts-1", voice="shimmer"))
    if not engines:
        raise RuntimeError("No TTS configured — set UPLIFTAI_API_KEY (+ the plugin) or OPENAI_API_KEY.")
    return engines[0] if len(engines) == 1 else tts.FallbackAdapter(engines)


def prewarm(proc: agents.JobProcess):
    proc.userdata["vad"] = silero.VAD.load(
        min_silence_duration=0.4,
        activation_threshold=0.3,
    )


def get_caller_phone(ctx: JobContext) -> str | None:
    try:
        for p in ctx.room.remote_participants.values():
            attrs = getattr(p, "attributes", {}) or {}
            num = attrs.get("sip.phoneNumber") or attrs.get("sip.from")
            if num:
                return num
    except Exception:
        pass
    return None


# ── Entrypoint ─────────────────────────────────────────────────────────────────
async def entrypoint(ctx: JobContext):
    await ctx.connect(auto_subscribe=AutoSubscribe.AUDIO_ONLY)
    call_start = datetime.now(timezone.utc)
    caller_phone = get_caller_phone(ctx)

    vad = ctx.proc.userdata.get("vad") or silero.VAD.load()

    session_kwargs: dict = dict(
        stt=build_stt(),
        llm=build_llm(),
        tts=build_tts(),
        vad=vad,
        preemptive_generation=True,
    )
    if HAS_TURN_DETECTOR:
        session_kwargs["turn_detection"] = MultilingualModel()

    session = AgentSession(**session_kwargs)
    nadia = NadiaAgent(system_prompt=build_system_prompt(), caller_phone=caller_phone)

    await session.start(
        agent=nadia,
        room=ctx.room,
        room_input_options=RoomInputOptions(close_on_disconnect=True),
    )

    # Post-call analysis + CRM at shutdown
    async def on_shutdown():
        try:
            transcript = ""
            try:
                hist = session.history.to_dict()
                transcript = "\n".join(
                    f"{m.get('role')}: {m.get('content')}"
                    for m in hist.get("items", []) if m.get("role") in ("user", "assistant")
                )
            except Exception:
                pass
            analysis = await analyze_call(transcript)

            # Attach the final transcript/summary to each CRM voice_bot_calls record
            if CRM_API_URL and CRM_TENANT_ID and nadia._voice_call_ids:
                dur = int((datetime.now(timezone.utc) - call_start).total_seconds())
                ce_url = (f"{CRM_API_URL.rstrip('/')}/api/v1/voice-bot/livekit/call-ended"
                          f"?tenantId={CRM_TENANT_ID}")
                ce_headers = {"Content-Type": "application/json"}
                if CRM_INGEST_SECRET:
                    ce_headers["Authorization"] = f"Bearer {CRM_INGEST_SECRET}"
                async with aiohttp.ClientSession() as http:
                    for vcid in nadia._voice_call_ids:
                        try:
                            await http.post(
                                ce_url,
                                json={
                                    "voiceCallId": vcid,
                                    "transcript": transcript,
                                    "summary": analysis.get("call_summary") if isinstance(analysis, dict) else None,
                                    "sentiment": analysis.get("caller_sentiment") if isinstance(analysis, dict) else None,
                                    "durationSeconds": dur,
                                },
                                headers=ce_headers, timeout=aiohttp.ClientTimeout(total=10),
                            )
                        except Exception as e:
                            logger.error(f"CRM call-ended POST failed: {e}")

            payload = {
                "agent": AGENT_NAME,
                "caller_phone": caller_phone,
                "started_at": call_start.isoformat(),
                "ended_at": datetime.now(timezone.utc).isoformat(),
                "duration_sec": (datetime.now(timezone.utc) - call_start).total_seconds(),
                "complaints": nadia.complaints,
                "analysis": analysis,
            }
            logger.info(f"Call summary: {json.dumps(payload, ensure_ascii=False)[:800]}")
            if DASHBOARD_URL or CRM_WEBHOOK_URL:
                async with aiohttp.ClientSession() as http:
                    for url in filter(None, [
                        f"{DASHBOARD_URL.rstrip('/')}/api/calls" if DASHBOARD_URL else None,
                        CRM_WEBHOOK_URL,
                    ]):
                        try:
                            await http.post(url, json=payload, timeout=aiohttp.ClientTimeout(total=10))
                        except Exception as e:
                            logger.error(f"Call payload POST failed: {e}")
        except Exception as e:
            logger.error(f"on_shutdown error: {e}")

    ctx.add_shutdown_callback(on_shutdown)


if __name__ == "__main__":
    agents.cli.run_app(
        WorkerOptions(
            entrypoint_fnc=entrypoint,
            prewarm_fnc=prewarm,
            agent_name=AGENT_NAME,   # dashboard dispatches by this name
        )
    )
