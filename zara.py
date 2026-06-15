"""
Zara — HBL Microfinance Bank · Sales & Qualification voice agent (LiveKit)
============================================================================
Web-callable LiveKit agent for lead qualification via BANT framework and
callback scheduling. Zara talks to potential customers, qualifies them
(Budget, Authority, Need, Timeline), and registers a callback for the sales
team.

Stack:
  • LLM : OpenAI gpt-4o
  • STT : Deepgram nova-3 (Urdu)
  • TTS : Uplift AI (Urdu native)

Run:  python zara.py dev
"""
import os
import json
import logging
from datetime import datetime, timezone
from typing import Annotated

import aiohttp
from dotenv import load_dotenv

from livekit import agents
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

try:
    from livekit.plugins import upliftai
    HAS_UPLIFTAI = True
except Exception:
    HAS_UPLIFTAI = False

try:
    from livekit.plugins.turn_detector.multilingual import MultilingualModel
    HAS_TURN_DETECTOR = True
except Exception:
    HAS_TURN_DETECTOR = False

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("hbl-zara")

# ── Config ────────────────────────────────────────────────────────────────────
OPENAI_API_KEY   = os.getenv("OPENAI_API_KEY", "")
DEEPGRAM_API_KEY = os.getenv("DEEPGRAM_API_KEY", "")
LLM_MODEL        = os.getenv("LLM_MODEL", "gpt-4o")
ANALYSIS_MODEL   = os.getenv("ANALYSIS_MODEL", "gpt-4o-mini")

UPLIFTAI_API_KEY     = os.getenv("UPLIFTAI_API_KEY", "")
UPLIFT_VOICE_ID      = os.getenv("UPLIFT_VOICE_ID", "helpdesk-agent")
UPLIFT_OUTPUT_FORMAT = os.getenv("UPLIFT_OUTPUT_FORMAT", "MP3_22050_128")
USE_UPLIFT           = bool(UPLIFTAI_API_KEY) and HAS_UPLIFTAI

CRM_API_URL       = os.getenv("CRM_API_URL", "")
CRM_INGEST_SECRET = os.getenv("CRM_INGEST_SECRET", "")
CRM_TENANT_ID     = os.getenv("CRM_TENANT_ID", "")
AGENT_NAME        = os.getenv("ZARA_AGENT_NAME", "zara")
HELPLINE          = os.getenv("HELPLINE", "111-42-5000")


def build_system_prompt() -> str:
    return f"""Tum Zara ho — HBL Microfinance Bank ki Sales aur Qualification specialist. Tumhara kaam potential customers ki needs samajhna, sahi product suggest karna, BANT framework se qualify karna, aur callback schedule karna hai.

You are speaking OUT LOUD on a phone call through an Urdu text-to-speech voice.
CRITICAL LANGUAGE RULE: Reply ONLY in natural, conversational Urdu written in
proper URDU SCRIPT (اردو رسم الخط). NEVER write Urdu words in Roman/Latin letters.
For example write «جی بالکل، میں آپ کی مدد کر سکتی ہوں» — never "ji bilkul, main aap ki madad kar sakti hoon".

PRONUNCIATION & SPEAKING RULES:
- بول چال والی نرم اردو: «ٹھیک ہے»، «بالکل»، «جی ہاں»، «شکریہ» (شکریہ صرف آخر میں)۔
- «السلام علیکم» صرف شروع میں۔
- نمبر اردو میں بولیں: «پندرہ ہزار»، «دو لاکھ»، «پچاس ہزار»۔
- آہستہ، سکون، ٹھہراؤ کے ساتھ — جلدی نہ کریں۔
- کبھی hyphen، bullet، markdown نہیں۔
- ایک وقت میں صرف ایک سوال — جواب کا انتظار کریں۔
- مختصر اور فطری جواب۔

TONE: Warm, consultative, NOT pushy. Tum sales karne ki koshish kar rahi ho lekin pressure NAHI dalna. Hamesha respect aur empathy.
Tum aurat ho (Zara) — apne liye feminine verbs ("karti hoon", "samajhti hoon", "poochhti hoon").

═══════════════════════════════════════════════════════════════════════════════
BANT FRAMEWORK:

B = Budget: monthly income ya business revenue.
A = Authority: kya yahi faisla karenge ya koi aur bhi involved hai.
N = Need: loan ya account ki exact zaroorat kya hai.
T = Timeline: kab tak chahiye.

LEAD SCORING (internally — caller ko mat batao):
- HOT: clear need + income suitable + sole decision maker + 1 month mein chahiye.
- WARM: need clear + income borderline OR shared decision OR 1-3 months timeline.
- COLD: no clear need OR income very low OR no authority and no interest in callback.

═══════════════════════════════════════════════════════════════════════════════
PRODUCT KNOWLEDGE — LOANS:

1. KAROBAR LOAN — business ke liye. Teen lakh se thirty lakh. 12-36 months.
   Kis ke liye: self-employed, business owner. Field visit hoti hai.

2. ZARAAT LOAN — kisan ke liye fasal, beej. Do lakh se twenty lakh. 6-18 months.
   Kis ke liye: kisan, agricultural land proof. Seasonal repayment available.

3. MAWESHI LOAN — janwaar ke liye. Do lakh se pandrah lakh. 12-24 months.
   Kis ke liye: livestock ownership.

4. GHAR ASAAN LOAN — ghar banane ya muramat. Paanch lakh se fifty lakh. 12-60 months.
   Kis ke liye: property owner, spouse guarantee often needed.

5. SOLAR LOAN — solar panels. Ek lakh se twenty lakh. 12-36 months.
   Kis ke liye: homeowner preferred, solar vendor quote needed.

6. PERSONAL LOAN — zaati zaroorat. Pachas hazaar se paanch lakh. 6-24 months.
   Kis ke liye: salaried, minimum forty thousand monthly income.

7. KHAWATEEN LOAN — sirf khawateen ke liye. Ek lakh se twenty lakh. 12-36 months.
   Kis ke liye: women entrepreneurs, supportive guarantor accepted.

PRODUCTS — ACCOUNTS:

1. BASIC ACCOUNT — zero minimum balance, roz marra transactions.
2. SAVINGS ACCOUNT — monthly profit, paanch hazaar minimum balance.
3. CURRENT ACCOUNT — unlimited transactions, cheque book, business ke liye.
4. FIXED DEPOSIT — fixed profit, teen month se paanch saal.

═══════════════════════════════════════════════════════════════════════════════
INCOME ELIGIBILITY GUIDE (internal — caller ko exact thresholds mat batao):

- Personal Loan: minimum chalees hazaar monthly.
- Karobar / Khawateen: minimum pachas hazaar business revenue.
- Zaraat / Maweshi: agricultural land ya livestock ownership zaroori.
- Ghar Asaan: minimum saath hazaar; property ownership preferred.
- Solar: minimum chalees hazaar; homeowner.

EMPLOYMENT TYPES ACCEPTED:
- Self-employed (dukan, workshop, karobar).
- Salaried (private ya government).
- Kisan (agricultural land owner).
- Khawateen entrepreneur.

═══════════════════════════════════════════════════════════════════════════════
OBJECTION HANDLING (pehle acknowledge karo, phir reframe):

"BYAAJ ZYADA HAI":
   "Samajh aaya — actually hamari rates reducing balance pe hain, yani har installment ke baad outstanding amount kam hoti hai, toh actual cost kam nikalta hai. Exact EMI representative batayega."

"DOCUMENTS BAHUT HAIN":
   "Process actually simple hai — CNIC aur utility bill se shuru hoti hai. Hamara field officer ghar pe aakar guide karega — aap ko kuch dhundna nahi padega."

"SOCHKE BATAUNGA":
   "Zaroor sochein — koi pressure nahi. Kya ek tentative callback schedule kar dun? Koi commitment nahi, sirf information ke liye."

"DOOSRI JAGAH SE LE RAHA HOON":
   "Bilkul aap ka haq hai. Ek baar hamari team se rate compare zaroor karein before finalizing — koi fee nahi."

"CREDIT HISTORY KHARAB HAI":
   "Samajh aaya — hamara field officer case by case dekhta hai. Theek history har case mein zaroori nahi hoti. Ek baar try zaroor karein."

"GUARANTOR NAHI HAI":
   "Kuch products mein relaxed guarantor requirements hain. Officer specifically batayega."

"INCOME PROOF NAHI HAI":
   "Hamara informal income assessment bhi hota hai — officer khud verify karta hai."

"PEHLE SE LOAN HAI":
   "Top-up ya second loan bhi possible hai eligibility pe depend karta hai."

═══════════════════════════════════════════════════════════════════════════════
CROSS-SELL OPPORTUNITIES:

- Karobar loan → Current account suggest karo ("karobar ke liye useful hoga").
- Khawateen loan → Savings account suggest karo ("profit ke saath").
- Any loan → HBL Mobile app mention karo ("account manage karna asaan ho jata hai").
- Savings account → Fixed deposit bhi mention karo ("higher profit").

═══════════════════════════════════════════════════════════════════════════════
CONVERSATION FLOW:

1. OPENING: "السلام علیکم، میں زارا ہوں، ایچ بی ایل مائیکرو فنانس بینک سے۔ کیا آپ کسی لون یا اکاؤنٹ میں دلچسپی رکھتے ہیں؟"

2. ROUTE:
   - Interested in product → go to NEED qualification.
   - Existing customer → acknowledge warmly, ask if new loan or info.
   - Not interested → polite close, leave door open.

3. NEED (ek ek sawaal):
   a. "Aap ka maqsad kya hai — karobar, ghar, zameen, ya koi aur zaroorat?"
   b. Sahi product identify karo aur brief introduction do.
   c. "Approximately kitna chahiye?"

4. BUDGET (ek sawaal):
   "Aap ki monthly income ya karobar ki average earning approximately kitni hai?"
   Agar hesitant: "Sirf approximate — main sahi product suggest karne ke liye poochhti hoon."

5. AUTHORITY:
   "Aap hi yeh faisla karenge, ya ghar mein ya business mein koi aur bhi involved hoga?"
   Sole = great, proceed. Shared = callback for both.

6. TIMELINE:
   "Approximately kab tak chahiye?"
   1 hafta = HOT. 1 mahina = HOT. 2-3 mahine = WARM. 3+ mahine = COLD.

7. OBJECTIONS: jab bhi aayein, handle karo (see section above).

8. CALLBACK REGISTRATION (HOT ya WARM lead):
   a. "Aap ka poora naam?"
   b. "Aap kaunse sheher mein hain?"
   c. "Yeh number sahi hai callback ke liye?"
   d. "Kab convenient hoga — subah, dopahar, ya shaam? Kaunsa din?"
   e. "Aur kaunse product ke baare mein baat karni hai?" (confirm)

   Phir register_callback tool call karo.

   Tool ke baad: "Theek hai [Naam] ji, main ne note kar liya. Hamari team [sheher] mein [waqt] pe rabta karegi [product] ke baare mein. Sab theek hai?"

9. COLD LEAD CLOSE: polite, door open:
   "Koi baat nahi — jab bhi zaroorat ho, hum available hain. Hamari helpline {HELPLINE} pe call karein."

10. CLOSE: "بالکل، آپ کے وقت اور ہم پر اعتماد کا شکریہ۔ ہماری ٹیم جلد آپ سے رابطہ کرے گی۔ شکریہ اور اللہ حافظ!"

═══════════════════════════════════════════════════════════════════════════════
HARD RULES:
- KOI PRESSURE NAHI — sales NHI hard sell.
- Authority maloom nahi — "aap" use karo.
- Tum aurat ho (Zara) — feminine verbs.
- OTP / PIN / password kabhi mat maango.
- Exact rates / EMI bind mat karo — "approximate hai, representative confirm karega".
- Existing customer ka balance / installment phone pe mat batao — refer to app/branch.
- Hamesha respect aur empathy — caller ka time precious hai.
"""


class ZaraAgent(Agent):
    def __init__(self, system_prompt: str, caller_phone: str | None = None):
        super().__init__(instructions=system_prompt)
        self.caller_phone = caller_phone
        self.callbacks: list[dict] = []  # all leads registered this call

    async def on_enter(self) -> None:
        self.session.say(
            "السلام علیکم، میں زارا ہوں، ایچ بی ایل مائیکرو فنانس بینک سے۔ "
            "کیا آپ کسی لون یا اکاؤنٹ میں دلچسپی رکھتے ہیں؟",
            allow_interruptions=True,
        )

    @function_tool
    async def register_callback(
        self,
        caller_name: Annotated[str, "Caller's full name. If unknown, 'Not provided'."],
        caller_city: Annotated[str, "Caller's city. If unknown, 'Not provided'."],
        product_interest: Annotated[
            str,
            "One of: karobar_loan, zaraat_loan, maweshi_loan, ghar_asaan_loan, solar_loan, personal_loan, khawateen_loan, savings_account, current_account, multiple, undecided",
        ],
        loan_purpose: Annotated[str, "Why they want the loan (e.g. 'expand shop', 'build house'). 'Not mentioned' if unknown."],
        income_range: Annotated[
            str,
            "Caller's monthly income. One of: below_50k, 50k_to_1lakh, above_1lakh, not_disclosed",
        ],
        employment_type: Annotated[
            str,
            "One of: self_employed, salaried, farmer, not_disclosed",
        ],
        lead_score: Annotated[
            str,
            "Your assessment. One of: hot, warm, cold, not_assessed",
        ],
        callback_time: Annotated[
            str,
            "Day and time agreed for callback (e.g. 'Monday morning'). 'Not scheduled' if no callback agreed.",
        ],
        objections_raised: Annotated[str, "Comma-separated objections raised by caller. 'None' if no objections."] = "None",
        callback_phone: Annotated[str, "Phone number for callback if different from caller phone. 'Same' if same."] = "Same",
    ) -> str:
        """Register the sales callback in the CRM. Call this AFTER you've collected enough BANT
        information and the caller has agreed to a callback time. Returns a reference number to
        repeat back to the caller."""
        record = {
            "caller_name": caller_name,
            "caller_phone": self.caller_phone if callback_phone == "Same" else callback_phone,
            "caller_city": caller_city,
            "product_interest": product_interest,
            "loan_purpose": loan_purpose,
            "income_range": income_range,
            "employment_type": employment_type,
            "lead_score": lead_score,
            "callback_time": callback_time,
            "objections_raised": objections_raised,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        self.callbacks.append(record)

        ref = await self._create_crm_lead(record)

        logger.info(f"Lead registered: {ref} [{product_interest}/{lead_score}] {caller_name}")

        return (
            f"CALLBACK REGISTERED. Reference number: {ref}. Lead score: {lead_score}. "
            f"Now: confirm summary to the caller in Urdu — name, city, time, product. "
            f"Say the reference number clearly in Urdu, character by character. "
            f"Then close the call warmly."
        )

    async def _create_crm_lead(self, record: dict) -> str:
        """Create the sales lead ticket via the CRM API. Returns the TKT number."""
        if CRM_API_URL and CRM_TENANT_ID:
            url = (f"{CRM_API_URL.rstrip('/')}/api/v1/voice-bot/livekit/lead"
                   f"?tenantId={CRM_TENANT_ID}")
            headers = {"Content-Type": "application/json"}
            if CRM_INGEST_SECRET:
                headers["Authorization"] = f"Bearer {CRM_INGEST_SECRET}"
            payload = {
                "agent": AGENT_NAME,
                "reporterName":  record["caller_name"],
                "reporterPhone": record["caller_phone"],
                "city":          record["caller_city"],
                "productInterest": record["product_interest"],
                "loanPurpose":   record["loan_purpose"],
                "incomeRange":   record["income_range"],
                "employmentType": record["employment_type"],
                "leadScore":     record["lead_score"],
                "callbackTime":  record["callback_time"],
                "objections":    record["objections_raised"],
            }
            try:
                async with aiohttp.ClientSession() as s:
                    async with s.post(url, json=payload, headers=headers,
                                      timeout=aiohttp.ClientTimeout(total=15)) as r:
                        data = await r.json()
                        if isinstance(data, dict) and data.get("ticketNumber"):
                            return data["ticketNumber"]
                        logger.error(f"CRM lead create returned: {data}")
            except Exception as e:
                logger.error(f"CRM lead create failed: {e}")
        # Fallback reference
        import random, string
        return f"LEAD-{''.join(random.choices(string.digits, k=4))}-{random.choice(string.ascii_uppercase)}"


ANALYSIS_PROMPT = """You are a call analytics engine for HBL Microfinance Bank sales calls.
Analyze the transcript and return a JSON object with EXACTLY these fields:
- caller_name: string or "Not provided"
- caller_city: string or "Not mentioned"
- product_interest: one of karobar_loan, zaraat_loan, maweshi_loan, ghar_asaan_loan, solar_loan, personal_loan, khawateen_loan, savings_account, current_account, multiple, undecided
- loan_purpose: string or "Not mentioned"
- income_range: one of below_50k, 50k_to_1lakh, above_1lakh, not_disclosed
- employment_type: one of self_employed, salaried, farmer, not_disclosed
- lead_score: one of hot, warm, cold, not_assessed
- callback_registered: one of yes, no
- callback_time: string or "Not scheduled"
- objections_raised: string or "None"
- caller_sentiment: one of calm, interested, hesitant, not_interested
- call_summary: 2-3 sentences (caller's need, qualification result, next step)"""


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


def build_llm():
    if not OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY is required for the LLM.")
    return llm.FallbackAdapter([
        lk_openai.LLM(model=LLM_MODEL),
        lk_openai.LLM(model="gpt-4o-mini"),
    ])


def build_stt():
    if not DEEPGRAM_API_KEY:
        raise RuntimeError("DEEPGRAM_API_KEY is required for STT.")
    return deepgram.STT(
        model="nova-3",
        language="ur",
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
    if OPENAI_API_KEY:
        engines.append(lk_openai.TTS(model="tts-1", voice="shimmer"))
    if not engines:
        raise RuntimeError("No TTS configured.")
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
    zara = ZaraAgent(system_prompt=build_system_prompt(), caller_phone=caller_phone)

    await session.start(
        agent=zara,
        room=ctx.room,
        room_input_options=RoomInputOptions(close_on_disconnect=True),
    )

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

            # If callback wasn't registered via tool but transcript shows engagement,
            # still log the call so it's visible in CRM.
            if CRM_API_URL and CRM_TENANT_ID and transcript and not zara.callbacks:
                dur = int((datetime.now(timezone.utc) - call_start).total_seconds())
                url = (f"{CRM_API_URL.rstrip('/')}/api/v1/voice-bot/livekit/log-call"
                       f"?tenantId={CRM_TENANT_ID}")
                headers = {"Content-Type": "application/json"}
                if CRM_INGEST_SECRET:
                    headers["Authorization"] = f"Bearer {CRM_INGEST_SECRET}"
                payload = {
                    "agent": AGENT_NAME,
                    "callerPhone": caller_phone,
                    "transcript": transcript,
                    "summary": analysis.get("call_summary"),
                    "sentiment": analysis.get("caller_sentiment"),
                    "durationSeconds": dur,
                    "metadata": {
                        "caller_name": analysis.get("caller_name"),
                        "product_interest": analysis.get("product_interest"),
                        "lead_score": analysis.get("lead_score"),
                        "callback_registered": "no",
                    },
                }
                try:
                    async with aiohttp.ClientSession() as s:
                        async with s.post(url, json=payload, headers=headers,
                                          timeout=aiohttp.ClientTimeout(total=10)) as r:
                            logger.info(f"CRM log-call (zara, no-callback) -> HTTP {r.status}")
                except Exception as e:
                    logger.error(f"CRM log-call POST failed: {e}")

            payload = {
                "agent": AGENT_NAME,
                "caller_phone": caller_phone,
                "started_at": call_start.isoformat(),
                "ended_at": datetime.now(timezone.utc).isoformat(),
                "duration_sec": (datetime.now(timezone.utc) - call_start).total_seconds(),
                "callbacks": zara.callbacks,
                "analysis": analysis,
            }
            logger.info(f"Call summary: {json.dumps(payload, ensure_ascii=False)[:800]}")
        except Exception as e:
            logger.error(f"on_shutdown error: {e}")

    ctx.add_shutdown_callback(on_shutdown)


if __name__ == "__main__":
    agents.cli.run_app(
        WorkerOptions(
            entrypoint_fnc=entrypoint,
            prewarm_fnc=prewarm,
            agent_name=AGENT_NAME,
            port=int(os.getenv("WORKER_PORT", "8083")),
        )
    )
