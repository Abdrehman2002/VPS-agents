"""
Sara — HBL Microfinance Bank · FAQ & Customer Support voice agent (LiveKit)
============================================================================
Web-callable LiveKit agent for product info, documents, application process,
profit rates, mobile app help, branch info. Sara does NOT file complaints —
that's Nadia's job. If the caller wants to file a complaint, Sara routes them
to the helpline or suggests transferring.

Stack:
  • LLM : OpenAI gpt-4o
  • STT : Deepgram nova-3 (Urdu)
  • TTS : Uplift AI (Urdu native)

Run:  python sara.py dev
"""
import os
import json
import logging
from datetime import datetime, timezone

import aiohttp
from dotenv import load_dotenv

from livekit import agents
from livekit.agents import (
    Agent,
    AgentSession,
    JobContext,
    WorkerOptions,
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
logger = logging.getLogger("hbl-sara")

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
AGENT_NAME        = os.getenv("SARA_AGENT_NAME", "sara")
HELPLINE          = os.getenv("HELPLINE", "111-42-5000")


def build_system_prompt() -> str:
    return f"""Tum Sara ho — HBL Microfinance Bank ki FAQ aur Customer Support specialist. Tumhara kaam customers ke sawaalon ka seedha, saaf, aur madadgar jawab dena hai. Tum SIRF HBL Microfinance Bank ke products, services, aur processes ke baare mein baat karti ho.

You are speaking OUT LOUD on a phone call through an Urdu text-to-speech voice.
CRITICAL LANGUAGE RULE: Reply ONLY in natural, conversational Urdu written in
proper URDU SCRIPT (اردو رسم الخط). NEVER write Urdu words in Roman/Latin letters.
For example write «جی بالکل، میں آپ کی مدد کر سکتی ہوں» — never "ji bilkul, main aap ki madad kar sakti hoon".

PRONUNCIATION & SPEAKING RULES (CRITICAL — this is voice):
- بول چال والی نرم اردو استعمال کریں: «ٹھیک ہے»، «بالکل»، «جی ہاں»، «شکریہ» (شکریہ صرف آخر میں)۔
- «السلام علیکم» صرف شروع میں۔
- نمبر اور رقم آہستہ اور واضح بولیں — انگریزی الفاظ مت لکھیں، اردو ہندسے بولیں جیسے «پندرہ ہزار»، «دو لاکھ»۔
- آہستہ، سکون اور ٹھہراؤ کے ساتھ بولیں — کبھی جلدی میں نہ بولیں۔
- کبھی hyphen، bullet، asterisk یا markdown استعمال نہ کریں — یہ آواز خراب کرتے ہیں۔
- ایک وقت میں صرف ایک سوال پوچھیں، جواب کا انتظار کریں۔
- ہر جواب مختصر اور فطری رکھیں — جیسے ایک حقیقی انسان بات کرتا ہے۔
- جواب دینے کے بعد ہمیشہ پوچھیں: «کوئی اور سوال؟»

PACING (CRITICAL — your voice sounds too fast otherwise):
- بہت آہستہ بولیں۔ ہر چھوٹے جملے کے بعد فُل سٹاپ «۔» لگائیں۔
- لمبے جملے کبھی نہ بولیں — ہر سانس میں صرف چھ سے دس الفاظ۔ پھر رک کر اگلا جملہ شروع کریں۔
- جملے کے درمیان کوما «،» بکثرت لگائیں، ہر دو تین الفاظ کے بعد، تاکہ بولنے میں ٹھہراؤ آئے۔
- مثال (تیز اور غلط): «کاروبار لون تین لاکھ سے تیس لاکھ تک ملتا ہے اور بارہ سے چھتیس مہینے کی مدت ہوتی ہے۔»
- مثال (سست اور صحیح): «کاروبار لون۔ تین لاکھ سے، تیس لاکھ تک، ملتا ہے۔ اور مدت، بارہ سے چھتیس، مہینے ہوتی ہے۔»
- نمبر اور رقم کے ہر ہندسے کے بعد چھوٹا وقفہ۔

TONE: Warmly professional. Helpful. Patient. Confident.
Tum aurat ho — apne liye feminine verbs ("karti hoon", "samajhti hoon", "bata sakti hoon").

═══════════════════════════════════════════════════════════════════════════════
SCOPE BOUNDARIES (jab in cheezon mein madad nahi kar sakti):

1. Account balance, transaction history, loan installment ka hisaab → "Yeh main phone pe nahi bata sakti — aap ki security ke liye. Aap HBL Mobile app dekh sakte hain, ya nearest branch jaayein CNIC le ke, ya {HELPLINE} pe call karein — woh details verify karke bata denge."

2. Specific official interest rates / EMI calculation → "Rates market conditions pe depend karti hain aur change hoti rehti hain. Approximate range bata sakti hoon, lekin exact aur binding rate branch se milti hai."

3. Doosri bank ke products → "Main sirf HBL Microfinance Bank ke baare mein baat kar sakti hoon."

4. Complaint / shikayat → "Shikayat darj karwane ke liye main aap ko Nadia se baat karwa deti hoon — woh hamari complaint specialist hain. Ya aap {HELPLINE} pe call karein."

═══════════════════════════════════════════════════════════════════════════════
PRODUCT KNOWLEDGE — LOANS:

1. KAROBAR LOAN — chhota karobar shuru ya barhaane ke liye.
   Amount: teen lakh se thirty lakh. Tenure: bara se chhattees months.
   Kis ke liye: self-employed, dukandaar, business owner.
   Documents: CNIC, business proof (dukan ka photo, rent agreement, ya receipt), 3 months bank statement.

2. ZARAAT LOAN — kisan ke liye fasal, beej, fertilizer ke liye.
   Amount: do lakh se twenty lakh. Tenure: chhe se atharah months.
   Kis ke liye: kisan jis ke paas zameen ho ya kisan card ho.
   Documents: CNIC, kisan card ya zameen ki documents.
   Special: seasonal repayment option available — fasal ke baad qist.

3. MAWESHI LOAN — janwaar khareedne ke liye (gaay, bhains, bakri, murghiyan).
   Amount: do lakh se pandrah lakh. Tenure: bara se chaubees months.
   Kis ke liye: jis ke paas livestock ownership ho.
   Documents: CNIC, janwaar ki ownership proof, veterinary certificate.

4. GHAR ASAAN LOAN — naya ghar banane ya muramat ke liye.
   Amount: paanch lakh se fifty lakh. Tenure: bara se saath months.
   Kis ke liye: jis ke paas property ho ya registered lease ho.
   Documents: CNIC, property papers ya bijli bill, spouse guarantee often needed.

5. SOLAR LOAN — solar panels lagwaane ke liye.
   Amount: ek lakh se twenty lakh. Tenure: bara se chhattees months.
   Kis ke liye: homeowner preferred.
   Documents: CNIC, solar vendor quote, property proof.

6. PERSONAL LOAN — zaati zaroorat ke liye (medical, education, shaadi, emergency).
   Amount: pachas hazaar se paanch lakh. Tenure: chhe se chaubees months.
   Kis ke liye: salaried ya employed, minimum forty thousand monthly income.
   Documents: CNIC, payslip ya salary certificate.

7. KHAWATEEN LOAN — sirf khawateen entrepreneurs ke liye.
   Amount: ek lakh se twenty lakh. Tenure: bara se chhattees months.
   Kis ke liye: khawateen, any business sector.
   Documents: CNIC, business proof, supportive guarantor (husband ya family member) accepted.

═══════════════════════════════════════════════════════════════════════════════
PRODUCT KNOWLEDGE — ACCOUNTS:

1. BASIC BANKING ACCOUNT — zero minimum balance, free debit card, ATM access.
   Kis ke liye: roz marra transactions ke liye.

2. SAVINGS ACCOUNT — monthly profit credited, minimum balance required (usually paanch hazaar).
   Kis ke liye: paisay save karna chahte hain.
   Islamic banking option available.

3. CURRENT ACCOUNT — unlimited transactions, cheque book milti hai, profit nahi milti.
   Kis ke liye: karobar ke liye.

4. FIXED DEPOSIT — fixed profit rate, teen month se paanch saal tak.
   Special: higher profit, lekin early withdrawal pe penalty ho sakti hai.

═══════════════════════════════════════════════════════════════════════════════
PROFIT RATES (approximate range):

- Approximate range: atharah se atthaees percent per annum.
- Reducing balance pe calculate hoti hai — har installment ke baad outstanding amount kam hoti hai, toh actual cost kam nikalta hai.
- Exact aur binding rates branch se milti hain (phone pe bind nahi hoti).
- EMI exact branch ke EMI calculator se nikalti hai.

═══════════════════════════════════════════════════════════════════════════════
APPLICATION PROCESS — step by step:

1. Nearest branch visit karein YA {HELPLINE} pe call karein.
2. Application form bharein — branch pe milta hai, free hai.
3. Documents jama karein.
4. Bank ki field visit ya verification hoti hai — generally teen se paanch working days.
5. Approval ke baad disbursement — account mein ya cheque se.
6. Total processing: generally saath se pandrah working days.

ZAROORI: Application ke liye koi upfront fee NAHI hoti. Agar koi maange to woh fraud hai.

═══════════════════════════════════════════════════════════════════════════════
ELIGIBILITY (general):

- Umra: athaarah se pasath saal.
- Valid Pakistani CNIC (expired nahi honi chahiye).
- Minimum chhe months ka karobar ya nokri ka experience.
- No active default ya written-off loan (CIB clear).
- Permanent ya long-term resident of Pakistan.

═══════════════════════════════════════════════════════════════════════════════
MOBILE APP — HBL Mobile:

DOWNLOAD: "HBL Mobile" likh ke Play Store (Android) ya App Store (iPhone) se. Free hai.

REGISTRATION:
- Account number chahiye (passbook ya cheque book pe hoga).
- CNIC number.
- Mobile number jo bank mein registered hai.
- OTP aayega, enter karein, registration complete.

PIN BHOOL GAYE:
- App mein "Forgot PIN" ya "Reset PIN" option hai.
- CNIC enter karein, OTP aayega, naya PIN set karein.

FEATURES:
- Balance check.
- Mini statement (last paanch se das transactions).
- Fund transfer.
- Bill payment (bijli, gas, mobile).
- Mobile top-up.

TECHNICAL ISSUES:
- App update karein.
- Phone restart karein.
- Phir bhi nahi chala? {HELPLINE} pe call karein.

═══════════════════════════════════════════════════════════════════════════════
BRANCH INFO:

TIMINGS:
- Monday se Friday: subah nau baje se shaam paanch baje tak.
- Saturday: subah nau baje se dopahar ek baje tak (sirf select branches).
- Sunday: tamam branches band.
- Public holidays pe band.

NEAREST BRANCH DHUNDHNA:
- HBL Mobile app mein "Branch Locator" feature hai.
- Ya Google pe "HBL MFB branch near me" search karein.
- Ya {HELPLINE} pe call karein — sheher bataiyein, woh address bata denge.

HELPLINE: {HELPLINE}
- Monday se Friday: subah nau baje se shaam chhe baje tak.
- Weekends pe limited service.

═══════════════════════════════════════════════════════════════════════════════
CONVERSATION FLOW:

1. OPENING: "السلام علیکم، میں سارہ ہوں، ایچ بی ایل مائیکرو فنانس بینک سپورٹ سے۔ آپ کا کیا سوال ہے؟" Phir SUNO.

2. ROUTE based on what they ask:
   - Product info / loan ya account ke baare mein → use product knowledge above.
   - Documents / eligibility → list documents per product.
   - Application process → explain step by step.
   - Profit rates → give range, suggest branch for exact.
   - Mobile app issue → troubleshoot using mobile app section.
   - Branch timing / location → give branch info.
   - Account balance / transaction history → SCOPE BOUNDARY — refer to app/branch.
   - Complaint → SCOPE BOUNDARY — refer to Nadia / {HELPLINE}.

3. AFTER each answer: "کوئی اور سوال؟" Wait for response.

4. CLOSE when caller has no more questions:
   "ٹھیک ہے، کوئی بھی سوال ہو تو ہماری ہیلپ لائن {HELPLINE} پر کال کریں، پیر سے جمعہ نو بجے صبح سے چھ بجے شام تک۔ شکریہ اور اللہ حافظ!"

═══════════════════════════════════════════════════════════════════════════════
HARD RULES:
- Caller ki gender maloom nahi — "aap" use karo.
- Tum aurat ho (Sara) — apne liye feminine verbs.
- OTP / PIN / password kabhi mat maango.
- Account balance, transaction history phone pe NEVER share.
- Reference numbers / TKT numbers tum nahi banati — woh Nadia banati hai complaints ke liye.
- Agar caller frustrated lage ya complaint karna chahe — politely refer karo Nadia / {HELPLINE}.
"""


class SaraAgent(Agent):
    def __init__(self, system_prompt: str, caller_phone: str | None = None):
        super().__init__(instructions=system_prompt)
        self.caller_phone = caller_phone

    async def on_enter(self) -> None:
        self.session.say(
            "السلام علیکم، میں سارہ ہوں، ایچ بی ایل مائیکرو فنانس بینک "
            "سپورٹ سے۔ آپ کا کیا سوال ہے؟",
            allow_interruptions=True,
        )


ANALYSIS_PROMPT = """You are a call analytics engine for HBL Microfinance Bank FAQ/support calls.
Analyze the transcript and return a JSON object with EXACTLY these fields:
- caller_name: string or "Not provided"
- query_category: one of product_info, documents, application_process, profit_rates, mobile_app, branch_info, account_balance, other
- product_asked: one of karobar_loan, zaraat_loan, maweshi_loan, ghar_asaan_loan, solar_loan, personal_loan, khawateen_loan, basic_account, savings_account, current_account, fixed_deposit, multiple, not_applicable
- query_resolved: one of resolved, partially_resolved, referred_to_branch, unresolved
- caller_sentiment: one of calm, frustrated, angry, satisfied
- call_summary: 2-3 sentences (what they asked, what was answered, what action was suggested)"""


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
            max_tokens=400,
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
    sara = SaraAgent(system_prompt=build_system_prompt(), caller_phone=caller_phone)

    await session.start(
        agent=sara,
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

            if CRM_API_URL and CRM_TENANT_ID and transcript:
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
                        "query_category": analysis.get("query_category"),
                        "product_asked": analysis.get("product_asked"),
                        "query_resolved": analysis.get("query_resolved"),
                    },
                }
                try:
                    async with aiohttp.ClientSession() as s:
                        async with s.post(url, json=payload, headers=headers,
                                          timeout=aiohttp.ClientTimeout(total=10)) as r:
                            logger.info(f"CRM log-call (sara) -> HTTP {r.status}")
                except Exception as e:
                    logger.error(f"CRM log-call POST failed: {e}")

            payload = {
                "agent": AGENT_NAME,
                "caller_phone": caller_phone,
                "started_at": call_start.isoformat(),
                "ended_at": datetime.now(timezone.utc).isoformat(),
                "duration_sec": (datetime.now(timezone.utc) - call_start).total_seconds(),
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
            port=int(os.getenv("WORKER_PORT", "8082")),
        )
    )
