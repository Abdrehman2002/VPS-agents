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

from typing import Annotated

from livekit import agents
from livekit.agents import (
    Agent,
    AgentSession,
    JobContext,
    WorkerOptions,
    RoomInputOptions,
    AutoSubscribe,
    function_tool,
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

═══════════════════════════════════════════════════════════════════════════════
🔴 MOST IMPORTANT RULE — READ FIRST, OBEY ALWAYS:

You have a tool called `lookup_customer`. It queries the LIVE CRM and returns
real ticket/complaint data. IT IS NOT OPTIONAL.

MANDATORY: Call `lookup_customer` IMMEDIATELY whenever the caller mentions:
  • "pehle" / "پہلے" ("before") / "call ki thi"
  • "status" / "update" / "kya hua meri complaint ka"
  • "reference number" / "TKT-XXXXX" / "CNIC" / any 13-digit number
  • "WhatsApp par shikayat" / any prior interaction

Standard 3-step identity flow (STT-safe):

  STEP 1: Ask NAME first — "Aap ka poora naam kya hai?" → repeat back to confirm.

  STEP 2: Ask ONLY LAST 4 DIGITS of CNIC — "Aap ke CNIC ke sirf aakhri 4 digits?"
     ⚠️ MANDATORY digit-by-digit readback: "Zero, Two, Four, Four — sahi?" wait
     for the caller's confirmation before moving on.

  STEP 2b: Say a NATURAL "checking" phrase (rotate randomly):
       "Aik minute dein, aap ki details pull karti hoon."
       "Bas ek lamha, aap ka record dhoondh rahi hoon."
       "Zara ruk jayein, main dekhti hoon."

     🚫 FORBIDDEN — NEVER say "tool", "function", "API", "database", "system",
     "query", "lookup" or any English tech word. Talk like a HUMAN bank rep.

  STEP 3: Call the tool:
     lookup_customer(caller_name="Ahmed Raza", cnic_last4="0244")

  STEP 4: Handle response:
    MATCH   → share status/subject naturally (no extra verification on this path).
    AMBIGUOUS → ask for full CNIC in 3 chunks (5, 7, 1) with readback.
    NO MATCH → naturally offer to route to Nadia for new complaint.
              NEVER say "record nahi mili".
    LOOKUP FAILED → route to helpline politely.
═══════════════════════════════════════════════════════════════════════════════

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
   - EXISTING ticket / prior call / WhatsApp complaint / status query → use lookup_customer flow below.

═══════════════════════════════════════════════════════════════════════════════
EXISTING TICKET STATUS CHECK (with lookup_customer tool):

  ⚠️ When caller mentions any prior complaint / call / WhatsApp / ticket / status query,
  call lookup_customer BEFORE anything else.

  STEP 1 — Ask NAME first, then LAST 4 CNIC (STT-friendly):
    "Zaroor. Aap ka poora naam kya hai?" → repeat back to confirm.
    "Aap ke CNIC ke aakhri 4 digits batayein?" → MANDATORY digit-by-digit readback:
    "Zero, Two, Four, Four — sahi?" wait for confirmation.

  STEP 2 — Say a NATURAL "checking" line, THEN call the tool:
    Pick ONE (rotate randomly):
      "Aik minute dein, aap ki details pull karti hoon."
      "Bas ek lamha, aap ka record dhoondh rahi hoon."

    🚫 NEVER say "tool", "system", "database", "API", "query", "lookup" — those
    are implementation details. Talk like a HUMAN bank rep.

    Then call:
      lookup_customer(caller_name="Ahmed Raza", cnic_last4="0244")

    FALLBACKS (only if caller volunteers):
      lookup_customer(cnic="42101-1234567-8")     — full CNIC
      lookup_customer(ticket_number="TKT-01059")  — ticket number

  STEP 3 — Handle tool response:

    A) "MATCH FOUND":
       a. Verify FIRST: "Confirm karne ke liye, apne CNIC ke aakhri 4 digits bata dein?"
       b. Digits match → share ticket status/subject/assignee (never full CNIC or name).
       c. Digits do NOT match → "Maazrat, main aap ki record verify nahi kar payi.
          Nadia ko transfer karti hoon ya {HELPLINE} pe call karein."

    B) "NO MATCH":
       → DO NOT leak the miss. Say naturally: "Hamare record mein aap ki koi
         pending ticket abhi register nahi hai. Agar aap nayi shikayat register
         karna chahte hain, Nadia se baat karwa deti hoon. Warna koi aur sawaal?"

    C) "LOOKUP FAILED" / "LOOKUP UNAVAILABLE":
       → Say: "Maazrat, thori si ruksani hai — abhi aap ki details nahi mil pa rahi.
         Aap {HELPLINE} pe call kar ke direct status check kar sakte hain, ya
         thodi der baad wapas try karein."

  ⚠️ SECURITY: Full CNIC / poora naam KABHI out loud mat kaho. Sirf first name +
   last-4 verify. Failure ke baare mein hint mat do.

═══════════════════════════════════════════════════════════════════════════════

3. AFTER each answer: "کوئی اور سوال؟" Wait for response.

4. CLOSE when caller has no more questions:
   "ٹھیک ہے، کوئی بھی سوال ہو تو ہماری ہیلپ لائن {HELPLINE} پر کال کریں، پیر سے جمعہ نو بجے صبح سے چھ بجے شام تک۔ شکریہ اور اللہ حافظ!"

═══════════════════════════════════════════════════════════════════════════════
HARD RULES:
- 🚫 NEVER speak these English tech words during the call — they break the human
  illusion and confuse Urdu-speaking callers:
     "tool", "function", "API", "database", "system", "query", "lookup",
     "record kholti hoon", "endpoint", "backend".
  Use natural Urdu banking phrases instead:
     "aap ki details dekhti hoon" · "aap ka record dhoondh rahi hoon"
     "aap ki information pull karti hoon" · "aap ke liye check karti hoon"
- Caller ki gender maloom nahi — "aap" use karo.
- Tum aurat ho (Sara) — apne liye feminine verbs.
- OTP / PIN / password kabhi mat maango.
- Account balance, transaction history phone pe NEVER share.
- Reference numbers / TKT numbers tum nahi banati — woh Nadia banati hai complaints ke liye.
- LEKIN existing reference / CNIC lookup kar sakti ho — lookup_customer tool use karo.
- Agar caller nayi complaint file karna chahe → politely refer karo Nadia / {HELPLINE}.
- Agar caller sirf status check karna chahe → lookup_customer + last-4-verify flow use karo.
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

    @function_tool
    async def lookup_customer(
        self,
        caller_name: Annotated[str, "Caller's full name as spoken (e.g. 'Ahmed Raza'). Preferred with cnic_last4."] = "",
        cnic_last4: Annotated[str, "Last 4 digits of the caller's CNIC (e.g. '0024'). Preferred — STT handles 4 digits reliably."] = "",
        cnic: Annotated[str, "OPTIONAL: full CNIC 42101-XXXXXXX-X. Only if caller reads all 13 digits AND STT captured them."] = "",
        ticket_number: Annotated[str, "OPTIONAL: existing ticket reference like TKT-00042 if caller provides one."] = "",
    ) -> str:
        """Look up caller identity + open ticket history in the CRM.

        PREFERRED shape (STT-safe, use by default):
            lookup_customer(caller_name="Ahmed Raza", cnic_last4="0024")

        Ask the caller for their name and just the last 4 digits of their CNIC.
        Do NOT ask for the full 13-digit CNIC by default.

        Fallbacks (only if caller volunteers them):
            lookup_customer(cnic="42101-1234567-8")
            lookup_customer(ticket_number="TKT-01059")

        Response semantics:
          MATCH — on name+last4 the tool sets verificationRequired='none'; the
            two signals ARE the verification. Share status/subject.
          AMBIGUOUS — multiple contacts match; ask for full CNIC in chunks.
          NO MATCH — say naturally: "Hamare record mein aap ki koi pending
            ticket abhi register nahi hai. Kya main Nadia ko transfer kar
            doon nayi shikayat register karne ke liye?"
          LOOKUP FAILED / UNAVAILABLE — route to helpline politely.

        NEVER speak the full CNIC or full name out loud.
        """
        logger.warning(f"[TOOL FIRED] lookup_customer(name={caller_name!r}, last4={cnic_last4!r}, cnic={cnic!r}, ticket={ticket_number!r})")
        if not CRM_API_URL or not CRM_TENANT_ID:
            return "LOOKUP UNAVAILABLE: CRM not configured. Route caller to helpline."

        name_trim = (caller_name or "").strip()
        last4_digits = "".join(ch for ch in (cnic_last4 or "") if ch.isdigit())[-4:]
        cnic_digits = "".join(ch for ch in (cnic or "") if ch.isdigit())

        have_name_last4 = bool(name_trim) and len(last4_digits) == 4
        have_full_cnic  = len(cnic_digits) == 13
        have_ticket     = bool((ticket_number or "").strip())

        if not (have_name_last4 or have_full_cnic or have_ticket):
            if cnic and len(cnic_digits) != 13:
                logger.warning(f"[TOOL] CNIC parse failed: {cnic!r} → {cnic_digits!r}")
                return (
                    "STT HICCUP — the CNIC came through unclear. Do NOT tell the "
                    "caller their input is invalid. Instead ask for identity the "
                    "EASY way: FIRST their full name, THEN just the LAST 4 DIGITS "
                    "of their CNIC. Then call lookup_customer(caller_name=..., "
                    "cnic_last4=...)."
                )
            return (
                "LOOKUP NEEDS INPUT — ask the caller for their FULL NAME first, "
                "THEN the LAST 4 DIGITS of their CNIC. Call lookup_customer with "
                "caller_name and cnic_last4."
            )

        url = (f"{CRM_API_URL.rstrip('/')}/api/v1/voice-bot/livekit/lookup"
               f"?tenantId={CRM_TENANT_ID}")
        if have_full_cnic:
            url += f"&cnic={cnic_digits[:5]}-{cnic_digits[5:12]}-{cnic_digits[12]}"
        if have_ticket:
            url += f"&ticket={ticket_number.strip()}"
        if have_name_last4:
            from urllib.parse import quote_plus
            url += f"&name={quote_plus(name_trim)}&last4={last4_digits}"

        headers = {}
        if CRM_INGEST_SECRET:
            headers["Authorization"] = f"Bearer {CRM_INGEST_SECRET}"

        try:
            async with aiohttp.ClientSession() as s:
                async with s.get(url, headers=headers,
                                 timeout=aiohttp.ClientTimeout(total=2)) as r:
                    data = await r.json()
        except Exception as e:
            logger.error(f"lookup_customer failed: {e}")
            return "LOOKUP FAILED: continue as if new caller. Do not mention the failure to the caller."

        if not data.get("found"):
            return "NO MATCH: this caller has no prior record. Proceed naturally — do NOT mention that the lookup returned nothing."

        matched_first = data.get("matchedFirstName") or ""
        summary = [
            f"MATCH FOUND (do NOT read ticket details out yet — first do the identity readback).",
            f"Matched first name: {matched_first!r}.",
            f"Contact display name: {data.get('displayName')}.",
            f"CNIC masked: {data.get('cnicMasked')}.",
            f"Total tickets: {data.get('totalTicketCount')}. Open: {data.get('openTicketCount')}.",
        ]
        if data.get("hasCriticalOpen"):
            summary.append("HAS URGENT OPEN TICKET — be extra empathetic.")
        latest = data.get("latestTicket")
        if latest:
            summary.append(
                f"Latest ticket: {latest.get('number')}, subject '{latest.get('subject')}', "
                f"status {latest.get('status')}, priority {latest.get('priority')}, "
                f"created {latest.get('daysAgo')} days ago"
            )
            if latest.get("assigneeFirstName"):
                summary.append(f"assigned to {latest.get('assigneeFirstName')}")
            if latest.get("slaHoursLeft") is not None:
                summary.append(f"SLA {latest.get('slaHoursLeft')} hours remaining")
        summary.append(
            f"MANDATORY NEXT STEP — first-name readback verification (guards against "
            f"last-4 CNIC collisions). Ask: 'Aap ka pehla naam {matched_first} hai — "
            f"sahi hai?' Wait for 'ji/haan/sahi'. If confirmed → NOW share ticket "
            f"status/subject/assignee naturally. If NO or hesitation → treat as NO "
            f"MATCH. Do NOT reveal any ticket details. Say 'Maazrat, aap ki record "
            f"verify nahi kar payi' and route caller to Nadia or helpline."
        )
        return " ".join(summary)


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
