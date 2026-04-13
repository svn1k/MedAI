import os
import json
import re
import time
import asyncio
import threading
import requests
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from dotenv import load_dotenv

load_dotenv()
app = Flask(__name__)
CORS(app)

# ── OpenGradient ─────────────────────────────────────────────────────────────
OG_OK = False
llm_client = None
og = None
WORKING_MODEL = None
_ready = False
_init_done = False
_init_lock = threading.Lock()

MODEL_PRIORITY = [
    "CLAUDE_HAIKU_4_5",
    "CLAUDE_SONNET_4_5",
    "CLAUDE_SONNET_4_6",
    "GPT_5_MINI",
    "GEMINI_2_5_FLASH_LITE",
    "GEMINI_2_5_FLASH",
    "GEMINI_2_5_PRO",
    "GEMINI_3_FLASH",
]

# ── Event loop ─────────────────────────────────────────────────────────────
_loop = None
_loop_thread = None

def _start_loop():
    global _loop
    _loop = asyncio.new_event_loop()
    asyncio.set_event_loop(_loop)
    _loop.run_forever()

def _ensure_loop():
    global _loop
    if _loop is None:
        t = threading.Thread(target=_start_loop, daemon=True)
        t.start()
        deadline = time.time() + 10
        while _loop is None and time.time() < deadline:
            time.sleep(0.05)

def _run(coro, timeout=120):
    _ensure_loop()
    if _loop is None:
        raise RuntimeError("Event loop not ready")
    async def _with_timeout():
        return await asyncio.wait_for(coro, timeout=timeout)
    return asyncio.run_coroutine_threadsafe(_with_timeout(), _loop).result(timeout=timeout + 5)

# ── OG init ────────────────────────────────────────────────────────────────
def _init_og():
    global OG_OK, llm_client, og, _ready, _init_done, WORKING_MODEL
    with _init_lock:
        if _init_done:
            return
        _init_done = True
    try:
        import opengradient as _og
        import ssl
        import urllib3
        og = _og
        ssl._create_default_https_context = ssl._create_unverified_context
        urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
        private_key = os.environ.get("OG_PRIVATE_KEY", "")
        if not private_key:
            raise ValueError("OG_PRIVATE_KEY not set")
        print(f"OG_PRIVATE_KEY found: {private_key[:6]}...")
        llm_client = og.LLM(private_key=private_key)
        try:
            approval = llm_client.ensure_opg_approval(min_allowance=0.1)
            print(f"OPG approval: {approval}")
        except Exception as e:
            print(f"Approval warning (continuing): {e}")
        OG_OK = True
        print("OG connected — selecting model...")
        _pick_model()
    except Exception as e:
        import traceback
        print(f"OG init failed: {e}\n{traceback.format_exc()}")
    finally:
        _ready = True
        print(f"OG ready. OG_OK={OG_OK}, model={WORKING_MODEL}")

def _pick_model():
    global WORKING_MODEL
    if not OG_OK or llm_client is None:
        return
    for name in MODEL_PRIORITY:
        if not hasattr(og.TEE_LLM, name):
            continue
        model = getattr(og.TEE_LLM, name)
        try:
            print(f"  Trying {name}...")
            result = _run(llm_client.chat(
                model=model,
                messages=[{"role": "user", "content": "Say: OK"}],
                max_tokens=5,
                temperature=0.0,
            ), timeout=90)
            raw = _extract_raw(result)
            if raw and raw.strip():
                WORKING_MODEL = model
                print(f"✓ Model selected: {name}")
                return
        except Exception as e:
            print(f"  {name} failed: {e}")
    print("WARNING: No working model found")

def _ensure_og():
    if not _init_done:
        t = threading.Thread(target=_init_og, daemon=True)
        t.start()
        t.join(timeout=180)

# ── Helpers ────────────────────────────────────────────────────────────────
def _extract_raw(result):
    if not result:
        return ""
    for attr in ['chat_output', 'completion_output', 'content', 'text', 'output']:
        val = getattr(result, attr, None)
        if val:
            if isinstance(val, dict) and val.get('content'):
                return str(val['content'])
            if isinstance(val, str) and val.strip():
                return val
    for attr in dir(result):
        if attr.startswith('_'):
            continue
        try:
            val = getattr(result, attr)
            if callable(val):
                continue
            if isinstance(val, str) and val.strip() and len(val) > 2:
                return val
        except:
            pass
    return ""

def parse_json(raw):
    if not raw or not raw.strip():
        return {"error": "Empty response from LLM"}
    m = re.search(r"<JSON>(.*?)</JSON>", raw, re.DOTALL)
    if m:
        text = m.group(1).strip()
        try:
            return json.loads(text)
        except Exception as e:
            print(f"PARSE: <JSON> tag found but JSON invalid: {e}")
    m = re.search(r'\{[\s\S]*?"disease"[\s\S]*\}', raw)
    if m:
        text = m.group(0)
        try:
            return json.loads(text)
        except:
            try:
                open_braces = text.count('{') - text.count('}')
                open_brackets = text.count('[') - text.count(']')
                fixed = text + ']' * max(0, open_brackets) + '}' * max(0, open_braces)
                return json.loads(fixed)
            except:
                pass
    print("PARSE FAILED. Full raw repr:", repr(raw[:500]))
    return {"error": "Parse failed", "raw": raw[:300]}

def call_llm(messages, retries=3):
    global WORKING_MODEL
    _ensure_og()
    if not OG_OK or llm_client is None:
        return demo_stats(messages)
    if WORKING_MODEL is None:
        _pick_model()
    if WORKING_MODEL is None:
        return demo_stats(messages)

    last_error = "Unknown error"
    for attempt in range(retries):
        try:
            print(f"\nLLM attempt {attempt+1}/{retries} | model: {WORKING_MODEL}")
            result = _run(llm_client.chat(
                model=WORKING_MODEL,
                messages=messages,
                max_tokens=3000,
                temperature=0.3,
            ), timeout=120)
            raw = _extract_raw(result)
            if not raw.strip():
                last_error = "Empty response"
                time.sleep(2)
                continue
            parsed = parse_json(raw)
            if "error" in parsed:
                last_error = parsed.get("error", "Parse failed")
                time.sleep(1)
                continue
            tx = getattr(result, "transaction_hash", None) or getattr(result, "payment_hash", None)
            if tx:
                parsed["proof"] = {
                    "transaction_hash": tx,
                    "explorer_url": f"https://explorer.opengradient.ai/tx/{tx}",
                }
            return parsed
        except Exception as e:
            last_error = str(e)
            print(f"LLM exception attempt {attempt+1}: {e}")
            if "402" in str(e):
                WORKING_MODEL = None
                _pick_model()
                if WORKING_MODEL is None:
                    break
            else:
                time.sleep(2)
    return demo_stats(messages)

def demo_stats(messages):
    disease = "Unknown"
    if messages and isinstance(messages, list):
        for msg in messages:
            if msg.get("role") == "user":
                content = msg.get("content", "")
                if isinstance(content, str):
                    for line in content.split("\n"):
                        if "Disease:" in line:
                            disease = line.replace("Disease:", "").strip()[:80]
                            break
                    break
    return {
        "disease": disease or "Unknown",
        "summary": f"Could not retrieve statistics for '{disease}' - OpenGradient TEE unavailable.",
        "key_stats": [],
        "trend_years": [],
        "age_groups": [],
        "countries": [],
        "risk_factors": [],
        "insights": ["TEE connection failed. Check console for details."],
        "sources": [],
        "proof": None,
        "error_state": True,
    }

# ── Web search via DuckDuckGo ────────────────────────────────────────────────
def web_search(query, max_results=6):
    results = []
    try:
        headers = {"User-Agent": "Mozilla/5.0 (compatible; MedAI/1.0)"}
        r = requests.get(
            "https://html.duckduckgo.com/html/",
            params={"q": query, "kl": "us-en"},
            headers=headers,
            timeout=8,
        )
        snippets = re.findall(r'class="result__snippet"[^>]*>(.*?)</a>', r.text, re.DOTALL)
        titles   = re.findall(r'class="result__a"[^>]*>(.*?)</a>',       r.text, re.DOTALL)
        urls     = re.findall(r'class="result__url"[^>]*>(.*?)</span>',   r.text, re.DOTALL)
        for i in range(min(max_results, len(snippets))):
            snip  = re.sub(r'<[^>]+>', '', snippets[i]).strip()
            title = re.sub(r'<[^>]+>', '', titles[i] if i < len(titles) else '').strip()
            url   = re.sub(r'<[^>]+>', '', urls[i]   if i < len(urls)   else '').strip()
            if snip:
                results.append({"title": title, "snippet": snip[:400], "url": url})
    except Exception as e:
        print(f"Search error: {e}")
    return results

def gather_statistics(disease):
    queries = [
        f"{disease} global prevalence statistics 2023 2024 WHO",
        f"{disease} incidence rate by age group epidemiology",
        f"{disease} mortality rate by country comparison",
        f"{disease} annual cases trend 2015 2024",
        f"{disease} risk factors demographics statistics",
    ]
    all_results = []
    for q in queries:
        all_results.extend(web_search(q, max_results=4))
    seen, unique = set(), []
    for r in all_results:
        if r["url"] not in seen:
            seen.add(r["url"])
            unique.append(r)
    return unique[:18]

# ── System prompt ─────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """You are a medical statistics analyst. Reply ONLY with valid JSON inside <JSON>...</JSON> tags. No text outside.

Return this structure (fill all fields with real epidemiological data):
<JSON>
{
  "disease": "Name",
  "summary": "2-3 sentences with key numbers.",
  "key_stats": [
    {"label": "Global prevalence", "value": "422M", "year": "2022", "source": "WHO"},
    {"label": "Annual deaths", "value": "1.5M", "year": "2022", "source": "WHO"},
    {"label": "Incidence rate", "value": "130/100k", "year": "2022", "source": "WHO"},
    {"label": "Treatment success", "value": "88%", "year": "2021", "source": "WHO"}
  ],
  "trend_years": [
    {"year": "2015", "value": 350},
    {"year": "2020", "value": 422}
  ],
  "age_groups": [
    {"range": "20-39", "value": 8},
    {"range": "40-64", "value": 28},
    {"range": "65+", "value": 35}
  ],
  "countries": [
    {"country": "USA", "value": 11.3},
    {"country": "China", "value": 11.0}
  ],
  "risk_factors": [
    {"factor": "Obesity", "relative_risk": 3.2}
  ],
  "insights": ["Finding 1 with number", "Finding 2"],
  "sources": [
    {"name": "WHO Report 2022", "url": "https://who.int"}
  ]
}
</JSON>

Rules:
- key_stats: MUST include exactly 4 items - prevalence/cases, deaths/mortality, incidence rate, and one more relevant stat
- trend_years: 5-7 points, consistent unit (millions or %)
- age_groups: prevalence % per bracket
- countries: cases in millions OR prevalence % - pick one, be consistent
- risk_factors: relative_risk = multiplier vs baseline
- Use search results if provided, supplement with training knowledge for gaps
- Mark estimated values with source "estimated"
"""

# ── Routes ────────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return send_from_directory(".", "index.html")

@app.route("/health")
def health():
    return jsonify({
        "status": "ok",
        "og": OG_OK,
        "ready": _ready,
        "model": str(WORKING_MODEL) if WORKING_MODEL else None,
    })

@app.route("/probe", methods=["GET"])
def probe():
    global WORKING_MODEL
    WORKING_MODEL = None
    _pick_model()
    return jsonify({
        "working_model": str(WORKING_MODEL) if WORKING_MODEL else None,
        "og_ok": OG_OK,
    })

@app.route("/search", methods=["POST"])
def search():
    data = request.json or {}
    disease = (data.get("disease") or data.get("query") or "").strip()
    if not disease:
        return jsonify({"error": "disease is required"}), 400

    print(f"\nSearching: {disease}")

    raw_results = gather_statistics(disease)
    snippets_text = "\n\n".join(
        f"[{r['title']}]\n{r['snippet'][:200]}"
        for r in raw_results[:6]
    )

    user_content = (
        f"Disease: {disease}\n\n"
        f"Web search snippets:\n{snippets_text}\n\n"
        f"Fill the JSON with statistics for {disease}. Use search data + your knowledge."
    )
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content}
    ]

    result = call_llm(messages)
    result["search_count"] = len(raw_results)
    return jsonify(result)

# ── Main ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    # Инициализируем OpenGradient до запуска сервера
    _ensure_og()
    print(f"MedAI Statistics on :{port} | OG: {'live' if OG_OK else 'demo'}, model: {WORKING_MODEL}")
    app.run(host="0.0.0.0", port=port, debug=False)
