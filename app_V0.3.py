# app_amount_assess_prompt_langchain.py (evidence-driven, concise bullets)
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional, Tuple
import os, json
from collections import defaultdict

# ---- LLM is mandatory (no fallback) ----
os.environ["LANGCHAIN_TRACING_V2"] = "true"
if os.getenv("LANGCHAIN_API_KEY"):
    os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY environment variable is required (no fallback).")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o")

# LangChain + OpenAI
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser

app = FastAPI(title="AFLE Amount Assessment & LLM Prompt (evidence-driven, concise)")

# ====== Models (amount version, no start/end lines in input) ======
class Finding(BaseModel):
    pgm_name: Optional[str] = None
    inc_name: Optional[str] = None
    type: Optional[str] = None
    name: Optional[str] = None
    class_implementation: Optional[str] = None
    issue_type: Optional[str] = None
    severity: Optional[str] = None
    message: Optional[str] = None
    suggestion: Optional[str] = None
    snippet: Optional[str] = None
    # line/start_line/end_line/code intentionally optional/omitted here

class Unit(BaseModel):
    pgm_name: str
    inc_name: str
    type: str
    name: Optional[str] = ""
    class_implementation: Optional[str] = ""
    code: Optional[str] = ""
    amount_findings: Optional[List[Finding]] = Field(default=None)

# ====== Helpers: severity ranking & evidence packing ======
_SEV_ORDER = {"error": 0, "warning": 1, "info": 2}

def _sev_rank(sev: Optional[str]) -> int:
    return _SEV_ORDER.get((sev or "info").lower(), 2)

def _compact_snippet(s: Optional[str], max_chars: int = 160) -> str:
    s = (s or "").replace("\\n", " ").replace("\n", " ").strip()
    return (s[:max_chars] + "…") if len(s) > max_chars else s

def build_evidence(unit: Unit, max_items: int = 8) -> List[Dict[str, Any]]:
    """
    Pick a small, high-signal set of findings:
    - Prioritize error > warning > info
    - Deduplicate by (issue_type, snippet) to avoid repetition
    - Keep line (if embedded in message/snippet is absent, we leave as None)
    """
    seen = set()
    items: List[Tuple[int, Dict[str, Any]]] = []
    for f in unit.amount_findings or []:
        key = (f.issue_type or "", _compact_snippet(f.snippet or f.message or ""))
        if key in seen:
            continue
        seen.add(key)
        items.append((_sev_rank(f.severity), {
            "issue_type": f.issue_type or "Unknown",
            "severity": (f.severity or "info").lower(),
            # Try to parse a line from message if present like "line 42", else leave None
            "line_hint": _extract_line_hint(f) or None,
            "snippet": _compact_snippet(f.snippet or f.message or ""),
            "suggestion": (f.suggestion or "").strip()
        }))
    # sort by severity then keep first N
    items.sort(key=lambda x: x[0])
    return [x[1] for x in items[:max_items]]

def _extract_line_hint(f: Finding) -> Optional[int]:
    # If upstream scanner included a "line" field, use it; else try to parse from message text patterns.
    # We keep it safe & optional.
    try:
        # Pydantic will allow extra fields; try getattr first
        line = getattr(f, "line", None)
        if isinstance(line, int):
            return line
    except Exception:
        pass
    # Fallback: parse "line 123" from message if present
    import re
    m = re.search(r"\bline\s+(\d{1,6})\b", (f.message or ""), flags=re.IGNORECASE)
    if m:
        try:
            return int(m.group(1))
        except Exception:
            return None
    return None

# ====== Planning helper (agentic-lite) ======
def summarize_amount_findings(unit: Unit) -> Dict[str, Any]:
    findings = unit.amount_findings or []
    sev_counts, type_counts = {}, {}
    for f in findings:
        sev = (f.severity or "info").lower()
        sev_counts[sev] = sev_counts.get(sev, 0) + 1
        it = f.issue_type or "Unknown"
        type_counts[it] = type_counts.get(it, 0) + 1
    return {
        "program": unit.pgm_name,
        "include": unit.inc_name,
        "unit_type": unit.type,
        "unit_name": unit.name or "",
        "stats": {
            "count": len(findings),
            "severity_counts": sev_counts,
            "issue_type_counts": type_counts,
        }
    }

# ====== Evidence-driven prompt ======
SYSTEM_MSG = "You are a precise ABAP remediation planner that outputs strict JSON only."

USER_TEMPLATE = """
You are a senior ABAP reviewer and modernization planner.

Task:
1) Convert 'amount_findings' into a concise human-readable **assessment** (1 short paragraph).
2) Produce a **remediation LLM prompt** with **no more than 5 numbered bullets**, each tied to the concrete evidence below.
   - Only remediate the exact lines/snippets listed in Evidence.
   - Minimal, behavior-preserving ECC-safe changes (no 7.4+ syntax).
   - Focus strictly on AFLE issues (amount length/scale, truncation, MOVE/MOVE-CORRESPONDING, SELECT INTO targets, compares).
   - Output of the later remediation step must be JSON with keys:
     original_code, remediated_code, changes[] (line, before, after, reason).
   - Do not change business logic, do not add suppressions or pseudo-comments.

Unit metadata:
- Program: {pgm_name}
- Include: {inc_name}
- Unit type: {unit_type}
- Unit name: {unit_name}

Evidence (targeted lines/snippets; limit 8):
{evidence_block}

ABAP code (may be empty; for context only, do not scan beyond Evidence):
{code}

Planning summary (just for context; keep the output concise):
{plan_json}

Return ONLY strict JSON with keys:
{{
  "assessment": "<1 short paragraph>",
  "llm_prompt": [
    "1) <specific action for evidence item(s)>",
    "2) <specific action for evidence item(s)>",
    "3) ... (max 5 bullets)"
  ]
}}
""".strip()

def _format_evidence_block(evidence: List[Dict[str, Any]]) -> str:
    # Render compact, deterministic evidence to drive specific bullets.
    # Example row:
    # - [error] OldMoveLengthConflict @ line 42 : "lv_amt = bseg-dmbtr." -> Suggest: widen lv_amt to P LENGTH 23 DECIMALS 2
    rows = []
    for e in evidence:
        sev = e.get("severity","info")
        it = e.get("issue_type","Unknown")
        ln = e.get("line_hint")
        sn = e.get("snippet","")
        sg = e.get("suggestion","")
        rows.append(f"- [{sev}] {it}" + (f" @ line {ln}" if ln else "") + f' : "{sn}"' + (f" | Suggest: {sg}" if sg else ""))
    return "\n".join(rows) if rows else "(none)"

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", SYSTEM_MSG),
        ("user", USER_TEMPLATE),
    ]
)

llm = ChatOpenAI(model=OPENAI_MODEL, temperature=0.0)
parser = JsonOutputParser()
chain = prompt | llm | parser

def llm_assess_and_prompt(unit: Unit) -> Dict[str, Any]:
    plan_json = json.dumps(summarize_amount_findings(unit), ensure_ascii=False, indent=2)
    evidence = build_evidence(unit, max_items=8)
    evidence_block = _format_evidence_block(evidence)

    # If there is no evidence, still return a harmless, minimal prompt.
    if not evidence:
        evidence_block = "(none)"

    try:
        return chain.invoke(
            {
                "pgm_name": unit.pgm_name,
                "inc_name": unit.inc_name,
                "unit_type": unit.type,
                "unit_name": unit.name or "",
                "code": unit.code or "",
                "plan_json": plan_json,
                "evidence_block": evidence_block,
            }
        )
    except Exception as e:
        # hard fail (no fallback)
        raise HTTPException(status_code=502, detail=f"LLM call failed: {e}")

# ====== API ======
@app.post("/assess-and-prompt-amount")
async def assess_and_prompt_amount(units: List[Unit]) -> List[Dict[str, Any]]:
    """
    Input: array of units (with amount_findings[]).
    Output: same array, replacing 'amount_findings' with:
      - 'assessment' (string)
      - 'llm_prompt' (array of up to 5 strings)
    """
    out: List[Dict[str, Any]] = []
    for u in units:
        obj = u.model_dump()
        llm_out = llm_assess_and_prompt(u)
        obj["assessment"] = llm_out.get("assessment", "")
        obj["llm_prompt"]  = llm_out.get("llm_prompt", [])
        obj.pop("amount_findings", None)  # remove as requested
        out.append(obj)
    return out

@app.get("/health")
def health():
    return {"ok": True, "model": OPENAI_MODEL}
