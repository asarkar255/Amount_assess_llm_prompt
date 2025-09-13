# app_amount_assess_prompt_langchain.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
import os, json, textwrap

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

app = FastAPI(title="AFLE Amount Assessment & LLM Prompt (LangChain, no fallback)")

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

# ====== Agentic planning helper (lite) ======
def summarize_amount_findings(unit: Unit) -> Dict[str, Any]:
    findings = unit.amount_findings or []
    sev_counts, type_counts = {}, {}
    examples = []
    for f in findings:
        sev = (f.severity or "info").lower()
        sev_counts[sev] = sev_counts.get(sev, 0) + 1
        it = f.issue_type or "Unknown"
        type_counts[it] = type_counts.get(it, 0) + 1
        if f.message:
            examples.append(f.message)
    return {
        "program": unit.pgm_name,
        "include": unit.inc_name,
        "unit_type": unit.type,
        "unit_name": unit.name or "",
        "stats": {
            "count": len(findings),
            "severity_counts": sev_counts,
            "issue_type_counts": type_counts,
            "example_messages": examples[:5],
        }
    }

# ====== NEW: collect up to 6 unique, trimmed offending snippets ======
def collect_critical_snippets(unit: Unit, max_snips: int = 6, max_chars: int = 240) -> List[str]:
    seen = set()
    out: List[str] = []
    for f in (unit.amount_findings or []):
        s = (f.snippet or "").strip()
        if not s:
            continue
        # normalize whitespace/newlines for stable dedupe
        norm = " ".join(s.split())
        if not norm:
            continue
        if norm in seen:
            continue
        seen.add(norm)
        # trim long snippets but keep verbatim feel
        if len(s) > max_chars:
            s = s[:max_chars].rstrip() + " …"
        out.append(s)
        if len(out) >= max_snips:
            break
    return out

def render_snippets_block(snips: List[str]) -> str:
    if not snips:
        return "(none)"
    # Render as a single ABAP code fence with numbered separators for high precision
    lines = []
    for idx, s in enumerate(snips, 1):
        # keep snippet exactly as provided to maximize verbatim anchoring
        lines.append(f"*[{idx}]* {s}")
    return "\n".join(lines)

# ====== LangChain prompt & chain (AFLE-focused) ======
SYSTEM_MSG = "You are a precise ABAP remediation planner that outputs strict JSON only."

USER_TEMPLATE = """
You are a senior ABAP reviewer and modernization planner.

Goal:
1) Turn 'amount_findings' into a concise, human-readable **assessment** paragraph for a reporting file (no code changes now).
   Summarize risks and why they matter for S/4HANA Amount Field Length Extension (AFLE).
   Consider guidance from SAP notes like 2628654 (S4TWL: Amount Field Length Extension), 2628040 (General info), and 2610650 (Code Adaptations).
2) Produce a **remediation LLM prompt** to be used later. It must be concise and contain **no more than 5 numbered bullet points**:
   - Reference the unit metadata (program/include/unit).
   - Ask for minimal, S4HANA Compatible ( 7.4+ syntax) focused strictly on AFLE risks
     (e.g., type conflicts in modularization calls/Open SQL/LOOP/READ, MOVE/MOVE-CORRESPONDING issues,
      WRITE/WRITE TO layout, floating-point rounding, arithmetic error handling, hardcoded min/max, data clusters, ALV extracts).
   - No business logic changes, no suppressions, no pseudo-comments.

Return ONLY strict JSON with keys:
{{
  "assessment": "<concise assessment>",
  "llm_prompt": "<prompt to use later as a single string>"
}}

Unit metadata:
- Program: {pgm_name}
- Include: {inc_name}
- Unit type: {unit_type}
- Unit name: {unit_name}

Critical code snippets (verbatim, use these to tailor the prompt precisely):
{snippets_block}

ABAP code (optional; may be empty):
{code}

Planning summary (agentic):
{plan_json}

amount_findings (JSON):
{findings_json}
""".strip()

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", SYSTEM_MSG),
        ("user", USER_TEMPLATE),
    ]
)

llm = ChatOpenAI(model=OPENAI_MODEL, temperature=0.0)
parser = JsonOutputParser()
chain = prompt | llm | parser

def llm_assess_and_prompt(unit: Unit) -> Dict[str, str]:
    findings_json = json.dumps([f.model_dump() for f in (unit.amount_findings or [])], ensure_ascii=False, indent=2)
    plan = summarize_amount_findings(unit)
    plan_json = json.dumps(plan, ensure_ascii=False, indent=2)

    # NEW: inject up to 6 unique trimmed snippets
    snippets = collect_critical_snippets(unit)
    snippets_block = render_snippets_block(snippets)

    try:
        return chain.invoke(
            {
                "pgm_name": unit.pgm_name,
                "inc_name": unit.inc_name,
                "unit_type": unit.type,
                "unit_name": unit.name or "",
                "code": unit.code or "",
                "plan_json": plan_json,
                "findings_json": findings_json,
                "snippets_block": snippets_block,
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
      - 'llm_prompt' (single string, <=5 numbered bullets)
    """
    out: List[Dict[str, Any]] = []
    for u in units:
        obj = u.model_dump()
        llm_out = llm_assess_and_prompt(u)
        obj["assessment"] = llm_out.get("assessment", "")
        # Ensure llm_prompt is a STRING (not array); if model returns list, join lines safely
        p = llm_out.get("llm_prompt", "")
        if isinstance(p, list):
            p = "\n".join(str(x) for x in p if x is not None)
        obj["llm_prompt"] = p
        obj.pop("amount_findings", None)  # remove as requested
        out.append(obj)
    return out

@app.get("/health")
def health():
    return {"ok": True, "model": OPENAI_MODEL}
