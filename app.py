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

app = FastAPI(title="AFLE Amount Assessment & LLM Prompt (LangChain, suggestions+snippets grounded)")

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

# ====== Helpers to ground bullets in suggestions + snippets ======
def _norm_snip(s: str, max_len: int = 160) -> str:
    if not s:
        return ""
    # collapse whitespace and trim
    s = " ".join(s.split())
    return s if len(s) <= max_len else s[: max_len - 1] + "…"

def _unique_preserve_order(items: List[str]) -> List[str]:
    seen = set()
    out = []
    for it in items:
        if not it:
            continue
        key = it.strip()
        if key and key not in seen:
            seen.add(key)
            out.append(key)
    return out

def summarize_amount_findings(unit: Unit) -> Dict[str, Any]:
    findings = unit.amount_findings or []
    sev_counts, type_counts = {}, {}
    example_msgs, suggestions, snippets = [], [], []
    for f in findings:
        sev = (f.severity or "info").lower()
        sev_counts[sev] = sev_counts.get(sev, 0) + 1
        it = f.issue_type or "Unknown"
        type_counts[it] = type_counts.get(it, 0) + 1
        if f.message:
            example_msgs.append(f.message)
        if f.suggestion:
            suggestions.append(f.suggestion)
        if f.snippet:
            snippets.append(_norm_snip(f.snippet))

    # dedupe + cap for prompt hygiene
    suggestions = _unique_preserve_order(suggestions)[:8]
    snippets    = _unique_preserve_order(snippets)[:6]

    return {
        "program": unit.pgm_name,
        "include": unit.inc_name,
        "unit_type": unit.type,
        "unit_name": unit.name or "",
        "stats": {
            "count": len(findings),
            "severity_counts": sev_counts,
            "issue_type_counts": type_counts,
            "example_messages": _unique_preserve_order(example_msgs)[:5],
        },
        "grounding": {
            "suggestions": suggestions,
            "snippets": snippets,
        }
    }

# ====== LangChain prompt & chain (AFLE-focused, grounded) ======
SYSTEM_MSG = "You are a precise ABAP remediation planner that outputs strict JSON only."

USER_TEMPLATE = """
You are a senior ABAP reviewer and modernization planner.

Goal:
1) Create a concise, human-readable **assessment** paragraph summarizing AFLE risks (why they matter for S/4HANA AFLE).
   Consider SAP notes like 2628654 (S4TWL: Amount Field Length Extension), 2628040 (General info), 2610650 (Code Adaptations).
2) Produce a **remediation LLM prompt** for later automated edits.
   - The prompt must be **concise** and contain **no more than 5 numbered bullets**.
   - **Ground each bullet** using the provided `suggestion` and/or `snippet` when present.
   - If a snippet exists, reference it explicitly using backticks around a short fragment.
   - If a suggestion exists, use it as the action to request.
   - Focus strictly on AFLE risks: type/length conflicts (MOVE/assignment/MOVE-CORRESPONDING), Open SQL INTO, LOOP/READ work areas,
     WRITE/WRITE TO layout, floating rounding (F/DECFLOAT16), arithmetic error handling, hardcoded min/max, data clusters, ALV extracts.
   - Require output JSON (later) with keys: original_code, remediated_code, changes[] (line, before, after, reason).
   - **ECC-safe** (no 7.4+ features) and **no business logic changes**.

Return ONLY strict JSON:
{{
  "assessment": "<concise assessment>",
  "llm_prompt": "<prompt as a single string (not an array), with ≤5 bullets>"
}}

Unit metadata:
- Program: {pgm_name}
- Include: {inc_name}
- Unit type: {unit_type}
- Unit name: {unit_name}

ABAP code (optional; may be empty):
{code}

Planning summary:
{plan_json}

# Grounding (use these explicitly)
## Critical code snippets (trimmed, unique, up to 6)
{critical_snippets}

## Finding details (use suggestion + snippet where applicable)
{findings_table}

amount_findings (raw JSON):
{findings_json}
""".strip()

def _format_findings_table(findings: List[Finding]) -> str:
    if not findings:
        return "(none)"
    rows = []
    for f in findings:
        rows.append(
            "- issue_type: {it}; severity: {sev}; suggestion: {sug}; snippet: `{snip}`".format(
                it=f.issue_type or "Unknown",
                sev=(f.severity or "info").lower(),
                sug=(f.suggestion or "").strip(),
                snip=_norm_snip(f.snippet or "")
            )
        )
    return "\n".join(rows[:25])  # keep prompt small

def _format_critical_snippets(snips: List[str]) -> str:
    if not snips:
        return "(none)"
    return "\n".join(f"- `{s}`" for s in snips)

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
    # Keep full finding objects for table + raw JSON; also pass only the needed fields for clarity
    findings_struct = unit.amount_findings or []
    findings_min = [
        {
            "issue_type": f.issue_type,
            "severity": f.severity,
            "message": f.message,
            "suggestion": f.suggestion,
            "snippet": f.snippet,
        }
        for f in findings_struct
    ]
    findings_json = json.dumps(findings_min, ensure_ascii=False, indent=2)

    plan = summarize_amount_findings(unit)
    plan_json = json.dumps(plan, ensure_ascii=False, indent=2)

    critical_snippets = _format_critical_snippets(plan.get("grounding", {}).get("snippets", []))
    findings_table = _format_findings_table(findings_struct)

    try:
        out = chain.invoke(
            {
                "pgm_name": unit.pgm_name,
                "inc_name": unit.inc_name,
                "unit_type": unit.type,
                "unit_name": unit.name or "",
                "code": unit.code or "",
                "plan_json": plan_json,
                "critical_snippets": critical_snippets,
                "findings_table": findings_table,
                "findings_json": findings_json,
            }
        )
        # Safety: some models may still (rarely) return array for llm_prompt
        if isinstance(out, dict) and isinstance(out.get("llm_prompt"), list):
            out["llm_prompt"] = "\n".join(str(x) for x in out["llm_prompt"])
        return out
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
      - 'llm_prompt' (string; grounded bullets using suggestion+snippet)
    """
    out: List[Dict[str, Any]] = []
    for u in units:
        obj = u.model_dump()
        llm_out = llm_assess_and_prompt(u)
        obj["assessment"] = llm_out.get("assessment", "")
        obj["llm_prompt"]  = llm_out.get("llm_prompt", "")
        obj.pop("amount_findings", None)  # remove as requested
        out.append(obj)
    return out

@app.get("/health")
def health():
    return {"ok": True, "model": OPENAI_MODEL}
