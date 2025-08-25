# app_amount_assess_prompt_langchain.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
import os, json

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

# ====== LangChain prompt & chain (AFLE-focused) ======
SYSTEM_MSG = "You are a precise ABAP remediation planner that outputs strict JSON only."

USER_TEMPLATE = """
You are a senior ABAP reviewer and modernization planner.

Goal:
1) Turn 'amount_findings' into a concise, human-readable **assessment** paragraph for a reporting file (no code changes now).
   Summarize risks and why they matter for S/4HANA Amount Field Length Extension (AFLE).
   Consider guidance from SAP notes like 2628654 (S4TWL: Amount Field Length Extension), 2628040 (General info), and 2610650 (Code Adaptations).
2) Produce a **remediation LLM prompt** to be used later.
    The prompt must be concise, to the point, and contain **no more than 5 numbered bullet points**.  
   - Reference the unit metadata (program/include/unit).
   - Ask for minimal, behavior-preserving ECC-safe changes (no 7.4+ syntax) focused strictly on AFLE risks
     (e.g., type conflicts in modularization calls/Open SQL/LOOP/READ, MOVE/MOVE-CORRESPONDING issues,
      WRITE/WRITE TO layout, floating-point rounding, arithmetic error handling, hardcoded min/max, data clusters, ALV extracts).
   - Require output JSON with keys: original_code, remediated_code, changes[] (line, before, after, reason).
   - No business logic changes, no suppressions, no pseudo-comments.

Return ONLY strict JSON with keys:
{
  "assessment": "<concise assessment>",
  "llm_prompt": "<prompt to use later>"
}

Unit metadata:
- Program: {pgm_name}
- Include: {inc_name}
- Unit type: {unit_type}
- Unit name: {unit_name}

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
      - 'llm_prompt' (string; normalized even if the model returns a list)
    """
    out: List[Dict[str, Any]] = []
    for u in units:
        obj = u.model_dump()
        llm_out = llm_assess_and_prompt(u)

        # Copy assessment
        obj["assessment"] = llm_out.get("assessment", "")

        # Normalize llm_prompt: if list → join into a single string
        llm_prompt_val = llm_out.get("llm_prompt", "")
        if isinstance(llm_prompt_val, list):
            llm_prompt_val = "\n".join([p for p in llm_prompt_val if isinstance(p, str)])
        elif not isinstance(llm_prompt_val, str):
            llm_prompt_val = str(llm_prompt_val)

        obj["llm_prompt"] = llm_prompt_val

        # Remove findings as requested
        obj.pop("amount_findings", None)
        out.append(obj)
    return out

@app.get("/health")
def health():
    return {"ok": True, "model": OPENAI_MODEL}
