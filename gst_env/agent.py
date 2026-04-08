from __future__ import annotations

import json
import os
import re
import sys
import time
from collections import defaultdict
from decimal import Decimal
from typing import Any, Dict, List, Optional, Set, Tuple

try:
    from dotenv import load_dotenv
    load_dotenv()
    print("[CONFIG] .env file loaded successfully")
except ImportError:
    print("[CONFIG] python-dotenv not installed, using system env vars")

import requests

try:
    from groq import Groq
except ImportError:
    print("[FATAL] groq package not installed.  Run: pip install groq")
    sys.exit(1)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
GROQ_API_KEY: str = os.environ.get("GROQ_API_KEY", "")
BASE_URL: str = os.environ.get("BASE_URL", "http://localhost:7860")
MODEL: str = os.environ.get("GROQ_MODEL", "llama-3.3-70b-versatile")
MAX_RETRIES: int = 3
RETRY_DELAY: float = 2.0
BATCH_SIZE: int = 10          # invoices per LLM call  (~4-5K tokens, safe under 12K TPM)
BATCH_SLEEP: float = 65.0     # seconds to wait between batches for TPM reset

# ---------------------------------------------------------------------------
# Grader weight hints per task
# ---------------------------------------------------------------------------
WEIGHT_HINTS: Dict[str, str] = {
    "task1_easy":   "All statuses equally weighted.",
    "task2_medium": "MISMATCH and MISSING_IN_2B errors cost 1.5x more than MATCHED/EXTRA.",
    "task3_hard":   "MISMATCH and MISSING_IN_2B errors cost 2.3x more. Be precise on mismatch_fields.",
}

# ---------------------------------------------------------------------------
# System Prompt
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = """You are an Indian GST reconciliation engine. Output ONLY a single JSON object. No thinking, no explanation, no markdown.

## YOUR TASK
Match each purchase invoice against GSTR-2B entries and assign exactly one status.

## MATCHING ALGORITHM (apply in this exact order)
1. Count how many GSTR-2B entries share this invoice's invoice_number.
   - 0 matches  -> MISSING_IN_2B
   - 2+ matches -> EXTRA_IN_2B
   - 1 match    -> proceed to field comparison:
       Compare all fields (tolerance +/-1 rupee on amounts):
         * supplier_gstin  == vendor_gstin
         * invoice_date    == invoice_date
         * taxable_value   within +/-1 of invoice taxable_value
         * itc_available   == True
       All match -> MATCHED
       Any mismatch OR itc_available==False -> MISMATCH
         (list EVERY differing field in mismatch_fields)

## ITC RULE
claimable_itc = SUM(cgst+sgst+igst) for MATCHED invoices ONLY.

## GRADER WEIGHTS
MISMATCH and MISSING_IN_2B errors cost 2x more than MATCHED and EXTRA_IN_2B errors.
Do NOT guess MATCHED when uncertain — prefer MISMATCH.

## OUTPUT SCHEMA
{
  "reconciliation_result": [
    {
      "invoice_id": "<string>",
      "status": "MATCHED" | "MISMATCH" | "MISSING_IN_2B" | "EXTRA_IN_2B",
      "correction_note": "<string or null>",
      "mismatch_fields": ["taxable_value", "invoice_date", ...]
    }
  ],
  "claimable_itc": <number>,
  "confidence": <0.0 to 1.0>
}

## STRICT RULES
- Entry for EVERY invoice — no omissions.
- mismatch_fields must use exact field names: taxable_value, invoice_date, supplier_gstin, cgst, sgst, igst, itc_available
- Return ONLY the JSON object.
- No ```json fences. No text before or after. No <think> tags in output."""


# ---------------------------------------------------------------------------
# HTTP Helpers
# ---------------------------------------------------------------------------

def _get(endpoint: str) -> Dict[str, Any]:
    resp = requests.get("{}{}".format(BASE_URL, endpoint), timeout=30)
    resp.raise_for_status()
    return resp.json()


def _post(endpoint: str, payload: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    resp = requests.post(
        "{}{}".format(BASE_URL, endpoint),
        json=payload or {},
        timeout=120,
    )
    if resp.status_code >= 400:
        raise RuntimeError(
            "POST {} failed {}: {}".format(endpoint, resp.status_code, resp.text[:500])
        )
    return resp.json()


# ---------------------------------------------------------------------------
# Pre-processing helpers
# ---------------------------------------------------------------------------

def _compute_mismatch_fields(
    inv: Dict[str, Any],
    gstr_entry: Dict[str, Any],
) -> List[str]:
    """Deterministically diff invoice vs GSTR-2B entry."""
    fields: List[str] = []
    if inv.get("vendor_gstin", "").upper() != gstr_entry.get("supplier_gstin", "").upper():
        fields.append("supplier_gstin")
    if str(inv.get("invoice_date")) != str(gstr_entry.get("invoice_date")):
        fields.append("invoice_date")
    for field in ["taxable_value", "cgst", "sgst", "igst"]:
        try:
            diff = abs(
                Decimal(str(inv.get(field, 0))) - Decimal(str(gstr_entry.get(field, 0)))
            )
            if diff > Decimal("1"):
                fields.append(field)
        except Exception:
            fields.append(field)
    if not gstr_entry.get("itc_available", True):
        fields.append("itc_available")
    return fields


def _preprocess_invoices(
    obs: Dict[str, Any],
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Deterministically resolve clear-cut cases.
    Returns (resolved_entries, reduced_obs_with_only_ambiguous_invoices).
    """
    invoices = obs.get("invoices", [])
    gstr_entries = obs.get("gstr2b_entries", [])

    gstr_index: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for e in gstr_entries:
        gstr_index[e["invoice_number"]].append(e)

    resolved: List[Dict[str, Any]] = []
    ambiguous_invoices: List[Dict[str, Any]] = []

    for inv in invoices:
        inv_no = inv["invoice_number"]
        matches = gstr_index.get(inv_no, [])

        if len(matches) == 0:
            resolved.append({
                "invoice_id": inv["invoice_id"],
                "status": "MISSING_IN_2B",
                "correction_note": "Not found in GSTR-2B",
                "mismatch_fields": [],
            })
        elif len(matches) > 1:
            resolved.append({
                "invoice_id": inv["invoice_id"],
                "status": "EXTRA_IN_2B",
                "correction_note": "Appears {} times in GSTR-2B".format(len(matches)),
                "mismatch_fields": [],
            })
        else:
            diff_fields = _compute_mismatch_fields(inv, matches[0])
            if not diff_fields:
                resolved.append({
                    "invoice_id": inv["invoice_id"],
                    "status": "MATCHED",
                    "correction_note": None,
                    "mismatch_fields": [],
                })
            else:
                ambiguous_invoices.append(inv)

    ambiguous_inv_nos = {inv["invoice_number"] for inv in ambiguous_invoices}
    reduced_gstr = [e for e in gstr_entries if e["invoice_number"] in ambiguous_inv_nos]

    ambiguous_obs = dict(obs)
    ambiguous_obs["invoices"] = ambiguous_invoices
    ambiguous_obs["gstr2b_entries"] = reduced_gstr

    return resolved, ambiguous_obs


def _recompute_itc(
    action: Dict[str, Any],
    obs: Dict[str, Any],
) -> float:
    """Recompute ITC deterministically from MATCHED entries using original invoice values."""
    invoice_map = {inv["invoice_id"]: inv for inv in obs.get("invoices", [])}
    total = Decimal("0")
    for entry in action.get("reconciliation_result", []):
        if entry.get("status") == "MATCHED":
            inv = invoice_map.get(entry["invoice_id"])
            if inv:
                total += (
                    Decimal(str(inv.get("cgst", 0)))
                    + Decimal(str(inv.get("sgst", 0)))
                    + Decimal(str(inv.get("igst", 0)))
                )
    return float(total)


def _enrich_mismatch_fields(
    action: Dict[str, Any],
    obs: Dict[str, Any],
) -> None:
    """Fill empty mismatch_fields for MISMATCH entries using Python differ."""
    invoice_map = {inv["invoice_id"]: inv for inv in obs.get("invoices", [])}
    gstr_index: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for e in obs.get("gstr2b_entries", []):
        gstr_index[e["invoice_number"]].append(e)

    for entry in action.get("reconciliation_result", []):
        if entry.get("status") == "MISMATCH" and not entry.get("mismatch_fields"):
            inv = invoice_map.get(entry["invoice_id"])
            if inv:
                matches = gstr_index.get(inv["invoice_number"], [])
                if matches:
                    entry["mismatch_fields"] = _compute_mismatch_fields(inv, matches[0])


# ---------------------------------------------------------------------------
# Prompt Builder
# ---------------------------------------------------------------------------

def _build_prompt(obs: Dict[str, Any]) -> str:
    invoices: List[Dict[str, Any]] = obs.get("invoices", [])
    gstr: List[Dict[str, Any]] = obs.get("gstr2b_entries", [])
    task_id: str = obs.get("task_id", "unknown")
    period: str = obs.get("tax_period", "unknown")
    max_itc: float = float(obs.get("max_itc_possible", 0))
    weight_hint: str = WEIGHT_HINTS.get(task_id, "")

    inv_lines: List[str] = []
    for inv in invoices:
        cgst = float(inv.get("cgst", 0))
        sgst = float(inv.get("sgst", 0))
        igst = float(inv.get("igst", 0))
        total_tax = cgst + sgst + igst
        inv_lines.append(
            "  ID:{} | InvNo:{} | GSTIN:{} | Date:{} | "
            "Taxable:{:.2f} | CGST:{:.2f} SGST:{:.2f} IGST:{:.2f} | Tax:{:.2f}".format(
                inv["invoice_id"],
                inv["invoice_number"],
                inv["vendor_gstin"],
                inv["invoice_date"],
                float(inv["taxable_value"]),
                cgst, sgst, igst, total_tax,
            )
        )

    gstr_lines: List[str] = []
    for e in gstr:
        gstr_lines.append(
            "  InvNo:{} | GSTIN:{} | Date:{} | Taxable:{:.2f} | "
            "CGST:{:.2f} SGST:{:.2f} IGST:{:.2f} | ITC:{}".format(
                e["invoice_number"],
                e["supplier_gstin"],
                e["invoice_date"],
                float(e["taxable_value"]),
                float(e.get("cgst", 0)),
                float(e.get("sgst", 0)),
                float(e.get("igst", 0)),
                e["itc_available"],
            )
        )

    inv_block = "\n".join(inv_lines) if inv_lines else "  (none)"
    gstr_block = "\n".join(gstr_lines) if gstr_lines else "  (none)"

    return (
        "GST RECONCILIATION TASK\n"
        "=======================\n"
        "Task: {} | Period: {} | Max ITC: Rs{:,.2f}\n"
        "Scoring note: {}\n\n"
        "PURCHASE INVOICES ({} records)\n{}\n{}\n\n"
        "GSTR-2B ENTRIES ({} records)\n{}\n{}\n\n"
        "Reconcile ALL {} invoices. Return JSON only."
    ).format(
        task_id, period, max_itc, weight_hint,
        len(invoices), "-" * 60, inv_block,
        len(gstr), "-" * 60, gstr_block,
        len(invoices),
    )


# ---------------------------------------------------------------------------
# Default Fallback Action
# ---------------------------------------------------------------------------

def _default_action(obs: Dict[str, Any]) -> Dict[str, Any]:
    claim = Decimal("0")
    result: List[Dict[str, Any]] = []
    for inv in obs.get("invoices", []):
        result.append({
            "invoice_id": inv["invoice_id"],
            "status": "MATCHED",
            "correction_note": None,
            "mismatch_fields": [],
        })
        claim += (
            Decimal(str(inv.get("cgst", 0)))
            + Decimal(str(inv.get("sgst", 0)))
            + Decimal(str(inv.get("igst", 0)))
        )
    return {
        "reconciliation_result": result,
        "claimable_itc": float(claim),
        "confidence": 0.1,
    }


# ---------------------------------------------------------------------------
# LLM Response Parser
# ---------------------------------------------------------------------------

def _parse_llm_response(text: str) -> Optional[Dict[str, Any]]:
    if not text:
        return None

    text = text.strip()

    # 1. Strip <think>...</think> blocks (Qwen models emit these)
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()

    # 2. Strip markdown code fences
    text = re.sub(r"^```[a-z]*\n?", "", text).rstrip("`").strip()

    # 3. Direct parse
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # 4. Extract outermost JSON object using brace counting
    start = text.find("{")
    if start != -1:
        depth = 0
        for i, ch in enumerate(text[start:], start):
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(text[start:i + 1])
                    except json.JSONDecodeError:
                        break

    # 5. Last resort: strip comma-formatted numbers and retry
    cleaned = re.sub(r"(\d),(\d{3})", r"\1\2", text)
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start != -1 and end > start:
        try:
            return json.loads(cleaned[start:end + 1])
        except json.JSONDecodeError:
            pass

    return None


# ---------------------------------------------------------------------------
# Action Sanitiser
# ---------------------------------------------------------------------------

def _sanitise_action(
    action: Dict[str, Any],
    obs: Dict[str, Any],
) -> Dict[str, Any]:
    valid_statuses: Set[str] = {"MATCHED", "MISMATCH", "MISSING_IN_2B", "EXTRA_IN_2B"}
    seen_ids: Set[str] = set()
    cleaned: List[Dict[str, Any]] = []

    for entry in action.get("reconciliation_result", []):
        inv_id = str(entry.get("invoice_id", "")).strip()
        status = str(entry.get("status", "MATCHED")).strip()

        if status not in valid_statuses:
            status = "MATCHED"

        mf = entry.get("mismatch_fields")
        if not isinstance(mf, list):
            mf = []
        mf = [str(f).strip() for f in mf if f]

        correction = entry.get("correction_note")
        if correction is not None:
            correction = str(correction).strip() or None

        cleaned.append({
            "invoice_id": inv_id,
            "status": status,
            "correction_note": correction,
            "mismatch_fields": mf,
        })
        seen_ids.add(inv_id)

    # Fill any invoices the LLM missed
    missing_count = 0
    for inv in obs.get("invoices", []):
        if inv["invoice_id"] not in seen_ids:
            missing_count += 1
            cleaned.append({
                "invoice_id": inv["invoice_id"],
                "status": "MATCHED",
                "correction_note": None,
                "mismatch_fields": [],
            })
    if missing_count > 0:
        print("    [WARN] LLM missed {} invoices, defaulted to MATCHED".format(missing_count))

    action["reconciliation_result"] = cleaned

    # Enrich mismatch_fields deterministically where LLM left them empty
    _enrich_mismatch_fields(action, obs)

    # Recompute ITC deterministically from status labels
    action["claimable_itc"] = _recompute_itc(action, obs)

    try:
        confidence = float(action.get("confidence", 0.5))
        confidence = max(0.0, min(1.0, confidence))
    except (TypeError, ValueError):
        confidence = 0.5
    action["confidence"] = confidence

    return action


# ---------------------------------------------------------------------------
# Groq LLM Caller with Retry
# ---------------------------------------------------------------------------

def _call_groq(client: Groq, prompt: str, attempt: int = 0) -> str:
    try:
        print("    [LLM] Calling Groq {} (attempt {}/{}) ...".format(
            MODEL, attempt + 1, MAX_RETRIES + 1
        ))
        completion = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            temperature=0.3,
            max_tokens=8192,
            top_p=0.95,
            stream=False,
            stop=None,
        )
        content = completion.choices[0].message.content or ""
        usage = completion.usage
        if usage:
            print("    [LLM] Tokens: prompt={} completion={} total={}".format(
                usage.prompt_tokens, usage.completion_tokens, usage.total_tokens
            ))
        else:
            print("    [LLM] Response length: {} chars".format(len(content)))
        return content
    except Exception as exc:
        if attempt < MAX_RETRIES:
            wait = RETRY_DELAY * (attempt + 1)
            print("    [LLM] Error: {}".format(exc))
            print("    [LLM] Retrying in {:.0f}s ...".format(wait))
            time.sleep(wait)
            return _call_groq(client, prompt, attempt + 1)
        raise exc


# ---------------------------------------------------------------------------
# Status Breakdown Printer
# ---------------------------------------------------------------------------

def _print_status_breakdown(action: Dict[str, Any]) -> None:
    counts: Dict[str, int] = {
        "MATCHED": 0, "MISMATCH": 0, "MISSING_IN_2B": 0, "EXTRA_IN_2B": 0,
    }
    for entry in action.get("reconciliation_result", []):
        status = entry.get("status", "UNKNOWN")
        if status in counts:
            counts[status] += 1
    total = sum(counts.values())
    print("    [BREAKDOWN] Total submitted: {}".format(total))
    for status, count in counts.items():
        pct = (count / total * 100) if total else 0
        print("      {:<16}: {:>4}  ({:5.1f}%)".format(status, count, pct))


# ---------------------------------------------------------------------------
# Single Task Runner
# ---------------------------------------------------------------------------

def run_task(task_id: str, client: Groq) -> Dict[str, Any]:
    print("\n{}".format("=" * 65))
    print("  TASK: {}".format(task_id.upper()))
    print("{}".format("=" * 65))

    # Step 1: Reset
    print("  [1/5] Resetting environment ...")
    obs = _post("/reset", {"task_id": task_id})
    n_invoices = len(obs.get("invoices", []))
    n_gstr = len(obs.get("gstr2b_entries", []))
    max_itc = float(obs.get("max_itc_possible", 0))
    print("       Invoices loaded  : {}".format(n_invoices))
    print("       GSTR-2B entries  : {}".format(n_gstr))
    print("       Max claimable ITC: Rs{:,.2f}".format(max_itc))

    # Step 2: Pre-process
    print("  [2/5] Running deterministic pre-pass ...")
    pre_resolved, ambiguous_obs = _preprocess_invoices(obs)
    n_ambiguous = len(ambiguous_obs.get("invoices", []))
    print("       Pre-resolved: {}  |  Ambiguous (→ LLM): {}".format(
        len(pre_resolved), n_ambiguous
    ))

    # Steps 3 & 4: LLM for ambiguous invoices — sent in batches of BATCH_SIZE
    llm_entries: List[Dict[str, Any]] = []
    llm_used = False

    if n_ambiguous > 0:
        ambiguous_invoices = ambiguous_obs.get("invoices", [])
        ambiguous_gstr    = ambiguous_obs.get("gstr2b_entries", [])

        # Split into batches
        batches = [
            ambiguous_invoices[i:i + BATCH_SIZE]
            for i in range(0, len(ambiguous_invoices), BATCH_SIZE)
        ]
        n_batches = len(batches)

        print("  [3/5] Splitting {} ambiguous invoices into {} batch(es) of up to {} ...".format(
            n_ambiguous, n_batches, BATCH_SIZE
        ))

        for batch_idx, batch_invoices in enumerate(batches):
            batch_inv_nos = {inv["invoice_number"] for inv in batch_invoices}
            batch_gstr    = [e for e in ambiguous_gstr if e["invoice_number"] in batch_inv_nos]

            batch_obs = dict(ambiguous_obs)
            batch_obs["invoices"]       = batch_invoices
            batch_obs["gstr2b_entries"] = batch_gstr

            print("\n  [4/5] Batch {}/{} — {} invoices ...".format(
                batch_idx + 1, n_batches, len(batch_invoices)
            ))

            prompt = _build_prompt(batch_obs)
            print("       Prompt length: {} characters".format(len(prompt)))

            try:
                raw = _call_groq(client, prompt)
                preview = raw[:300].replace("\n", "\\n")
                print("    [DEBUG] Raw response preview: {}".format(preview))
                parsed = _parse_llm_response(raw)
                if parsed is None:
                    print("    [WARN] Could not parse LLM response for batch {}".format(
                        batch_idx + 1
                    ))
                else:
                    sanitised = _sanitise_action(parsed, batch_obs)
                    llm_entries.extend(sanitised.get("reconciliation_result", []))
                    llm_used = True
            except Exception as exc:
                print("    [ERROR] Batch {} LLM call failed: {}".format(batch_idx + 1, exc))

            # Sleep between batches so TPM window resets (skip after last batch)
            if batch_idx < n_batches - 1:
                print("    [WAIT] Sleeping {}s between batches for TPM reset ...".format(
                    int(BATCH_SLEEP)
                ))
                time.sleep(BATCH_SLEEP)

    else:
        print("  [3/5] Skipping LLM — all invoices resolved deterministically.")
        print("  [4/5] (skipped)")

    # Step 5: Merge and submit
    print("\n  [5/5] Merging results and submitting ...")
    all_entries = pre_resolved + llm_entries

    # Fallback for any invoices not covered (LLM batch failures)
    resolved_ids = {e["invoice_id"] for e in all_entries}
    for inv in obs.get("invoices", []):
        if inv["invoice_id"] not in resolved_ids:
            all_entries.append({
                "invoice_id": inv["invoice_id"],
                "status": "MATCHED",
                "correction_note": None,
                "mismatch_fields": [],
            })

    # Final enrichment + ITC recompute on full obs
    _enrich_mismatch_fields({"reconciliation_result": all_entries}, obs)
    action: Dict[str, Any] = {
        "reconciliation_result": all_entries,
        "claimable_itc": _recompute_itc({"reconciliation_result": all_entries}, obs),
        "confidence": 0.95 if not n_ambiguous else (0.85 if llm_used else 0.3),
    }

    _print_status_breakdown(action)
    print("    [ACTION] Claimed ITC : Rs{:,.2f}".format(action["claimable_itc"]))
    print("    [ACTION] Confidence  : {:.2f}".format(action["confidence"]))

    result = _post("/step", action)

    reward   = result.get("reward", {})
    info     = result.get("info", {})
    total_r  = float(reward.get("total", 0.0))
    match_sc = float(reward.get("match_score", 0.0))
    itc_sc   = float(reward.get("itc_score", 0.0))
    pen_sc   = float(reward.get("penalty_day_penalty", 0.0))
    correct  = info.get("correct_matches", 0)
    total_inv = info.get("total_invoices", n_invoices)
    itc_error = info.get("itc_error", "?")
    task_score = info.get("task_score", "?")

    print("\n  {}".format("-" * 60))
    print("  RESULTS FOR {}".format(task_id.upper()))
    print("  {}".format("-" * 60))
    print("  Total Reward      : {:.4f} / 1.0000".format(total_r))
    print("  Match Score       : {:.4f}  ({}/{} correct)".format(match_sc, correct, total_inv))
    print("  ITC Score         : {:.4f}  (error={})".format(itc_sc, itc_error))
    print("  Penalty Score     : {:.4f}".format(pen_sc))
    print("  Task Grader Score : {}".format(task_score))
    print("  LLM Used          : {}".format("Yes (Groq)" if llm_used else "No (pre-pass only)"))
    print("  {}".format("-" * 60))

    return {
        "task_id": task_id,
        "total": total_r,
        "match_score": match_sc,
        "itc_score": itc_sc,
        "correct": correct,
        "total_invoices": total_inv,
        "llm_used": llm_used,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("\n" + "#" * 65)
    print("  GST RECONCILIATION - OpenEnv Agent Runner (Groq)")
    print("#" * 65)
    print("  Model   : {}".format(MODEL))
    print("  Server  : {}".format(BASE_URL))
    print("  API Key : {}".format(
        "SET ({}...)".format(GROQ_API_KEY[:8]) if GROQ_API_KEY else "NOT SET"
    ))
    print("#" * 65)

    if not GROQ_API_KEY:
        print(
            "\n[FATAL] GROQ_API_KEY is not set.\n\n"
            "  Get your free API key from: https://console.groq.com/keys\n\n"
            "  Then set it:\n"
            "  Option A: Create .env file:\n"
            "            GROQ_API_KEY=gsk_your_key_here\n\n"
            "  Option B: Terminal:\n"
            "    Windows CMD  : set GROQ_API_KEY=gsk_your_key_here\n"
            "    PowerShell   : $env:GROQ_API_KEY='gsk_your_key_here'\n"
            "    Linux/Mac    : export GROQ_API_KEY=gsk_your_key_here\n"
        )
        sys.exit(1)

    print("\n[STARTUP] Checking server at {} ...".format(BASE_URL))
    try:
        health = _get("/health")
        print("[STARTUP] Server OK: {}".format(health))
    except Exception as exc:
        print(
            "\n[FATAL] Cannot reach server at {}\n"
            "  Start it first:\n\n"
            "    cd GST\n"
            "    python -m uvicorn gst_env.main:app --host 0.0.0.0 --port 7860\n\n"
            "  Error: {}\n".format(BASE_URL, exc)
        )
        sys.exit(1)

    try:
        tasks = _get("/tasks")
        print("[STARTUP] Available tasks: {}".format([t["task_id"] for t in tasks]))
    except Exception:
        print("[STARTUP] /tasks endpoint not available (non-fatal)")

    client = Groq(api_key=GROQ_API_KEY)
    print("[STARTUP] Groq client created with model: {}".format(MODEL))

    all_results: List[Dict[str, Any]] = []
    for task_id in ["task1_easy", "task2_medium", "task3_hard"]:
        try:
            result = run_task(task_id, client)
            all_results.append(result)
        except Exception as exc:
            print("\n  [ERROR] Task {} failed: {}".format(task_id, exc))
            import traceback
            traceback.print_exc()
            all_results.append({
                "task_id": task_id,
                "total": 0.0, "match_score": 0.0, "itc_score": 0.0,
                "correct": 0, "total_invoices": 0, "llm_used": False,
            })

    avg = sum(r["total"] for r in all_results) / len(all_results)

    print("\n\n" + "#" * 65)
    print("  FINAL SUMMARY")
    print("#" * 65)
    print("  {:<22} {:>8} {:>8} {:>8} {}".format("Task", "Total", "Match", "ITC", "Correct"))
    print("  {}".format("-" * 60))
    for r in all_results:
        correct_str = "{}/{}".format(r["correct"], r["total_invoices"])
        llm_tag = "" if r["llm_used"] else " [pre-pass]"
        print("  {:<22} {:>8.4f} {:>8.4f} {:>8.4f} {:>10}{}".format(
            r["task_id"], r["total"], r["match_score"],
            r["itc_score"], correct_str, llm_tag,
        ))
    print("  {}".format("-" * 60))
    print("  {:<22} {:>8.4f}".format("AVERAGE", avg))
    print("#" * 65)

    if avg >= 0.85:
        rating = "EXCELLENT *****"
    elif avg >= 0.70:
        rating = "GOOD      ****"
    elif avg >= 0.50:
        rating = "FAIR      ***"
    elif avg >= 0.30:
        rating = "POOR      **"
    else:
        rating = "FAILED    *"

    print("\n  Overall Rating: {}".format(rating))
    print("  Average Score : {:.4f} / 1.0000\n".format(avg))


if __name__ == "__main__":
    main()