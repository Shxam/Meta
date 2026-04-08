from __future__ import annotations

from typing import Any, Dict, List

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from .env import GSTReconciliationEnv
from .models import (
    Action,
    Observation,
    StateResponse,
    TaskInfo,
)
from .graders import grade as grade_action

app = FastAPI(
    title="GST Reconciliation OpenEnv",
    version="1.0.0",
    description=(
        "An OpenEnv-compliant reinforcement-learning environment for "
        "Indian GST invoice reconciliation against GSTR-2B."
    ),
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

env = GSTReconciliationEnv()


class ResetRequest(BaseModel):
    task_id: str


TASKS: List[TaskInfo] = [
    TaskInfo(
        task_id="task1_easy",
        description="10 invoices all perfectly matched in GSTR-2B.",
        difficulty="easy",
        num_invoices=10,
        invoice_range="1000-500000",
    ),
    TaskInfo(
        task_id="task2_medium",
        description="50 invoices with 8 deliberate mismatches.",
        difficulty="medium",
        num_invoices=50,
        invoice_range="1000-500000",
    ),
    TaskInfo(
        task_id="task3_hard",
        description="200 invoices with all mismatch types plus penalty days.",
        difficulty="hard",
        num_invoices=200,
        invoice_range="1000-500000",
    ),
    TaskInfo(
        task_id="task4_credit_notes",
        description="75 invoices with credit notes and itc_available mismatches.",
        difficulty="medium",
        num_invoices=75,
        invoice_range="1000-500000",
    ),
    TaskInfo(
        task_id="task5_stress",
        description="500 invoices high volume stress test with false positive penalties.",
        difficulty="hard",
        num_invoices=500,
        invoice_range="1000-500000",
    ),
    TaskInfo(
        task_id="task6_mixed_docs",
        description="150 invoices with mixed document types and all mismatch types.",
        difficulty="hard",
        num_invoices=150,
        invoice_range="1000-500000",
    ),
]


@app.get("/health", tags=["System"])
async def health() -> Dict[str, str]:
    return {"status": "ok", "version": "1.0.0"}


@app.get("/tasks", tags=["OpenEnv"])
async def list_tasks() -> List[Dict[str, Any]]:
    return [t.model_dump() for t in TASKS]


@app.get("/state", tags=["OpenEnv"])
async def get_state() -> Dict[str, Any]:
    return env.state().model_dump()


@app.post("/reset", tags=["OpenEnv"])
async def reset(request: ResetRequest) -> Dict[str, Any]:
    try:
        obs: Observation = env.reset(request.task_id)
        return obs.model_dump(mode="json")
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Internal error: {e}"
        )


@app.post("/step", tags=["OpenEnv"])
async def step(action: Action) -> Dict[str, Any]:
    try:
        obs, reward, done, info = env.step(action)
        return {
            "observation": obs.model_dump(mode="json"),
            "reward": reward.model_dump(mode="json"),
            "done": done,
            "info": info,
        }
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Internal error: {e}"
        )


@app.post("/grader", tags=["Evaluation"])
async def grader_endpoint(action: Action) -> Dict[str, Any]:
    if not env.current_task_id:
        raise HTTPException(
            status_code=400, detail="No active episode. Call /reset first."
        )

    gt = env.ground_truth
    task = env.current_task_id

    try:
        score = grade_action(task, action, gt)
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Grading error: {e}"
        )

    true_itc = float(gt.get("max_itc", 0.0))
    pred_itc = float(action.claimable_itc)
    itc_error = abs(pred_itc - true_itc) / (true_itc + 1e-9)

    total = len(action.reconciliation_result)
    correct = sum(
        1
        for e in action.reconciliation_result
        if gt.get(e.invoice_id) == e.status
    )

    return {
        "task_id": task,
        "score": score,
        "breakdown": {
            "accuracy": round(correct / total, 4) if total else 0.0,
            "itc_error": round(itc_error, 4),
            "itc_score": round(max(0.0, 1.0 - itc_error), 4),
            "confidence": float(action.confidence),
            "correct_matches": correct,
            "total_submitted": total,
        },
    }


@app.get("/baseline", tags=["Evaluation"])
async def baseline() -> Dict[str, Any]:
    from .baseline import run_baseline
    return run_baseline()