from __future__ import annotations

from typing import Any, Dict

from .grader1 import grade as grade_task1
from .grader2 import grade as grade_task2
from .grader3 import grade as grade_task3
from .grader4 import grade as grade_task4
from .grader5 import grade as grade_task5
from .grader6 import grade as grade_task6


def grade(task_id: str, action: Any, ground_truth: Dict[str, Any]) -> float:
    if task_id == "task1_easy":
        return grade_task1(action, ground_truth)
    elif task_id == "task2_medium":
        return grade_task2(action, ground_truth)
    elif task_id == "task3_hard":
        return grade_task3(action, ground_truth)
    elif task_id == "task4_credit_notes":
        return grade_task4(action, ground_truth)
    elif task_id == "task5_stress":
        return grade_task5(action, ground_truth)
    elif task_id == "task6_mixed_docs":
        return grade_task6(action, ground_truth)
    else:
        raise ValueError(f"Unknown task_id: {task_id!r}")


__all__ = [
    "grade_task1", "grade_task2", "grade_task3",
    "grade_task4", "grade_task5", "grade_task6",
    "grade",
]