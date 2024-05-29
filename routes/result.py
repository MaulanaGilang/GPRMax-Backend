from fastapi import APIRouter
from celery.result import AsyncResult
from typing import Optional, Dict, Any
from config.celery import celery_app

router = APIRouter()


@router.get("/result/{job_id}")
def get_result(job_id: str):
    task: AsyncResult = AsyncResult(job_id, app=celery_app)

    response = {"state": task.state, "total": 0,
                "current": None, "current_id": None, "result": {}}

    if task.info is not None:
        response.update({
            "total": task.info.get("total", 0),
            "current": task.info.get("current", None),
            "current_id": task.info.get("id", None)
        })

    if task.ready() and isinstance(task.result, dict):
        response["result"] = task.result.get("result", {})

    return response
