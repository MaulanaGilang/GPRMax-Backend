from fastapi import APIRouter, HTTPException, File, UploadFile, Form
from typing import List
from uuid import uuid4
import os
import tempfile
from celery.result import AsyncResult
from tasks.process_image import process_image

router = APIRouter()


@router.post("/predict/")
async def create_predict(ids: List[str] = Form(...), files: List[UploadFile] = File(...), detection_method: str = Form(...)):
    if detection_method not in ["cnn", "roboflow"]:
        raise HTTPException(
            status_code=400, detail="Detection method must be either 'cnn' or 'roboflow'.")

    job_id = str(uuid4())

    if len(ids) != len(files):
        raise HTTPException(
            status_code=400, detail="Each ID must correspond to exactly one file.")

    temp_base_dir = tempfile.gettempdir()
    job_dir = os.path.join(temp_base_dir, "upload", job_id)
    os.makedirs(job_dir, exist_ok=True)

    data: list[dict[str, str]] = []
    for idx, image_file in enumerate(files):
        image_id = ids[idx]

        extension = os.path.splitext(str(image_file.filename))[1]
        contents = await image_file.read()
        if len(contents) > 1024 * 1024:
            raise HTTPException(
                status_code=400, detail="File size too large. Each image must be less than 1MB.")

        file_path = os.path.join(job_dir, f"{image_id}{extension}")

        with open(file_path, "wb") as file:
            file.write(contents)

        data.append({
            "id": image_id,
            "path": file_path
        })

    task: AsyncResult = process_image.apply_async(
        args=[data, detection_method], task_id=job_id)

    return {"jobId": task.id}
