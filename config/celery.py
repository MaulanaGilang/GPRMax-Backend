from celery import Celery
import os
from dotenv import load_dotenv

load_dotenv()


def create_celery_app():
    celery = Celery(
        "worker",
        backend=os.getenv("REDIS_URL"),
        broker=os.getenv("REDIS_URL"),
    )
    celery.conf.task_routes = {"*": {"queue": "tasks"}}
    celery.conf.result_expires = 3600

    return celery


celery_app = create_celery_app()
