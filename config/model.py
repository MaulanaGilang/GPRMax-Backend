import os
from dotenv import load_dotenv

load_dotenv()

TENSORFLOW_MODEL_API_URL = os.getenv("TENSORFLOW_API")
ROBOFLOW_API_URL = os.getenv("ROBOFLOW_API")
ROBOFLOW_API_KEY = os.getenv("ROBOFLOW_API_KEY")
ROBOFLOW_MODEL_VERSION = os.getenv("ROBOFLOW_MODEL_VERSION")
YOLO_MODEL_NAME = os.getenv("YOLO_MODEL_NAME")
CNN_MODEL_NAME = os.getenv("CNN_MODEL_NAME")
