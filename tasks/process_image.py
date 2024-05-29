import os
import shutil
from config.celery import celery_app
import cv2
import logging
from clients.tensorflow_api import TensorFlowApiClient
from clients.roboflow_api import RoboflowModelClient
from config.model import TENSORFLOW_MODEL_API_URL, CNN_MODEL_NAME, YOLO_MODEL_NAME, ROBOFLOW_API_KEY, ROBOFLOW_API_URL, ROBOFLOW_MODEL_VERSION
import numpy as np
from utils.predict_utils import predict_class
import base64

logger = logging.getLogger(__name__)

tensorflow_client = TensorFlowApiClient(TENSORFLOW_MODEL_API_URL)
roboflow_client = RoboflowModelClient(
    ROBOFLOW_API_KEY, ROBOFLOW_API_URL, ROBOFLOW_MODEL_VERSION)


def process(data: list[dict[str, str]], detection_method: str):
    counter = 0

    for d in data:
        path, id = d["path"], d["id"]

        try:
            img = cv2.imread(path)
            if img is None:
                raise Exception("Failed to read image.")

            img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            result = {}
            airgap = False
            if detection_method == "cnn":
                img_resized = cv2.resize(img_gray, (540, 330))
                img_binarization = cv2.adaptiveThreshold(
                    img_resized, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
                img_normalized = img_binarization.astype(float) / 255.0
                input = np.expand_dims(img_normalized, axis=0)
                input = np.expand_dims(input, axis=-1)

                try:
                    cnn_result = tensorflow_client.predict(
                        CNN_MODEL_NAME, input.tolist())
                    probability = cnn_result["predictions"][0][0]
                    print(probability)

                    predicted_class, confidence = predict_class(probability)

                    if (predicted_class == 0 and confidence >= 0.7):  # 0 is the class for airgap
                        airgap = True

                    result["airgap_detected"] = airgap
                    result["classification_confidence"] = confidence

                except Exception as e:
                    logger.exception(e)
                    raise Exception("Failed to predict using CNN model.", e)

            elif detection_method == "roboflow":
                img_resized = cv2.resize(img_gray, (640, 640))

                _, img_encoded = cv2.imencode('.jpg', img_resized)
                img_base64 = base64.b64encode(img_encoded.tobytes())
                img_base64_string = 'data:image/jpeg;base64,' + \
                    img_base64.decode('utf-8')

                roboflow_result = roboflow_client.predict(img_base64_string)

                predicted_class = roboflow_result["top"]
                confidence = roboflow_result["confidence"]

                if predicted_class == "airgap" and confidence >= 0.7:
                    airgap = True

                result["airgap_detected"] = airgap
                result["classification_confidence"] = confidence

            else:
                raise Exception("Invalid detection method.")

            if airgap == True:
                _, img_encoded = cv2.imencode('.jpg', img_gray)
                img_base64 = base64.urlsafe_b64encode(img_encoded.tobytes())
                img_base64_string = img_base64.decode('utf-8')

                input = [{
                    "b64_image": img_base64_string
                }]

                yolo_result = tensorflow_client.predict(YOLO_MODEL_NAME, input)
                yolo_predictions = yolo_result["predictions"]

                result["bboxes"] = []
                for prediction in yolo_predictions:
                    ymin, xmin, ymax, xmax, prediction_class, confidence = prediction

                    result["bboxes"].append({
                        "ymin": ymin,
                        "xmin": xmin,
                        "ymax": ymax,
                        "xmax": xmax,
                        "class": "airgap" if prediction_class == 0 else "background",
                        "confidence": confidence
                    })

            yield counter, result, id, True, None

        except Exception as e:
            logger.exception(e)

            yield counter, None, id, False, str(e)

        counter += 1


@celery_app.task(name="process_image", bind=True, trail=True)
def process_image(self, image_data: list[dict[str, str]], method: str):
    total = len(image_data)

    successful = []
    failed = []
    for progress in process(image_data, method):
        i, result, id, success, error_reason = progress

        self.update_state(state="IN_PROGRESS", meta={
                          "current": i, "total": total, "id": id})

        if success:
            successful.append({
                "id": id,
                "result": result
            })
        else:
            failed.append({
                "id": id,
                "reason": error_reason
            })

    logger.info("Cleaning up temporary files.")
    folder_path = os.path.dirname(image_data[0]['path'])
    shutil.rmtree(folder_path)

    return {
        "total": total,
        "current": total,
        "id": None,
        "detection_method": method,
        "result": {
            "successful": successful,
            "failed": failed
        },
    }
