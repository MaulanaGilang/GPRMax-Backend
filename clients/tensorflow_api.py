import requests
import json


class TensorFlowApiClient:
    def __init__(self, server_url):
        self.server_url = server_url

    def predict(self, model_name, model_input):
        endpoint = f"{self.server_url}/v1/models/{model_name}:predict"
        payload = {
            "instances": model_input
        }

        json_payload = json.dumps(payload)

        response = requests.post(endpoint, data=json_payload)
        response.raise_for_status()

        return response.json()
