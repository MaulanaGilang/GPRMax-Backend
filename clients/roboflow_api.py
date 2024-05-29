import requests


class RoboflowModelClient:
    def __init__(self, api_key, server_url, model_version):
        self.api_key = api_key
        self.server_url = server_url
        self.model_version = model_version

    def predict(self, model_input):
        endpoint = f"{self.server_url}/{self.model_version}?api_key={self.api_key}"

        response = requests.post(endpoint, data=model_input, headers={
                                 "Content-Type": "application/json"})
        response.raise_for_status()

        return response.json()
