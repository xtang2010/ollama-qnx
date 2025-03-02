import json
import os
import requests
from requests.exceptions import RequestException
from urllib.parse import urljoin
from typing import Any, Callable, Optional

class StatusError(Exception):
    def __init__(self, status_code: int, status: str, error_message: str):
        self.status_code = status_code
        self.status = status
        self.error_message = error_message
        super().__init__(f"Status {status_code}: {status} - {error_message}")

class Client:
    def __init__(self, base_url: str = "http://localhost:11434"):
        self.base_url = base_url
        self.session = requests.Session()

    @classmethod
    def from_environment(cls) -> 'Client':
        base_url = os.getenv("OLLAMA_HOST", "http://localhost:11434")
        return cls(base_url)

    def _do_request(self, method: str, path: str, req_data: Any = None, resp_data: Any = None) -> Optional[dict]:
        url = urljoin(self.base_url, path)
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
            "User-Agent": f"ollama/0.1.0 (x86_64 linux) Python/3.9"
        }

        if method == "GET":
            response = self.session.get(url, headers=headers)
        elif method == "POST":
            response = self.session.post(url, headers=headers, json=req_data)
        elif method == "DELETE":
            response = self.session.delete(url, headers=headers, json=req_data)
        else:
            raise ValueError(f"Unsupported method: {method}")

        if response.status_code >= 400:
            try:
                error_data = response.json()
                error_message = error_data.get("error", "")
            except json.JSONDecodeError:
                error_message = response.text
            raise StatusError(response.status_code, response.reason, error_message)

        if response.text is None or response.text == '':
            return None
        return response.json()

    def stream(self, method: str, path: str, data: Any, callback: Callable[[bytes], None]) -> None:
        url = urljoin(self.base_url, path)
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/x-ndjson",
            "User-Agent": f"ollama/0.1.0 (x86_64 linux) Python/3.9"
        }

        if method == "POST":
            response = self.session.post(url, headers=headers, json=data, stream=True)
        else:
            raise ValueError(f"Unsupported method: {method}")

        if response.status_code >= 400:
            try:
                error_data = response.json()
                error_message = error_data.get("error", "")
            except json.JSONDecodeError:
                error_message = response.text
            raise StatusError(response.status_code, response.reason, error_message)

        for line in response.iter_lines():
            if line:
                callback(line)

    def generate(self, req: dict, callback: Callable[[dict], None]) -> None:
        self.stream("POST", "/api/generate", req, lambda b: callback(json.loads(b)))

    def chat(self, req: dict, callback: Callable[[dict], None]) -> None:
        self.stream("POST", "/api/chat", req, lambda b: callback(json.loads(b)))

    def pull(self, req: dict, callback: Callable[[dict], None]) -> None:
        self.stream("POST", "/api/pull", req, lambda b: callback(json.loads(b)))

    def push(self, req: dict, callback: Callable[[dict], None]) -> None:
        self.stream("POST", "/api/push", req, lambda b: callback(json.loads(b)))

    def create(self, req: dict, callback: Callable[[dict], None]) -> None:
        self.stream("POST", "/api/create", req, lambda b: callback(json.loads(b)))

    def list(self) -> dict:
        return self._do_request("GET", "/api/tags")

    def list_running(self) -> dict:
        return self._do_request("GET", "/api/ps")

    def copy(self, req: dict) -> None:
        self._do_request("POST", "/api/copy", req)

    def delete(self, req: dict) -> None:
        self._do_request("DELETE", "/api/delete", req)

    def show(self, req: dict) -> dict:
        return self._do_request("POST", "/api/show", req)

    def heartbeat(self) -> None:
        self._do_request("HEAD", "/")

    def embed(self, req: dict) -> dict:
        return self._do_request("POST", "/api/embed", req)

    def embeddings(self, req: dict) -> dict:
        return self._do_request("POST", "/api/embeddings", req)

    def create_blob(self, digest: str, file_data: bytes) -> None:
        url = urljoin(self.base_url, f"/api/blobs/{digest}")
        headers = {
            "Content-Type": "application/octet-stream",
            "Accept": "application/json",
            "User-Agent": f"ollama/0.1.0 (x86_64 linux) Python/3.9"
        }
        response = self.session.post(url, headers=headers, data=file_data)
        if response.status_code >= 400:
            try:
                error_data = response.json()
                error_message = error_data.get("error", "")
            except json.JSONDecodeError:
                error_message = response.text
            raise StatusError(response.status_code, response.reason, error_message)

    def version(self) -> str:
        response = self._do_request("GET", "/api/version")
        return response.get("version", "")

# Example usage
if __name__ == "__main__":
    client = Client.from_environment()
    try:
        response = client.list()
        print(response)
    except StatusError as e:
        print(f"Error: {e}")