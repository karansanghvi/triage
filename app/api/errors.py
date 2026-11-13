"""Errors in OpenAI's format: {"error": {"message", "type", "param", "code"}}."""
import json
from typing import Optional


class APIError(Exception):
    def __init__(self, status: int, message: str, type_: str = "invalid_request_error",
                 param: Optional[str] = None, code: Optional[str] = None, headers: Optional[dict] = None):
        super().__init__(message)
        self.status, self.message, self.type, self.param, self.code = status, message, type_, param, code
        self.headers = headers or {}

    def body(self) -> dict:
        return {"error": {"message": self.message, "type": self.type, "param": self.param, "code": self.code}}


def bad_request(message: str, param: Optional[str] = None, code: str = "invalid_value") -> APIError:
    return APIError(400, message, "invalid_request_error", param, code)


def stream_error_event(message: str, code: str = "stream_interrupted") -> str:
    """The SSE event that ends a stream which failed after text was sent (no [DONE] follows).
    The openai SDKs raise an APIError when they see it."""
    body = {"error": {"message": message, "type": "server_error", "param": None, "code": code}}
    return f"data: {json.dumps(body)}\n\n"
