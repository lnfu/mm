"""Gemini API client: one image + one prompt -> the answer text. No ROS.

The API key comes from the GEMINI_API_KEY environment variable (read by
google-genai), never from parameters, so it stays out of logs and config files.
"""

import os
import threading
import time

from ..errors import Canceled


class GeminiError(RuntimeError):
    pass


class GeminiClient:
    def __init__(self, model: str, thinking_level: str = "low", timeout: float = 20.0):
        """
        Args:
            model: e.g. "gemini-robotics-er-2-preview".
            thinking_level: "minimal" | "low" | "medium" | "high"; more is slower.
            timeout: s for one request.
        Raises GeminiError if GEMINI_API_KEY is not set or google-genai is missing.
        """
        if not os.environ.get("GEMINI_API_KEY"):
            raise GeminiError("GEMINI_API_KEY not set")
        try:
            from google import genai  # slow import, and only needed here
            from google.genai import types
        except ImportError as e:
            raise GeminiError("google-genai is not installed") from e
        self._types = types
        self._client = genai.Client(
            http_options=types.HttpOptions(timeout=int(timeout * 1000))
        )
        self._model = model
        self._thinking_level = thinking_level
        self._timeout = timeout

    def ask(self, image: bytes, mime_type: str, prompt: str, should_cancel) -> str:
        """Gemini's text answer about the image. Raises GeminiError or Canceled.

        The request runs in a thread so a cancel does not wait for it (its answer
        is then dropped).
        """
        types = self._types
        done = threading.Event()
        outcome = {}

        def request():
            try:
                response = self._client.models.generate_content(
                    model=self._model,
                    contents=[
                        types.Part.from_bytes(data=image, mime_type=mime_type),
                        prompt,
                    ],
                    config=types.GenerateContentConfig(
                        thinking_config=types.ThinkingConfig(
                            thinking_level=self._thinking_level
                        ),
                    ),
                )
                outcome["text"] = response.text
            except Exception as e:  # noqa: BLE001  any API or network error
                outcome["error"] = e
            finally:
                done.set()

        threading.Thread(target=request, daemon=True).start()
        deadline = time.monotonic() + self._timeout + 1.0
        while not done.wait(0.02):
            if should_cancel():
                raise Canceled()
            if time.monotonic() > deadline:
                raise GeminiError(f"no answer within {self._timeout} s")
        if "error" in outcome:
            raise GeminiError(f"{self._model}: {outcome['error']}") from outcome[
                "error"
            ]
        if not outcome.get("text"):
            raise GeminiError(f"{self._model}: empty answer")
        return outcome["text"]
