import os
import threading

from ...cancel import CancelToken


class GeminiError(RuntimeError):
    pass


class GeminiClient:
    def __init__(self, model: str, thinking_level: str = "low", timeout: float = 20.0):
        # The key comes only from the environment, so it stays out of logs.
        if not os.environ.get("GEMINI_API_KEY"):
            raise GeminiError("GEMINI_API_KEY not set")
        try:
            from google import genai
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

    def ask(
        self, image: bytes, mime_type: str, prompt: str, cancel: CancelToken
    ) -> str:
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

        # A thread, so that a cancel does not wait for the answer.
        threading.Thread(target=request, daemon=True).start()
        if not cancel.wait_until(done.is_set, self._timeout + 1.0):
            raise GeminiError(f"no answer within {self._timeout} s")
        if "error" in outcome:
            raise GeminiError(f"{self._model}: {outcome['error']}") from outcome[
                "error"
            ]
        if not outcome.get("text"):
            raise GeminiError(f"{self._model}: empty answer")
        return outcome["text"]
