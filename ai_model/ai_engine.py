# ai_model/ai_engine.py
import requests
import json
from .config import MODEL_NAME, OLLAMA_API_URL


def stream_response(prompt: str):
    """
    Connect to Ollama and yield text chunks as they arrive.
    UI layers consume this generator for live typing.
    """
    payload = {"model": MODEL_NAME, "prompt": prompt, "stream": True}
    headers = {"Content-Type": "application/json"}

    try:
        with requests.post(
            OLLAMA_API_URL, headers=headers, json=payload, stream=True, timeout=120
        ) as resp:
            resp.raise_for_status()

            for line in resp.iter_lines():
                if not line:
                    continue
                try:
                    data = json.loads(line.decode("utf-8"))
                    chunk = data.get("response", "")
                    if chunk:
                        yield chunk
                except json.JSONDecodeError:
                    continue
    except Exception as e:
        yield f"⚠️ Error connecting to model: {e}"