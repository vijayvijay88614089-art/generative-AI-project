from functools import lru_cache
import json
from urllib.error import URLError
from urllib.request import Request, urlopen

DEFAULT_QWEN_MODEL = "qwen2.5-coder:1.5b"
QWEN_MIN_AVAILABLE_RAM_GIB = 18.0
OLLAMA_URL = "http://127.0.0.1:11434"


def is_ollama_model_id(model_id: str) -> bool:
    return ":" in model_id and "/" not in model_id and "\\" not in model_id


def ollama_model_status(model_id: str = DEFAULT_QWEN_MODEL) -> tuple[bool, str]:
    """Report whether the local Ollama service has the selected model pulled."""
    try:
        with urlopen(f"{OLLAMA_URL}/api/tags", timeout=2) as response:
            models = json.load(response).get("models", [])
    except (OSError, URLError, TimeoutError):
        return False, "Ollama is not running. Start Ollama and retry."
    names = {item.get("name", "") for item in models}
    if model_id in names or any(name.split(":", 1)[0] == model_id.split(":", 1)[0] and model_id.endswith(":latest") for name in names):
        return True, ""
    return False, f"Model `{model_id}` is not downloaded yet. In a terminal run `ollama pull {model_id}`."


def model_capacity_status(model_id: str) -> tuple[bool, str]:
    """Avoid starting full Qwen 7B BF16 inference when host RAM cannot hold it."""
    if "qwen2.5-coder-7b-instruct" not in model_id.casefold():
        return True, ""
    import torch
    if torch.cuda.is_available():
        return True, ""
    import psutil
    memory = psutil.virtual_memory()
    available_gib = memory.available / (1024 ** 3)
    total_gib = memory.total / (1024 ** 3)
    if available_gib < QWEN_MIN_AVAILABLE_RAM_GIB:
        return False, (
            f"Qwen2.5-Coder-7B-Instruct needs about {QWEN_MIN_AVAILABLE_RAM_GIB:.0f} GiB of available RAM "
            f"for this unquantized CPU setup. This system has {total_gib:.1f} GiB total and "
            f"{available_gib:.1f} GiB available, with no CUDA GPU. Use a quantized Qwen build "
            "with a compatible runtime, or a computer with more memory."
        )
    return True, ""


def _execution_device(model, torch):
    """Choose a real compute device, not a meta placeholder from disk offload."""
    device_map = getattr(model, "hf_device_map", {}) or {}
    for placement in device_map.values():
        if placement in {"disk", "meta"}:
            continue
        if isinstance(placement, int):
            return torch.device(f"cuda:{placement}")
        return torch.device(placement)
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


@lru_cache(maxsize=1)
def _load_model(model_id: str):
    """Load one Hugging Face model and keep it in memory for this app process."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype="auto",
        device_map="auto",
    )
    model.eval()
    return tokenizer, model, torch


def generate_local(model_id: str, prompt: str, max_new_tokens: int = 256,
                   system_prompt: str = "You are a helpful assistant.") -> str:
    """Generate with a local or Hugging Face model ID using its chat template.

    The first call for a remote ID downloads its weights into the Hugging Face
    cache. Later calls reuse the loaded model in memory.
    """
    if not model_id.strip():
        raise ValueError("Enter a local model folder or Hugging Face model ID.")
    model_id = model_id.strip()
    if is_ollama_model_id(model_id):
        payload = json.dumps({
            "model": model_id,
            "messages": [
                {"role": "system", "content": system_prompt[:4000]},
                {"role": "user", "content": prompt[:12000]},
            ],
            "stream": False,
            "options": {"temperature": 0, "num_predict": max_new_tokens},
        }).encode("utf-8")
        request = Request(f"{OLLAMA_URL}/api/chat", data=payload, headers={"Content-Type": "application/json"})
        try:
            with urlopen(request, timeout=600) as response:
                return json.load(response)["message"]["content"].strip()
        except (OSError, URLError, TimeoutError) as exc:
            raise RuntimeError(f"Ollama could not generate a response: {exc}") from exc
    ready, reason = model_capacity_status(model_id.strip())
    if not ready:
        raise RuntimeError(reason)
    tokenizer, model, torch = _load_model(model_id.strip())
    messages = [
        {"role": "system", "content": system_prompt[:4000]},
        {"role": "user", "content": prompt[:12000]},
    ]
    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    ).to(_execution_device(model, torch))
    with torch.inference_mode():
        output = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    generated = output[0][inputs["input_ids"].shape[-1]:]
    return tokenizer.decode(generated, skip_special_tokens=True).strip()
