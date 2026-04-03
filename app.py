import gc
import json
import os
import re
import time
from collections import OrderedDict

import gradio as gr
import torch
from huggingface_hub import snapshot_download
from PIL import Image
from transformers import BitsAndBytesConfig

IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".gif", ".webp")
VIDEO_EXTENSIONS = (".mp4", ".mov", ".avi", ".webm", ".mkv", ".gif", ".flv")

DEFAULT_PROMPT = "Describe this media in extensive details"

DEFAULT_MERGE_PROMPT = """You are a caption consolidation model.

Your task is to merge multiple source captions of the same image into one final training caption.

Rules:
1. Preserve the required leading tags exactly if they already appear in the source captions or the user prompt requires them.
2. Combine the most specific non-conflicting visible details from the source captions.
3. Prefer concrete visible details over vague adjectives.
4. Keep explicit anatomical, pose, expression, clothing, and composition details if present in the source captions.
5. Remove duplication and awkward repetition.
6. Do not invent any detail that does not appear in at least one source caption.
7. If source captions conflict, choose the more concrete visually grounded wording or use conservative neutral wording.
8. Do not moralize, censor, apologize, refuse, warn, or add safety commentary.
9. Do not explain your reasoning.
10. Output only one final caption, as a plain training caption, with no label or extra formatting.
"""

MERGE_PROMPT_VERSION = "merge_prompt_v1"

DEFAULT_MODEL_ID = "prithivMLmods/Qwen3-VL-8B-Abliterated-Caption-it"
DEFAULT_MERGE_MODEL_ID = "prithivMLmods/Qwen3-VL-8B-Abliterated-Caption-it"
DEFAULT_QUANT = "None"  # "None" | "8-bit" | "4-bit"


def detect_flash_attention_2_available():
    try:
        from transformers.utils import is_flash_attn_2_available

        return bool(is_flash_attn_2_available())
    except Exception:
        return False


FLASH_ATTN_AVAILABLE = detect_flash_attention_2_available()
DEFAULT_ATTN = (
    "flash_attention_2" if FLASH_ATTN_AVAILABLE and os.name != "nt" else "eager"
)
RAW_SUBFOLDER_NAME = "_captions_raw"

AVAILABLE_MODELS = [
    "Qwen/Qwen3-VL-4B-Instruct",
    "Qwen/Qwen3-VL-8B-Instruct",
    "Qwen/Qwen2.5-VL-3B-Instruct",
    "Qwen/Qwen2.5-VL-7B-Instruct",
    "Disty0/Qwen3-VL-8B-NSFW-Caption-V4",
    "Disty0/Qwen3-VL-8B-NSFW-Caption-V4.5",
    "huihui-ai/Huihui-Qwen3-VL-8B-Instruct-abliterated",
    "prithivMLmods/Qwen3-VL-8B-Abliterated-Caption-it",
    "huihui-ai/Huihui-Qwen3-VL-30B-A3B-Instruct-abliterated",
    "prithivMLmods/Qwen3-VL-30B-A3B-Instruct-abliterated-v1",
    "fancyfeast/llama-joycaption-alpha-two-hf-llava",
    "fancyfeast/llama-joycaption-beta-one-hf-llava",
    "Custom...",
]

ATTN_CHOICES = ["flash_attention_2", "eager"] if FLASH_ATTN_AVAILABLE else ["eager"]
RUN_MODE_CHOICES = ["Single", "Multi-Pass Folder"]


def is_qwen35_model(model_id: str) -> bool:
    return "Qwen3.5" in model_id or "Qwen3_5" in model_id


def is_qwen3_vl_moe_model(model_id: str) -> bool:
    model_id_l = model_id.lower()
    return "qwen3-vl-30b-a3b" in model_id_l or "qwen3_vl_moe" in model_id_l


def is_gguf_model(model_id: str) -> bool:
    model_id_l = model_id.lower().strip()
    return "gguf" in model_id_l or model_id_l.endswith(".gguf")


def is_joycaption_model(model_id: str) -> bool:
    model_id_l = model_id.lower()
    return "joycaption" in model_id_l or "hf-llava" in model_id_l


def get_model_backend(model_id: str) -> str:
    if is_gguf_model(model_id):
        return "unsupported_gguf"
    if is_joycaption_model(model_id):
        return "joycaption_llava"
    return "qwen"


MERGE_MODEL_CHOICES = [
    m for m in AVAILABLE_MODELS if m != "Custom..." and get_model_backend(m) == "qwen"
]


QUANT_CHOICES = ["None", "8-bit", "4-bit"]
RESOLUTION_CHOICES = ["auto", "auto_high", "fast", "high"]
RAW_FILE_HANDLING_CHOICES = ["Reuse existing raw files", "Overwrite raw files"]
USER_DEFAULTS_PATH = os.path.join(
    os.path.dirname(__file__), "captioner_user_defaults.json"
)

APP_DEFAULTS = {
    "model_dropdown": DEFAULT_MODEL_ID,
    "model_multiselect": [DEFAULT_MODEL_ID],
    "custom_model_box": "",
    "merge_model_dropdown": DEFAULT_MERGE_MODEL_ID,
    "quant_dropdown": DEFAULT_QUANT,
    "attn_dropdown": DEFAULT_ATTN,
    "attn_multiselect": [DEFAULT_ATTN],
    "run_mode_radio": "Single",
    "multi_model_checkbox": False,
    "multi_atten_checkbox": False,
    "advanced_dual_load_checkbox": False,
    "enable_merge_checkbox": True,
    "show_merge_prompt_checkbox": False,
    "save_raw_captions_checkbox": True,
    "save_audit_checkbox": True,
    "skip_existing_checkbox": False,
    "retain_preview_checkbox": True,
    "summary_mode": False,
    "one_sentence_mode": False,
    "resolution_mode": "auto",
    "raw_file_handling_radio": "Reuse existing raw files",
    "max_tokens_slider": 512,
}


def refresh_model_choice_lists():
    global MERGE_MODEL_CHOICES
    MERGE_MODEL_CHOICES = [
        m
        for m in AVAILABLE_MODELS
        if m != "Custom..." and get_model_backend(m) == "qwen"
    ]


def add_model_choice(model_id: str):
    model_id = (model_id or "").strip()
    if not model_id or model_id == "Custom...":
        return False

    if model_id not in AVAILABLE_MODELS:
        insert_at = (
            AVAILABLE_MODELS.index("Custom...")
            if "Custom..." in AVAILABLE_MODELS
            else len(AVAILABLE_MODELS)
        )
        AVAILABLE_MODELS.insert(insert_at, model_id)

    refresh_model_choice_lists()
    return True


def clamp_max_tokens(value):
    try:
        value = int(value)
    except Exception:
        value = APP_DEFAULTS["max_tokens_slider"]
    return max(304, min(2048, value))


def sanitize_saved_settings(settings: dict):
    s = dict(APP_DEFAULTS)
    s.update(settings or {})

    bool_keys = [
        "multi_model_checkbox",
        "multi_atten_checkbox",
        "advanced_dual_load_checkbox",
        "enable_merge_checkbox",
        "show_merge_prompt_checkbox",
        "save_raw_captions_checkbox",
        "save_audit_checkbox",
        "skip_existing_checkbox",
        "retain_preview_checkbox",
        "summary_mode",
        "one_sentence_mode",
    ]
    for key in bool_keys:
        s[key] = bool(s.get(key, APP_DEFAULTS[key]))

    s["model_dropdown"] = (
        str(
            s.get("model_dropdown", APP_DEFAULTS["model_dropdown"]) or DEFAULT_MODEL_ID
        ).strip()
        or DEFAULT_MODEL_ID
    )
    s["custom_model_box"] = str(s.get("custom_model_box", "") or "").strip()
    s["merge_model_dropdown"] = (
        str(
            s.get("merge_model_dropdown", APP_DEFAULTS["merge_model_dropdown"])
            or DEFAULT_MERGE_MODEL_ID
        ).strip()
        or DEFAULT_MERGE_MODEL_ID
    )

    s["quant_dropdown"] = (
        s["quant_dropdown"]
        if s.get("quant_dropdown") in QUANT_CHOICES
        else DEFAULT_QUANT
    )
    s["attn_dropdown"] = (
        s["attn_dropdown"] if s.get("attn_dropdown") in ATTN_CHOICES else DEFAULT_ATTN
    )
    s["run_mode_radio"] = (
        s["run_mode_radio"] if s.get("run_mode_radio") in RUN_MODE_CHOICES else "Single"
    )
    s["resolution_mode"] = (
        s["resolution_mode"]
        if s.get("resolution_mode") in RESOLUTION_CHOICES
        else "auto"
    )
    s["raw_file_handling_radio"] = (
        s["raw_file_handling_radio"]
        if s.get("raw_file_handling_radio") in RAW_FILE_HANDLING_CHOICES
        else RAW_FILE_HANDLING_CHOICES[0]
    )
    s["max_tokens_slider"] = clamp_max_tokens(
        s.get("max_tokens_slider", APP_DEFAULTS["max_tokens_slider"])
    )

    model_multiselect = s.get("model_multiselect", [s["model_dropdown"]])
    if not isinstance(model_multiselect, list):
        model_multiselect = [model_multiselect]
    model_multiselect = [
        str(m).strip()
        for m in model_multiselect
        if str(m).strip() and str(m).strip() != "Custom..."
    ]
    if not model_multiselect and s["model_dropdown"] != "Custom...":
        model_multiselect = [s["model_dropdown"]]
    s["model_multiselect"] = model_multiselect

    attn_multiselect = s.get("attn_multiselect", [s["attn_dropdown"]])
    if not isinstance(attn_multiselect, list):
        attn_multiselect = [attn_multiselect]
    attn_multiselect = [a for a in attn_multiselect if a in ATTN_CHOICES]
    if not attn_multiselect:
        attn_multiselect = [s["attn_dropdown"]]
    s["attn_multiselect"] = attn_multiselect

    add_model_choice(s["model_dropdown"])
    for m in s["model_multiselect"]:
        add_model_choice(m)
    add_model_choice(s["merge_model_dropdown"])

    if get_model_backend(s["merge_model_dropdown"]) != "qwen":
        s["merge_model_dropdown"] = DEFAULT_MERGE_MODEL_ID
        add_model_choice(s["merge_model_dropdown"])

    return s


def load_user_defaults():
    if not os.path.exists(USER_DEFAULTS_PATH):
        return sanitize_saved_settings({})

    try:
        with open(USER_DEFAULTS_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            data = {}
    except Exception:
        data = {}

    return sanitize_saved_settings(data)


def save_user_defaults_file(settings: dict):
    settings = sanitize_saved_settings(settings)
    with open(USER_DEFAULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(settings, f, indent=2, ensure_ascii=False)
    return settings


def extract_hf_repo_id(text: str) -> str:
    text = (text or "").strip()
    if not text:
        raise ValueError("Paste a Hugging Face model URL or repo id.")

    text = text.split("?", 1)[0].split("#", 1)[0].rstrip("/")

    if "huggingface.co/" in text:
        text = text.split("huggingface.co/", 1)[1]

    parts = [p for p in text.split("/") if p]
    if parts and parts[0] == "models":
        parts = parts[1:]

    if len(parts) >= 2:
        repo_id = f"{parts[0]}/{parts[1]}"
    else:
        repo_id = text.strip("/")

    if not re.match(r"^[^/]+/[^/]+$", repo_id):
        raise ValueError("Could not parse a valid Hugging Face repo id.")

    return repo_id


# Globals
processor = None
current_model_id = DEFAULT_MODEL_ID
current_quant = DEFAULT_QUANT
current_backend = "qwen"
current_attn_impl = DEFAULT_ATTN
model = None
should_abort = False
ui_e = {}

# Optional cache for advanced dual-load
model_cache = OrderedDict()
current_cache_key = None


def build_bnb_config(quant_choice: str):
    if quant_choice == "8-bit":
        return BitsAndBytesConfig(load_in_8bit=True)
    if quant_choice == "4-bit":
        return BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16
        )
    return None


def pick_model_class(model_id: str):
    from transformers import AutoModelForImageTextToText

    backend = get_model_backend(model_id)

    if backend == "unsupported_gguf":
        raise ValueError(
            f"GGUF models are not supported in this app backend: {model_id}. "
            f"Use a standard Hugging Face Transformers model instead."
        )

    if backend == "joycaption_llava":
        from transformers import LlavaForConditionalGeneration

        return LlavaForConditionalGeneration

    try:
        if is_qwen35_model(model_id):
            from transformers import Qwen3_5ForConditionalGeneration

            return Qwen3_5ForConditionalGeneration

        if is_qwen3_vl_moe_model(model_id):
            from transformers import Qwen3VLMoeForConditionalGeneration

            return Qwen3VLMoeForConditionalGeneration

        if "Qwen3-VL" in model_id:
            from transformers import Qwen3VLForConditionalGeneration

            return Qwen3VLForConditionalGeneration

        if "Qwen2.5-VL" in model_id or "Qwen2_5-VL" in model_id:
            from transformers import Qwen2_5_VLForConditionalGeneration

            return Qwen2_5_VLForConditionalGeneration
    except Exception:
        pass

    return AutoModelForImageTextToText


def make_cache_key(model_id: str, quant_choice: str, attn_impl: str):
    return f"{model_id}||{quant_choice}||{attn_impl}"


def clear_cuda_cache():
    if torch.cuda.is_available():
        try:
            torch.cuda.synchronize()
        except Exception:
            pass
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass
        try:
            torch.cuda.ipc_collect()
        except Exception:
            pass


def evict_cache_entry(key):
    global model, processor, current_cache_key
    entry = model_cache.pop(key, None)
    if not entry:
        return

    if current_cache_key == key:
        model = None
        processor = None
        current_cache_key = None

    try:
        del entry["model"]
    except Exception:
        pass
    try:
        del entry["processor"]
    except Exception:
        pass

    gc.collect()
    clear_cuda_cache()


def unload_model(clear_cache=True):
    print("[DEBUG] Unload currently loaded model")
    global model, processor, current_cache_key

    if clear_cache:
        for key in list(model_cache.keys()):
            evict_cache_entry(key)

    try:
        del model
    except Exception:
        pass

    try:
        del processor
    except Exception:
        pass

    model = None
    processor = None
    current_cache_key = None
    gc.collect()
    clear_cuda_cache()


def supports_attention(model_id: str, attn_impl: str) -> bool:
    if attn_impl not in ATTN_CHOICES:
        return False

    if attn_impl == "flash_attention_2" and not FLASH_ATTN_AVAILABLE:
        return False

    backend = get_model_backend(model_id)
    if backend == "unsupported_gguf":
        return False

    if backend in ("qwen", "joycaption_llava"):
        return attn_impl == "eager" or (
            attn_impl == "flash_attention_2" and FLASH_ATTN_AVAILABLE
        )

    return False


def sanitize_model_slug(model_id: str) -> str:
    slug = model_id.strip().lower()
    slug = slug.replace("/", "-")
    slug = slug.replace("\\", "-")
    slug = re.sub(r"[^a-z0-9._-]+", "-", slug)
    slug = re.sub(r"-{2,}", "-", slug).strip("-")
    return slug or "model"


def get_attention_suffix(attn_impl: str) -> str:
    if attn_impl == "flash_attention_2":
        return "fa2"
    return attn_impl.strip().lower()


def serialize_for_debug(obj):
    if isinstance(obj, dict):
        return {k: serialize_for_debug(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [serialize_for_debug(i) for i in obj]
    elif isinstance(obj, Image.Image):
        return f"<Image {obj.size} {obj.mode}>"
    else:
        return obj


def _instantiate_model_processor(
    model_id: str, quant_choice: str, attn_impl: str, allow_attn_fallback: bool = True
):
    backend = get_model_backend(model_id)

    if backend == "unsupported_gguf":
        raise ValueError(
            f"GGUF model detected: {model_id}\n\n"
            "This app only supports standard Hugging Face Transformers checkpoints.\n"
            "GGUF multimodal models need a different runtime/backend."
        )

    kwargs = {
        "device_map": "auto",
    }

    kwargs["torch_dtype"] = (
        torch.bfloat16 if torch.cuda.is_available() else torch.float32
    )
    kwargs["attn_implementation"] = attn_impl

    bnb = build_bnb_config(quant_choice)
    if bnb is not None:
        kwargs["quantization_config"] = bnb

    ModelCls = pick_model_class(model_id)
    actual_attn = attn_impl

    try:
        loaded_model = ModelCls.from_pretrained(model_id, **kwargs)
    except Exception:
        if allow_attn_fallback and attn_impl == "flash_attention_2":
            kwargs["attn_implementation"] = "eager"
            loaded_model = ModelCls.from_pretrained(model_id, **kwargs)
            actual_attn = "eager"
        else:
            raise

    from transformers import AutoProcessor as _AP

    loaded_processor = _AP.from_pretrained(model_id)

    return loaded_model, loaded_processor, backend, actual_attn


def activate_cached_model(key):
    global \
        model, \
        processor, \
        current_model_id, \
        current_quant, \
        current_backend, \
        current_attn_impl, \
        current_cache_key

    entry = model_cache[key]
    model = entry["model"]
    processor = entry["processor"]
    current_model_id = entry["model_id"]
    current_quant = entry["quant_choice"]
    current_backend = entry["backend"]
    current_attn_impl = entry["actual_attn"]
    current_cache_key = key
    model_cache.move_to_end(key)


def load_selected_model_cached(
    model_id: str,
    quant_choice: str,
    attn_impl: str = DEFAULT_ATTN,
    max_cached_models: int = 2,
):
    global current_model_id, current_quant, current_backend, current_attn_impl

    key = make_cache_key(model_id, quant_choice, attn_impl)
    if key in model_cache:
        activate_cached_model(key)
        return get_model_info()

    loaded_model, loaded_processor, backend, actual_attn = _instantiate_model_processor(
        model_id, quant_choice, attn_impl, allow_attn_fallback=False
    )

    while len(model_cache) >= max_cached_models:
        oldest_key = next(iter(model_cache.keys()))
        evict_cache_entry(oldest_key)

    model_cache[key] = {
        "model": loaded_model,
        "processor": loaded_processor,
        "model_id": model_id,
        "quant_choice": quant_choice,
        "backend": backend,
        "actual_attn": actual_attn,
    }

    activate_cached_model(key)
    return get_model_info()


def load_selected_model(
    model_id: str,
    quant_choice: str,
    attn_impl: str = DEFAULT_ATTN,
    allow_attn_fallback: bool = True,
):
    print("[DEBUG] Loading selected model:", model_id)
    global \
        model, \
        processor, \
        current_model_id, \
        current_quant, \
        current_backend, \
        current_attn_impl

    unload_model(clear_cache=True)

    loaded_model, loaded_processor, backend, actual_attn = _instantiate_model_processor(
        model_id, quant_choice, attn_impl, allow_attn_fallback=allow_attn_fallback
    )

    model = loaded_model
    processor = loaded_processor
    current_model_id = model_id
    current_quant = quant_choice
    current_backend = backend
    current_attn_impl = actual_attn
    return get_model_info()


def get_model_info():
    global model, current_backend, current_attn_impl
    if model is None:
        return "Model not loaded.", "N/A", "N/A", "N/A", "N/A"

    model_name = (
        model.config._name_or_path
        if hasattr(model.config, "_name_or_path")
        else "Unknown Model"
    )
    device = "CUDA" if torch.cuda.is_available() else "CPU"

    if torch.cuda.is_available():
        vram_used = f"{torch.cuda.memory_allocated() / 1e9:.2f} GB"
        vram_total = f"{torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB"
    else:
        vram_used, vram_total = "N/A", "N/A"

    try:
        dtype = str(next(model.parameters()).dtype)
    except Exception:
        dtype = "Unknown"

    cfg = (
        f"backend={current_backend}\nattention={current_attn_impl}\n{str(model.config)}"
    )
    return model_name, device, f"{vram_used} / {vram_total}", dtype, cfg


print("[DEBUG] Cuda available:", torch.cuda.is_available())


def move_inputs_to_device(inputs, target_device: str):
    if hasattr(inputs, "to"):
        try:
            return inputs.to(target_device)
        except Exception:
            pass

    moved = {}
    for k, v in inputs.items():
        if hasattr(v, "to"):
            moved[k] = v.to(target_device)
        else:
            moved[k] = v
    return moved


def generate_caption_qwen(
    media_path,
    prompt,
    max_tokens,
    summary_mode=False,
    one_sentence_mode=False,
    resolution_mode="auto",
):
    global processor, model, current_model_id
    from qwen_vl_utils import process_vision_info

    assert model is not None, "Model must be loaded before generating captions."
    assert processor is not None, "Processor must be loaded before generating captions."

    if summary_mode and one_sentence_mode:
        prompt += " Give a one-sentence summary of the scene."
    elif summary_mode:
        prompt += " Give a short summary of the scene."
    elif one_sentence_mode:
        prompt += " Describe this image in one sentence."

    ext = os.path.splitext(media_path)[-1].lower()
    is_video = ext in VIDEO_EXTENSIONS
    content_type = "video" if is_video else "image"

    if is_video:
        media_data = media_path
    else:
        media_data = Image.open(media_path).convert("RGB")

    content_block = {"type": content_type, content_type: media_data}

    if not is_video:
        if resolution_mode == "auto":
            content_block["min_pixels"] = 256 * 28 * 28
            content_block["max_pixels"] = 896 * 28 * 28
        elif resolution_mode == "auto_high":
            content_block["min_pixels"] = 256 * 28 * 28
            content_block["max_pixels"] = 1280 * 28 * 28
        elif resolution_mode == "fast":
            content_block["resized_height"] = 392
            content_block["resized_width"] = 392
        elif resolution_mode == "high":
            content_block["resized_height"] = 728
            content_block["resized_width"] = 728

    messages = [
        {
            "role": "user",
            "content": [
                content_block,
                {"type": "text", "text": prompt},
            ],
        }
    ]

    qwen35 = is_qwen35_model(current_model_id)
    if qwen35:
        messages.append({"role": "assistant", "content": "<think>\n\n</think>\n\n"})

    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=not qwen35,
        continue_final_message=qwen35,
    )

    vision_info = process_vision_info(messages)
    if not isinstance(vision_info, tuple):
        raise ValueError("process_vision_info did not return a tuple")

    if len(vision_info) == 3:
        image_inputs, video_inputs, _ = vision_info
    elif len(vision_info) == 2:
        image_inputs, video_inputs = vision_info
    else:
        raise ValueError(
            f"Expected 2 or 3 values from process_vision_info, but got {len(vision_info)}."
        )

    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )

    target_device = "cuda" if torch.cuda.is_available() else "cpu"
    inputs = move_inputs_to_device(inputs, target_device)

    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_new_tokens=max_tokens)

    generated_ids_trimmed = [
        out_ids[len(in_ids) :]
        for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]

    caption = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0]

    if qwen35 and "<think>" in caption:
        caption = re.sub(r"<think>.*?</think>\s*", "", caption, flags=re.DOTALL)

    return caption.strip()


def generate_caption_llava(
    media_path,
    prompt,
    max_tokens,
    summary_mode=False,
    one_sentence_mode=False,
    resolution_mode="auto",
):
    global processor, model

    assert model is not None, "Model must be loaded before generating captions."
    assert processor is not None, "Processor must be loaded before generating captions."

    ext = os.path.splitext(media_path)[-1].lower()
    if ext in VIDEO_EXTENSIONS:
        raise ValueError(
            "JoyCaption/LLaVA currently supports images only, not video files."
        )

    image = Image.open(media_path).convert("RGB")

    if summary_mode and one_sentence_mode:
        prompt += " Give a one-sentence summary of the scene."
    elif summary_mode:
        prompt += " Give a short summary of the scene."
    elif one_sentence_mode:
        prompt += " Describe this image in one sentence."

    convo = [
        {
            "role": "system",
            "content": "You are a helpful image captioner.",
        },
        {
            "role": "user",
            "content": prompt,
        },
    ]

    convo_string = processor.apply_chat_template(
        convo,
        tokenize=False,
        add_generation_prompt=True,
    )
    assert isinstance(convo_string, str)

    inputs = processor(
        text=[convo_string],
        images=[image],
        return_tensors="pt",
    )

    target_device = "cuda" if torch.cuda.is_available() else "cpu"
    inputs = move_inputs_to_device(inputs, target_device)

    if "pixel_values" in inputs and torch.cuda.is_available():
        try:
            inputs["pixel_values"] = inputs["pixel_values"].to(torch.bfloat16)
        except Exception:
            pass

    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            do_sample=True,
            suppress_tokens=None,
            use_cache=True,
            temperature=0.6,
            top_k=None,
            top_p=0.9,
        )[0]

    generated_ids = generated_ids[inputs["input_ids"].shape[1] :]

    caption = processor.tokenizer.decode(
        generated_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )

    return caption.strip()


def generate_caption(
    media_path,
    prompt,
    max_tokens,
    summary_mode=False,
    one_sentence_mode=False,
    resolution_mode="auto",
):
    backend = get_model_backend(current_model_id)

    if backend == "joycaption_llava":
        return generate_caption_llava(
            media_path,
            prompt,
            max_tokens,
            summary_mode,
            one_sentence_mode,
            resolution_mode,
        )

    return generate_caption_qwen(
        media_path, prompt, max_tokens, summary_mode, one_sentence_mode, resolution_mode
    )


def generate_text_response_qwen(prompt_text: str, max_tokens: int = 1024):
    global processor, model, current_model_id

    assert model is not None, "Merge model must be loaded before text generation."
    assert processor is not None, (
        "Merge processor must be loaded before text generation."
    )

    if get_model_backend(current_model_id) != "qwen":
        raise ValueError("Merge model must use a Qwen backend in this app.")

    messages = [{"role": "user", "content": [{"type": "text", "text": prompt_text}]}]
    qwen35 = is_qwen35_model(current_model_id)

    if qwen35:
        messages.append({"role": "assistant", "content": "<think>\n\n</think>\n\n"})

    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=not qwen35,
        continue_final_message=qwen35,
    )

    inputs = processor(
        text=[text],
        padding=True,
        return_tensors="pt",
    )

    target_device = "cuda" if torch.cuda.is_available() else "cpu"
    inputs = move_inputs_to_device(inputs, target_device)

    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            do_sample=False,
            use_cache=True,
        )

    generated_ids_trimmed = [
        out_ids[len(in_ids) :]
        for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]

    output = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0].strip()

    if qwen35 and "<think>" in output:
        output = re.sub(r"<think>.*?</think>\s*", "", output, flags=re.DOTALL)

    return output.strip()


def is_image_file(filename):
    return filename.lower().endswith(IMAGE_EXTENSIONS)


def is_video_file(filename):
    return filename.lower().endswith(VIDEO_EXTENSIONS)


def build_final_prompt(user_prompt, summary, one_sentence):
    parts = [user_prompt.strip()]
    if summary:
        parts.append("Please provide a short summary.")
    if one_sentence:
        parts.append("Keep the description to one sentence.")
    return " ".join(parts)


def reset_prompt():
    return DEFAULT_PROMPT


def reset_merge_prompt():
    return DEFAULT_MERGE_PROMPT


def extract_model_id(selected_value: str, custom_value: str) -> str:
    if selected_value == "Custom..." and custom_value and custom_value.strip():
        return custom_value.strip()
    return selected_value


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def read_text_file(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read().strip()


def write_text_file(path: str, text: str):
    ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as f:
        f.write(text.strip())


def get_final_txt_path(media_path: str) -> str:
    txt_filename = os.path.splitext(os.path.basename(media_path))[0] + ".txt"
    return os.path.join(os.path.dirname(media_path), txt_filename)


def get_raw_output_dir(media_path: str) -> str:
    return os.path.join(os.path.dirname(media_path), RAW_SUBFOLDER_NAME)


def get_raw_caption_path(media_path: str, model_id: str, attn_impl: str) -> str:
    base = os.path.splitext(os.path.basename(media_path))[0]
    model_slug = sanitize_model_slug(model_id)
    attn_slug = get_attention_suffix(attn_impl)
    return os.path.join(
        get_raw_output_dir(media_path), f"{base}.{model_slug}.{attn_slug}.txt"
    )


def get_merged_caption_copy_path(media_path: str) -> str:
    base = os.path.splitext(os.path.basename(media_path))[0]
    return os.path.join(get_raw_output_dir(media_path), f"{base}.merged.txt")


def get_audit_path(media_path: str) -> str:
    base = os.path.splitext(os.path.basename(media_path))[0]
    return os.path.join(get_raw_output_dir(media_path), f"{base}.audit.txt")


def get_combo_key(model_id: str, attn_impl: str) -> str:
    return f"{sanitize_model_slug(model_id)}.{get_attention_suffix(attn_impl)}"


def format_combo_label(combo: dict) -> str:
    return f"{combo['model_id']} [{combo['attn_impl']}]"


def build_run_combinations(selected_models, selected_attns, quant_choice):
    combos = []
    for model_id in selected_models:
        for attn_impl in selected_attns:
            combos.append(
                {
                    "model_id": model_id,
                    "backend": get_model_backend(model_id),
                    "attn_impl": attn_impl,
                    "quant_choice": quant_choice,
                    "slug": sanitize_model_slug(model_id),
                    "combo_key": get_combo_key(model_id, attn_impl),
                }
            )
    return combos


def filter_valid_combinations(combos):
    valid = []
    invalid = []
    for combo in combos:
        model_id = combo["model_id"]
        attn_impl = combo["attn_impl"]
        backend = get_model_backend(model_id)

        if backend == "unsupported_gguf":
            invalid.append((combo, "GGUF unsupported"))
            continue

        if not supports_attention(model_id, attn_impl):
            invalid.append((combo, f"attention '{attn_impl}' unsupported"))
            continue

        valid.append(combo)
    return valid, invalid


def collect_existing_raw_captions(media_path: str, combos):
    raw_map = {}
    for combo in combos:
        raw_path = get_raw_caption_path(
            media_path, combo["model_id"], combo["attn_impl"]
        )
        if os.path.exists(raw_path):
            try:
                raw_map[combo["combo_key"]] = {
                    "caption": read_text_file(raw_path),
                    "path": raw_path,
                    "model_id": combo["model_id"],
                    "attn_impl": combo["attn_impl"],
                    "status": "reused",
                }
            except Exception:
                pass
    return raw_map


def cleanup_merge_output(text: str) -> str:
    text = text.strip().strip("`")
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        return ""

    joined = "\n".join(lines)
    joined = re.sub(
        r"^(final caption|merged caption|caption)\s*:\s*",
        "",
        joined,
        flags=re.IGNORECASE,
    )
    return joined.strip()


def select_fallback_caption(
    raw_entries: dict,
    preferred_model_id: str = DEFAULT_MODEL_ID,
    preferred_attn: str = DEFAULT_ATTN,
) -> str:
    if not raw_entries:
        return ""

    preferred_key = get_combo_key(preferred_model_id, preferred_attn)
    if (
        preferred_key in raw_entries
        and raw_entries[preferred_key].get("caption", "").strip()
    ):
        return raw_entries[preferred_key]["caption"].strip()

    longest = sorted(
        [v for v in raw_entries.values() if v.get("caption", "").strip()],
        key=lambda x: len(x["caption"]),
        reverse=True,
    )
    if longest:
        return longest[0]["caption"].strip()

    first_key = sorted(raw_entries.keys())[0]
    return raw_entries[first_key].get("caption", "").strip()


def build_merge_input_text(
    media_path: str, user_prompt: str, merge_prompt: str, raw_entries: dict
):
    source_blocks = []
    for idx, key in enumerate(sorted(raw_entries.keys()), start=1):
        item = raw_entries[key]
        source_blocks.append(
            f"[Source {idx}]\n"
            f"Model: {item['model_id']}\n"
            f"Attention: {item['attn_impl']}\n"
            f"Caption: {item['caption']}\n"
        )

    joined_sources = "\n".join(source_blocks).strip()
    media_name = os.path.basename(media_path)

    return (
        f"{merge_prompt.strip()}\n\n"
        f"Image file: {media_name}\n"
        f"Original user prompt:\n{user_prompt.strip()}\n\n"
        f"Source captions:\n{joined_sources}\n\n"
        f"Return only the final merged caption."
    )


def write_audit_file(
    media_path: str,
    user_prompt: str,
    merge_model_id: str,
    merge_prompt: str,
    attempted_combos,
    successful_entries: dict,
    failed_combo_messages,
    merged_caption: str,
    rationale: str = "",
    fallback_used: bool = False,
):
    lines = []
    lines.append(f"Image: {media_path}")
    lines.append(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("")
    lines.append("Raw generation prompt:")
    lines.append(user_prompt.strip())
    lines.append("")
    lines.append(f"Merge prompt version: {MERGE_PROMPT_VERSION}")
    lines.append(f"Merge model: {merge_model_id}")
    lines.append(f"Fallback used: {fallback_used}")
    lines.append("")
    lines.append("Attempted combinations:")
    for combo in attempted_combos:
        lines.append(f"- {format_combo_label(combo)}")
    lines.append("")
    lines.append("Successful combinations:")
    for key in sorted(successful_entries.keys()):
        item = successful_entries[key]
        lines.append(
            f"- {item['model_id']} [{item['attn_impl']}] ({item.get('status', 'generated')})"
        )
    lines.append("")
    lines.append("Failed combinations:")
    if failed_combo_messages:
        for msg in failed_combo_messages:
            lines.append(f"- {msg}")
    else:
        lines.append("- None")
    lines.append("")
    lines.append("Source captions:")
    for key in sorted(successful_entries.keys()):
        item = successful_entries[key]
        lines.append("")
        lines.append(f"[{item['model_id']} | {item['attn_impl']}]")
        lines.append(item["caption"])
    lines.append("")
    lines.append("Merged caption:")
    lines.append(merged_caption.strip())
    if rationale.strip():
        lines.append("")
        lines.append("Rationale:")
        lines.append(rationale.strip())

    write_text_file(get_audit_path(media_path), "\n".join(lines))


def resolve_selected_models(
    run_mode,
    multi_model,
    selected_single_model,
    selected_multi_models,
    custom_model_box_value,
):
    if run_mode == "Single":
        return [extract_model_id(selected_single_model, custom_model_box_value)]

    if multi_model:
        models = [m for m in selected_multi_models if m and m != "Custom..."]
        if not models:
            fallback = extract_model_id(selected_single_model, custom_model_box_value)
            return [fallback]
        return models

    return [extract_model_id(selected_single_model, custom_model_box_value)]


def resolve_selected_attns(
    run_mode, multi_atten, selected_single_attn, selected_multi_attns
):
    if run_mode == "Single":
        return [selected_single_attn]

    if multi_atten:
        attns = [a for a in selected_multi_attns if a in ATTN_CHOICES]
        return attns or [selected_single_attn]

    return [selected_single_attn]


def scan_media_files(folder_path: str):
    media_files = []
    for root, _, files in os.walk(folder_path):
        for file in files:
            if is_image_file(file) or is_video_file(file):
                media_files.append(os.path.join(root, file))
    media_files.sort()
    return media_files


def elapsed_string(start_time: float):
    elapsed = int(time.time() - start_time)
    return f"{elapsed // 60:02d}:{elapsed % 60:02d}"


def pack_process_output(
    status, image, image_md, caption, progress, elapsed, preflight_text, control_updates
):
    return (
        status,
        image,
        image_md,
        caption,
        progress,
        elapsed,
        preflight_text,
        *control_updates,
    )


def maybe_load_for_combo(combo, dual_load_enabled):
    if dual_load_enabled:
        return load_selected_model_cached(
            combo["model_id"],
            combo["quant_choice"],
            combo["attn_impl"],
            max_cached_models=2,
        )
    return load_selected_model(
        combo["model_id"],
        combo["quant_choice"],
        combo["attn_impl"],
        allow_attn_fallback=False,
    )


def choose_merge_attention(selected_attns):
    if FLASH_ATTN_AVAILABLE and "flash_attention_2" in selected_attns:
        return "flash_attention_2"
    if selected_attns:
        return selected_attns[0]
    return DEFAULT_ATTN


def maybe_load_merge_model(
    merge_model_id, quant_choice, merge_attn_impl, dual_load_enabled
):
    if dual_load_enabled:
        return load_selected_model_cached(
            merge_model_id,
            quant_choice,
            merge_attn_impl,
            max_cached_models=2,
        )
    return load_selected_model(
        merge_model_id,
        quant_choice,
        merge_attn_impl,
        allow_attn_fallback=True,
    )


def build_preflight_summary(
    folder_path,
    media_files,
    requested_models,
    requested_attns,
    valid_combos,
    invalid_combos,
    enable_merge,
    merge_model_id,
    skip_existing,
    save_raw,
    save_audit,
    reuse_raw,
    overwrite_raw,
    dual_load_enabled,
):
    lines = []
    lines.append("=== Preflight Summary ===")
    lines.append(f"Folder: {folder_path}")
    lines.append(f"Media files found: {len(media_files)}")
    lines.append("")
    lines.append("Selected models:")
    for m in requested_models:
        lines.append(f"- {m}")
    lines.append("")
    lines.append("Selected attention implementations:")
    for a in requested_attns:
        lines.append(f"- {a}")
    lines.append("")
    lines.append(f"Valid combinations: {len(valid_combos)}")
    for combo in valid_combos:
        lines.append(f"- {format_combo_label(combo)}")
    lines.append("")
    lines.append(f"Invalid combinations filtered out: {len(invalid_combos)}")
    if invalid_combos:
        for combo, reason in invalid_combos:
            lines.append(f"- {format_combo_label(combo)} -> {reason}")
    else:
        lines.append("- None")
    lines.append("")
    lines.append(f"Merge enabled: {enable_merge}")
    lines.append(f"Merge model: {merge_model_id if enable_merge else 'N/A'}")
    lines.append(f"Save raw captions: {save_raw}")
    lines.append(f"Save audit file: {save_audit}")
    lines.append(f"Skip existing final .txt: {skip_existing}")
    lines.append(f"Reuse existing raw files: {reuse_raw}")
    lines.append(f"Overwrite raw files: {overwrite_raw}")
    lines.append(f"Advanced dual-load enabled: {dual_load_enabled}")
    if dual_load_enabled:
        lines.append(
            "Note: if dual-load causes a load failure, the run falls back to sequential mode."
        )
    return "\n".join(lines)


def process_folder_single(
    folder_path,
    prompt,
    skip_existing,
    max_tokens,
    summary_mode,
    one_sentence_mode,
    retain_preview,
    resolution_mode,
):
    global should_abort, current_model_id, current_quant, current_backend

    processed_media = 0
    skipped_media = 0
    failed_media = 0
    last_media_to_show = None
    last_caption = ""
    last_media_name_markdown = ""
    preflight_text = "Single mode: one model, one attention, one caption per media."
    start_time = time.time()

    print("[DEBUG] starting folder processing...")
    print("[DEBUG] using model: ", current_model_id)
    print("[DEBUG] backend: ", current_backend)
    print("[DEBUG] model quantization: ", current_quant)
    print("[DEBUG] folder_path: ", folder_path)
    print("[DEBUG] prompt: ", prompt)
    print("[DEBUG] skip_existing: ", skip_existing)
    print("[DEBUG] max_tokens: ", max_tokens)
    print("[DEBUG] summary_mode: ", summary_mode)
    print("[DEBUG] one_sentence_mode: ", one_sentence_mode)
    print("[DEBUG] retain_preview: ", retain_preview)
    print("[DEBUG] resolution_mode: ", resolution_mode)
    print("[DEBUG] should_abort: ", should_abort)

    if model is None or processor is None:
        control_updates = finish_process()
        yield pack_process_output(
            "⚠️ Load a model first.",
            None,
            None,
            "No model loaded.",
            0,
            "",
            preflight_text,
            control_updates,
        )
        return

    if not folder_path.strip():
        control_updates = finish_process()
        yield pack_process_output(
            "⚠️ Please enter a valid folder path.",
            None,
            None,
            "No media to process.",
            0,
            "",
            preflight_text,
            control_updates,
        )
        return

    if not os.path.exists(folder_path):
        control_updates = finish_process()
        yield pack_process_output(
            f"❌ Folder not found: {folder_path}",
            None,
            None,
            "No media to process.",
            0,
            "",
            preflight_text,
            control_updates,
        )
        return

    media_files = scan_media_files(folder_path)
    total_media = len(media_files)

    if total_media == 0:
        control_updates = finish_process()
        yield pack_process_output(
            "📂 No media found in the folder or subfolders.",
            None,
            None,
            "No media to process.",
            0,
            "",
            preflight_text,
            control_updates,
        )
        return

    for idx, media_path in enumerate(media_files):
        if should_abort:
            should_abort = False
            control_updates = enable_controls_dict()
            control_updates[control_keys.index("abort_button")] = gr.update(
                interactive=False
            )
            yield pack_process_output(
                "⛔ Aborted by user.",
                None,
                None,
                "Aborted.",
                0,
                elapsed_string(start_time),
                preflight_text,
                control_updates,
            )
            return

        try:
            txt_path = get_final_txt_path(media_path)
            rel_path = os.path.relpath(media_path, folder_path)
            media_name_markdown = f"**File:** `{rel_path}`"

            if skip_existing and os.path.exists(txt_path):
                skipped_media += 1
                progress = int(((idx + 1) / total_media) * 100)
                control_updates = start_process()
                yield pack_process_output(
                    f"⏭️ Skipped {idx + 1}/{total_media}: {rel_path} (already captioned)",
                    last_media_to_show if retain_preview else None,
                    last_media_name_markdown if retain_preview else None,
                    last_caption if retain_preview else "Skipped (already captioned)",
                    progress,
                    elapsed_string(start_time),
                    preflight_text,
                    control_updates,
                )
                continue

            caption = generate_caption(
                media_path,
                prompt,
                max_tokens,
                summary_mode,
                one_sentence_mode,
                resolution_mode,
            )

            if is_image_file(media_path):
                media_to_show = Image.open(media_path)
            else:
                media_to_show = None

            write_text_file(txt_path, caption)

            progress = int(((idx + 1) / total_media) * 100)
            last_media_to_show = media_to_show
            last_caption = caption
            last_media_name_markdown = media_name_markdown
            processed_media += 1

            control_updates = start_process()
            yield pack_process_output(
                f"🖼️ Processing {idx + 1}/{total_media}: {rel_path}",
                media_to_show,
                media_name_markdown,
                caption,
                progress,
                elapsed_string(start_time),
                preflight_text,
                control_updates,
            )

        except Exception as e:
            failed_media += 1
            print(f"[ERROR] Failed processing file {media_path}: {e}")
            control_updates = start_process()
            yield pack_process_output(
                f"⚠️ Error processing {media_path}: {str(e)}",
                None,
                None,
                "Error in captioning.",
                int(((idx + 1) / total_media) * 100),
                elapsed_string(start_time),
                preflight_text,
                control_updates,
            )

    control_updates = finish_process()
    yield pack_process_output(
        f"✅ Processing complete! processed {processed_media} media in {elapsed_string(start_time)}, skipped {skipped_media} media. Failed to process {failed_media} media.",
        last_media_to_show,
        last_media_name_markdown,
        last_caption,
        100,
        elapsed_string(start_time),
        preflight_text,
        control_updates,
    )


def process_folder_multi(
    folder_path,
    prompt,
    skip_existing,
    max_tokens,
    summary_mode,
    one_sentence_mode,
    retain_preview,
    resolution_mode,
    selected_models,
    selected_attns,
    quant_choice,
    enable_merge,
    merge_model_id,
    merge_prompt,
    save_raw_captions,
    save_audit,
    overwrite_raw,
    reuse_raw,
    advanced_dual_load,
):
    global should_abort

    processed_media = 0
    skipped_media = 0
    failed_media = 0
    partial_media = 0
    raw_failures = 0
    merge_failures = 0
    last_media_to_show = None
    last_caption = ""
    last_media_name_markdown = ""

    if not folder_path.strip():
        control_updates = finish_process()
        yield pack_process_output(
            "⚠️ Please enter a valid folder path.",
            None,
            None,
            "No media to process.",
            0,
            "",
            "",
            control_updates,
        )
        return

    if not os.path.exists(folder_path):
        control_updates = finish_process()
        yield pack_process_output(
            f"❌ Folder not found: {folder_path}",
            None,
            None,
            "No media to process.",
            0,
            "",
            "",
            control_updates,
        )
        return

    media_files = scan_media_files(folder_path)
    if not media_files:
        control_updates = finish_process()
        yield pack_process_output(
            "📂 No media found in the folder or subfolders.",
            None,
            None,
            "No media to process.",
            0,
            "",
            "",
            control_updates,
        )
        return

    requested_combos = build_run_combinations(
        selected_models, selected_attns, quant_choice
    )
    valid_combos, invalid_combos = filter_valid_combinations(requested_combos)

    if not valid_combos:
        control_updates = finish_process()
        preflight_text = build_preflight_summary(
            folder_path,
            media_files,
            selected_models,
            selected_attns,
            valid_combos,
            invalid_combos,
            enable_merge,
            merge_model_id,
            skip_existing,
            save_raw_captions,
            save_audit,
            reuse_raw,
            overwrite_raw,
            advanced_dual_load,
        )
        yield pack_process_output(
            "❌ No valid model/attention combinations remain after filtering.",
            None,
            None,
            "No valid combinations.",
            0,
            "",
            preflight_text,
            control_updates,
        )
        return

    preflight_text = build_preflight_summary(
        folder_path,
        media_files,
        selected_models,
        selected_attns,
        valid_combos,
        invalid_combos,
        enable_merge,
        merge_model_id,
        skip_existing,
        save_raw_captions,
        save_audit,
        reuse_raw,
        overwrite_raw,
        advanced_dual_load,
    )

    start_time = time.time()
    target_media = []
    for media_path in media_files:
        if skip_existing and os.path.exists(get_final_txt_path(media_path)):
            skipped_media += 1
        else:
            target_media.append(media_path)

    total_steps = max(1, (len(target_media) * len(valid_combos)) + len(target_media))
    completed_steps = 0

    control_updates = start_process()
    yield pack_process_output(
        f"📋 Preflight complete. {len(target_media)} target media, {len(valid_combos)} valid combinations.",
        None,
        None,
        "Preparing multi-pass run...",
        0,
        elapsed_string(start_time),
        preflight_text,
        control_updates,
    )

    dual_load_active = advanced_dual_load

    for combo_index, combo in enumerate(valid_combos, start=1):
        if should_abort:
            should_abort = False
            control_updates = enable_controls_dict()
            control_updates[control_keys.index("abort_button")] = gr.update(
                interactive=False
            )
            yield pack_process_output(
                "⛔ Aborted by user.",
                None,
                None,
                "Aborted.",
                int((completed_steps / total_steps) * 100),
                elapsed_string(start_time),
                preflight_text,
                control_updates,
            )
            return

        try:
            maybe_load_for_combo(combo, dual_load_active)
        except Exception as e:
            if dual_load_active:
                dual_load_active = False
                print(f"[WARN] Dual-load failed, falling back to sequential mode: {e}")
                try:
                    unload_model(clear_cache=True)
                    load_selected_model(
                        combo["model_id"],
                        combo["quant_choice"],
                        combo["attn_impl"],
                        allow_attn_fallback=False,
                    )
                except Exception as e2:
                    raw_failures += len(target_media)
                    control_updates = start_process()
                    yield pack_process_output(
                        f"⚠️ Failed loading combination {combo_index}/{len(valid_combos)}: {format_combo_label(combo)} -> {str(e2)}",
                        None,
                        None,
                        "Combination load failed.",
                        int((completed_steps / total_steps) * 100),
                        elapsed_string(start_time),
                        preflight_text,
                        control_updates,
                    )
                    continue
            else:
                raw_failures += len(target_media)
                control_updates = start_process()
                yield pack_process_output(
                    f"⚠️ Failed loading combination {combo_index}/{len(valid_combos)}: {format_combo_label(combo)} -> {str(e)}",
                    None,
                    None,
                    "Combination load failed.",
                    int((completed_steps / total_steps) * 100),
                    elapsed_string(start_time),
                    preflight_text,
                    control_updates,
                )
                continue

        for media_idx, media_path in enumerate(target_media, start=1):
            if should_abort:
                should_abort = False
                control_updates = enable_controls_dict()
                control_updates[control_keys.index("abort_button")] = gr.update(
                    interactive=False
                )
                yield pack_process_output(
                    "⛔ Aborted by user.",
                    None,
                    None,
                    "Aborted.",
                    int((completed_steps / total_steps) * 100),
                    elapsed_string(start_time),
                    preflight_text,
                    control_updates,
                )
                return

            rel_path = os.path.relpath(media_path, folder_path)
            media_name_markdown = f"**File:** `{rel_path}`"
            raw_path = get_raw_caption_path(
                media_path, combo["model_id"], combo["attn_impl"]
            )
            ensure_dir(get_raw_output_dir(media_path))

            try:
                if reuse_raw and (not overwrite_raw) and os.path.exists(raw_path):
                    caption = read_text_file(raw_path)
                    status_msg = (
                        f"♻️ Reused raw {combo_index}/{len(valid_combos)} for {rel_path} "
                        f"using {format_combo_label(combo)}"
                    )
                else:
                    caption = generate_caption(
                        media_path,
                        prompt,
                        max_tokens,
                        summary_mode,
                        one_sentence_mode,
                        resolution_mode,
                    )
                    if save_raw_captions:
                        write_text_file(raw_path, caption)
                    status_msg = (
                        f"🖼️ Raw {combo_index}/{len(valid_combos)} for {rel_path} "
                        f"using {format_combo_label(combo)}"
                    )

                media_to_show = (
                    Image.open(media_path) if is_image_file(media_path) else None
                )
                last_media_to_show = media_to_show
                last_caption = caption
                last_media_name_markdown = media_name_markdown

                completed_steps += 1
                control_updates = start_process()
                yield pack_process_output(
                    status_msg,
                    media_to_show,
                    media_name_markdown,
                    caption,
                    int((completed_steps / total_steps) * 100),
                    elapsed_string(start_time),
                    preflight_text,
                    control_updates,
                )

            except Exception as e:
                raw_failures += 1
                completed_steps += 1
                print(
                    f"[ERROR] Failed raw caption for {media_path} [{format_combo_label(combo)}]: {e}"
                )
                control_updates = start_process()
                yield pack_process_output(
                    f"⚠️ Raw caption failed for {rel_path} using {format_combo_label(combo)} -> {str(e)}",
                    None,
                    None,
                    "Raw caption failed.",
                    int((completed_steps / total_steps) * 100),
                    elapsed_string(start_time),
                    preflight_text,
                    control_updates,
                )

    merge_model_loaded = False
    merge_attn_impl = choose_merge_attention(selected_attns)

    if enable_merge:
        try:
            maybe_load_merge_model(
                merge_model_id,
                quant_choice,
                merge_attn_impl,
                dual_load_active,
            )
            merge_model_loaded = True
        except Exception as e:
            merge_failures += 1
            print(
                f"[WARN] Failed loading merge model {merge_model_id} with attention {merge_attn_impl}: {e}"
            )
            merge_model_loaded = False

    for media_path in target_media:
        if should_abort:
            should_abort = False
            control_updates = enable_controls_dict()
            control_updates[control_keys.index("abort_button")] = gr.update(
                interactive=False
            )
            yield pack_process_output(
                "⛔ Aborted by user.",
                None,
                None,
                "Aborted.",
                int((completed_steps / total_steps) * 100),
                elapsed_string(start_time),
                preflight_text,
                control_updates,
            )
            return

        rel_path = os.path.relpath(media_path, folder_path)
        media_name_markdown = f"**File:** `{rel_path}`"
        final_txt_path = get_final_txt_path(media_path)
        merged_copy_path = get_merged_caption_copy_path(media_path)

        raw_entries = collect_existing_raw_captions(media_path, valid_combos)
        failed_combo_messages = []
        for combo in valid_combos:
            if combo["combo_key"] not in raw_entries:
                failed_combo_messages.append(
                    f"{format_combo_label(combo)} -> no raw caption file"
                )

        final_caption = ""
        fallback_used = False

        try:
            if len(raw_entries) == 0:
                failed_media += 1
                completed_steps += 1
                if save_audit:
                    write_audit_file(
                        media_path=media_path,
                        user_prompt=prompt,
                        merge_model_id=merge_model_id,
                        merge_prompt=merge_prompt,
                        attempted_combos=valid_combos,
                        successful_entries=raw_entries,
                        failed_combo_messages=failed_combo_messages,
                        merged_caption="",
                        fallback_used=True,
                    )
                control_updates = start_process()
                yield pack_process_output(
                    f"⚠️ No successful raw captions available for {rel_path}.",
                    None,
                    None,
                    "No final caption created.",
                    int((completed_steps / total_steps) * 100),
                    elapsed_string(start_time),
                    preflight_text,
                    control_updates,
                )
                continue

            if len(raw_entries) == 1 or not enable_merge or not merge_model_loaded:
                final_caption = select_fallback_caption(
                    raw_entries, DEFAULT_MODEL_ID, DEFAULT_ATTN
                )
                fallback_used = True
            else:
                merge_input = build_merge_input_text(
                    media_path, prompt, merge_prompt, raw_entries
                )
                merged_text = generate_text_response_qwen(
                    merge_input, max_tokens=min(max_tokens, 1024)
                )
                merged_text = cleanup_merge_output(merged_text)
                if not merged_text:
                    raise ValueError("Merge model returned empty text")
                final_caption = merged_text

            if not final_caption.strip():
                final_caption = select_fallback_caption(
                    raw_entries, DEFAULT_MODEL_ID, DEFAULT_ATTN
                )
                fallback_used = True

            write_text_file(final_txt_path, final_caption)
            write_text_file(merged_copy_path, final_caption)

            if save_audit:
                write_audit_file(
                    media_path=media_path,
                    user_prompt=prompt,
                    merge_model_id=merge_model_id,
                    merge_prompt=merge_prompt,
                    attempted_combos=valid_combos,
                    successful_entries=raw_entries,
                    failed_combo_messages=failed_combo_messages,
                    merged_caption=final_caption,
                    fallback_used=fallback_used,
                )

            media_to_show = (
                Image.open(media_path) if is_image_file(media_path) else None
            )
            last_media_to_show = media_to_show
            last_caption = final_caption
            last_media_name_markdown = media_name_markdown

            if fallback_used and len(raw_entries) > 1:
                partial_media += 1
            else:
                processed_media += 1

            completed_steps += 1
            control_updates = start_process()
            status_msg = f"✅ Finalized {rel_path}"
            if fallback_used:
                status_msg += " (fallback)"
            yield pack_process_output(
                status_msg,
                media_to_show,
                media_name_markdown,
                final_caption,
                int((completed_steps / total_steps) * 100),
                elapsed_string(start_time),
                preflight_text,
                control_updates,
            )

        except Exception as e:
            merge_failures += 1
            try:
                final_caption = select_fallback_caption(
                    raw_entries, DEFAULT_MODEL_ID, DEFAULT_ATTN
                )
                fallback_used = True
                if final_caption.strip():
                    write_text_file(final_txt_path, final_caption)
                    write_text_file(merged_copy_path, final_caption)
                    partial_media += 1
                    if save_audit:
                        write_audit_file(
                            media_path=media_path,
                            user_prompt=prompt,
                            merge_model_id=merge_model_id,
                            merge_prompt=merge_prompt,
                            attempted_combos=valid_combos,
                            successful_entries=raw_entries,
                            failed_combo_messages=failed_combo_messages
                            + [f"Merge failure -> {str(e)}"],
                            merged_caption=final_caption,
                            fallback_used=True,
                        )
                else:
                    failed_media += 1
            except Exception:
                failed_media += 1

            completed_steps += 1
            control_updates = start_process()
            yield pack_process_output(
                f"⚠️ Merge/finalization failed for {rel_path}: {str(e)}",
                None,
                None,
                final_caption if final_caption else "Finalization failed.",
                int((completed_steps / total_steps) * 100),
                elapsed_string(start_time),
                preflight_text,
                control_updates,
            )

    control_updates = finish_process()
    yield pack_process_output(
        f"✅ Multi-pass complete! finalized {processed_media} media, partial/fallback {partial_media}, skipped {skipped_media}, failed {failed_media}, raw failures {raw_failures}, merge failures {merge_failures}.",
        last_media_to_show,
        last_media_name_markdown,
        last_caption,
        100,
        elapsed_string(start_time),
        preflight_text,
        control_updates,
    )


def process_folder_dispatch(
    folder_path,
    prompt,
    skip_existing,
    max_tokens,
    summary_mode,
    one_sentence_mode,
    retain_preview,
    resolution_mode,
    run_mode,
    multi_model,
    multi_atten,
    model_dropdown_value,
    model_multiselect_values,
    custom_model_value,
    attn_dropdown_value,
    attn_multiselect_values,
    quant_choice,
    enable_merge,
    merge_model_id,
    merge_prompt,
    save_raw_captions,
    save_audit,
    raw_file_handling,
    advanced_dual_load,
):
    selected_models = resolve_selected_models(
        run_mode,
        multi_model,
        model_dropdown_value,
        model_multiselect_values,
        custom_model_value,
    )
    selected_attns = resolve_selected_attns(
        run_mode,
        multi_atten,
        attn_dropdown_value,
        attn_multiselect_values,
    )

    overwrite_raw = raw_file_handling == "Overwrite raw files"
    reuse_raw = raw_file_handling == "Reuse existing raw files"

    if run_mode == "Single":
        yield from process_folder_single(
            folder_path,
            prompt,
            skip_existing,
            max_tokens,
            summary_mode,
            one_sentence_mode,
            retain_preview,
            resolution_mode,
        )
        return

    yield from process_folder_multi(
        folder_path=folder_path,
        prompt=prompt,
        skip_existing=skip_existing,
        max_tokens=max_tokens,
        summary_mode=summary_mode,
        one_sentence_mode=one_sentence_mode,
        retain_preview=retain_preview,
        resolution_mode=resolution_mode,
        selected_models=selected_models,
        selected_attns=selected_attns,
        quant_choice=quant_choice,
        enable_merge=enable_merge,
        merge_model_id=merge_model_id,
        merge_prompt=merge_prompt,
        save_raw_captions=save_raw_captions,
        save_audit=save_audit,
        overwrite_raw=overwrite_raw,
        reuse_raw=reuse_raw,
        advanced_dual_load=advanced_dual_load,
    )


css = """
.generating {
    border: none;
}
.small-note {
    font-size: 0.9rem;
    opacity: 0.9;
}
"""


def _toggle_custom(choice):
    return gr.update(visible=(choice == "Custom..."))


def _build_defaults_ui_payload(settings, current_prompt, status_message):
    settings = sanitize_saved_settings(settings)

    model_choices = AVAILABLE_MODELS
    model_multi_choices = [m for m in AVAILABLE_MODELS if m != "Custom..."]
    merge_choices = MERGE_MODEL_CHOICES

    model_value = (
        settings["model_dropdown"]
        if settings["model_dropdown"] in model_choices
        else DEFAULT_MODEL_ID
    )
    model_multi_value = [
        m for m in settings["model_multiselect"] if m in model_multi_choices
    ]
    if not model_multi_value and model_value != "Custom...":
        model_multi_value = [model_value]

    merge_value = (
        settings["merge_model_dropdown"]
        if settings["merge_model_dropdown"] in merge_choices
        else DEFAULT_MERGE_MODEL_ID
    )

    prompt_preview_value = build_final_prompt(
        current_prompt, settings["summary_mode"], settings["one_sentence_mode"]
    )

    return (
        status_message,
        gr.update(choices=model_choices, value=model_value),
        gr.update(choices=model_multi_choices, value=model_multi_value),
        gr.update(value=settings["custom_model_box"]),
        gr.update(choices=merge_choices, value=merge_value),
        gr.update(value=settings["quant_dropdown"]),
        gr.update(choices=ATTN_CHOICES, value=settings["attn_dropdown"]),
        gr.update(choices=ATTN_CHOICES, value=settings["attn_multiselect"]),
        gr.update(value=settings["run_mode_radio"]),
        gr.update(value=settings["multi_model_checkbox"]),
        gr.update(value=settings["multi_atten_checkbox"]),
        gr.update(value=settings["advanced_dual_load_checkbox"]),
        gr.update(value=settings["enable_merge_checkbox"]),
        gr.update(value=settings["show_merge_prompt_checkbox"]),
        gr.update(value=settings["save_raw_captions_checkbox"]),
        gr.update(value=settings["save_audit_checkbox"]),
        gr.update(value=settings["skip_existing_checkbox"]),
        gr.update(value=settings["retain_preview_checkbox"]),
        gr.update(value=settings["summary_mode"]),
        gr.update(value=settings["one_sentence_mode"]),
        gr.update(value=settings["resolution_mode"]),
        gr.update(value=settings["raw_file_handling_radio"]),
        gr.update(value=settings["max_tokens_slider"]),
        gr.update(value=prompt_preview_value),
    )


def ui_download_hf_model(repo_input, current_multi_models, current_merge_model):
    try:
        repo_id = extract_hf_repo_id(repo_input)
        if is_gguf_model(repo_id):
            raise ValueError("GGUF repos are not supported in this app.")
        snapshot_download(repo_id=repo_id, repo_type="model", resume_download=True)
        add_model_choice(repo_id)

        multi_value = [
            m
            for m in (current_multi_models or [])
            if m in AVAILABLE_MODELS and m != "Custom..."
        ]
        if repo_id not in multi_value and repo_id != "Custom...":
            multi_value.append(repo_id)

        merge_value = (
            current_merge_model
            if current_merge_model in MERGE_MODEL_CHOICES
            else DEFAULT_MERGE_MODEL_ID
        )

        return (
            f"✅ Downloaded and added {repo_id}.",
            gr.update(value=""),
            gr.update(choices=AVAILABLE_MODELS, value=repo_id),
            gr.update(
                choices=[m for m in AVAILABLE_MODELS if m != "Custom..."],
                value=multi_value or [repo_id],
            ),
            gr.update(choices=MERGE_MODEL_CHOICES, value=merge_value),
        )
    except Exception as e:
        return (
            f"❌ Download failed: {str(e)}",
            gr.update(),
            gr.update(choices=AVAILABLE_MODELS),
            gr.update(choices=[m for m in AVAILABLE_MODELS if m != "Custom..."]),
            gr.update(choices=MERGE_MODEL_CHOICES),
        )


def ui_save_current_as_defaults(
    model_value,
    model_multiselect_values,
    custom_model_value,
    merge_model_value,
    quant_value,
    attn_value,
    attn_multiselect_values,
    run_mode_value,
    multi_model_value,
    multi_atten_value,
    advanced_dual_load_value,
    enable_merge_value,
    show_merge_prompt_value,
    save_raw_value,
    save_audit_value,
    skip_existing_value,
    retain_preview_value,
    summary_value,
    one_sentence_value,
    resolution_value,
    raw_file_handling_value,
    max_tokens_value,
):
    settings = {
        "model_dropdown": model_value,
        "model_multiselect": model_multiselect_values,
        "custom_model_box": custom_model_value,
        "merge_model_dropdown": merge_model_value,
        "quant_dropdown": quant_value,
        "attn_dropdown": attn_value,
        "attn_multiselect": attn_multiselect_values,
        "run_mode_radio": run_mode_value,
        "multi_model_checkbox": multi_model_value,
        "multi_atten_checkbox": multi_atten_value,
        "advanced_dual_load_checkbox": advanced_dual_load_value,
        "enable_merge_checkbox": enable_merge_value,
        "show_merge_prompt_checkbox": show_merge_prompt_value,
        "save_raw_captions_checkbox": save_raw_value,
        "save_audit_checkbox": save_audit_value,
        "skip_existing_checkbox": skip_existing_value,
        "retain_preview_checkbox": retain_preview_value,
        "summary_mode": summary_value,
        "one_sentence_mode": one_sentence_value,
        "resolution_mode": resolution_value,
        "raw_file_handling_radio": raw_file_handling_value,
        "max_tokens_slider": max_tokens_value,
    }
    save_user_defaults_file(settings)
    return f"✅ Saved current UI values as your defaults: {os.path.basename(USER_DEFAULTS_PATH)}"


def ui_load_my_defaults(current_prompt):
    settings = load_user_defaults()
    return _build_defaults_ui_payload(
        settings,
        current_prompt,
        f"✅ Loaded your defaults from {os.path.basename(USER_DEFAULTS_PATH)}",
    )


def ui_unload_model():
    unload_model(clear_cache=True)
    name, device, vram, dtype, cfg = get_model_info()
    status = "🧹 Model unloaded and CUDA cache release requested."
    return status, name, device, vram, dtype, cfg


def ui_load_model(sel, custom_id, quant, attn):
    model_id = (
        custom_id.strip()
        if sel == "Custom..." and custom_id and custom_id.strip()
        else sel
    )

    if attn not in ATTN_CHOICES:
        attn = DEFAULT_ATTN

    backend = get_model_backend(model_id)
    name, device, vram, dtype, cfg = load_selected_model(model_id, quant, attn)
    status = f"✅ Loaded {model_id} with {quant} quantization, attention {current_attn_impl}, backend {backend}."
    print("[DEBUG]", status)
    return status, name, device, vram, dtype, cfg


def ui_reset_merge_prompt():
    return DEFAULT_MERGE_PROMPT


# Override control handling so status_output is not duplicated in callback outputs
control_keys = [
    "model_dropdown",
    "quant_dropdown",
    "attn_dropdown",
    "load_button",
    "unload_button",
    "download_model_button",
    "save_defaults_button",
    "load_defaults_button",
    "reset_button",
    "start_button",
    "abort_button",
    "folder_input",
    "prompt_input",
    "skip_existing_checkbox",
    "max_tokens_slider",
    "summary_mode",
    "one_sentence_mode",
    "retain_preview_checkbox",
    "resolution_mode",
    "run_mode_radio",
    "multi_model_checkbox",
    "multi_atten_checkbox",
    "advanced_dual_load_checkbox",
    "model_multiselect",
    "attn_multiselect",
    "enable_merge_checkbox",
    "merge_model_dropdown",
    "merge_prompt_box",
    "show_merge_prompt_checkbox",
    "reset_merge_prompt_button",
    "save_raw_captions_checkbox",
    "save_audit_checkbox",
    "raw_file_handling_radio",
    "hf_model_url_input",
]


def toggle_controls(disabled=True):
    updates = {}
    for name in control_keys:
        updates[name] = gr.update(interactive=not disabled)
    return updates


def disable_controls_dict():
    return [toggle_controls(disabled=True)[k] for k in control_keys]


def enable_controls_dict():
    return [toggle_controls(disabled=False)[k] for k in control_keys]


def finish_process():
    updates = enable_controls_dict()
    abort_index = control_keys.index("abort_button")
    updates[abort_index] = gr.update(interactive=False)
    return updates


def abort_process():
    global should_abort
    should_abort = True
    updates = enable_controls_dict()
    abort_index = control_keys.index("abort_button")
    updates[abort_index] = gr.update(interactive=False)
    return (gr.update(value="⛔ Aborting process..."), *updates)


def start_process():
    updates = disable_controls_dict()
    abort_index = control_keys.index("abort_button")
    updates[abort_index] = gr.update(interactive=True)
    return updates


def _update_mode_visibility(
    run_mode, multi_model, multi_atten, enable_merge, show_merge_prompt
):
    single_mode = run_mode == "Single"
    multi_mode = not single_mode

    forced_multi_model = False if single_mode else multi_model
    forced_multi_atten = False if single_mode else multi_atten

    show_single_model = single_mode or (multi_mode and not forced_multi_model)
    show_multi_model = multi_mode and forced_multi_model

    show_single_attn = single_mode or (multi_mode and not forced_multi_atten)
    show_multi_attn = multi_mode and forced_multi_atten

    merge_checkbox_value = False if single_mode else enable_merge
    show_merge_controls = multi_mode and merge_checkbox_value
    show_merge_prompt_box = show_merge_controls and show_merge_prompt

    return (
        gr.update(value=forced_multi_model, interactive=multi_mode),
        gr.update(value=forced_multi_atten, interactive=multi_mode),
        gr.update(visible=show_single_model),
        gr.update(visible=show_multi_model),
        gr.update(visible=show_single_attn),
        gr.update(visible=show_multi_attn),
        gr.update(value=merge_checkbox_value, visible=multi_mode),
        gr.update(visible=show_merge_controls),
        gr.update(visible=show_merge_controls),
        gr.update(visible=show_merge_controls),
        gr.update(visible=show_merge_prompt_box),
        gr.update(visible=show_merge_controls),
        gr.update(visible=show_merge_controls),
        gr.update(visible=show_merge_controls),
        gr.update(visible=show_merge_controls),
        gr.update(visible=show_merge_controls),
        gr.update(visible=show_merge_controls),
    )


def _update_custom_visibility(choice, run_mode, multi_model):
    visible = (choice == "Custom...") and (run_mode == "Single" or not multi_model)
    return gr.update(visible=visible)


def ui_reset_prompt_and_preview(summary_mode_value, one_sentence_value):
    prompt = DEFAULT_PROMPT
    preview = build_final_prompt(prompt, summary_mode_value, one_sentence_value)
    return prompt, preview


with gr.Blocks() as iface:
    gr.Markdown("# Simple Captioner")
    gr.Markdown("Image and video captioning with Qwen VL and JoyCaption/LLaVA.")
    gr.Markdown(
        "Single mode keeps the classic one-model workflow. Multi-Pass mode can create raw captions from multiple model and attention combinations, then merge them into one final adjacent `.txt` file."
    )

    with gr.Accordion("Run Mode", open=True):
        with gr.Row():
            run_mode_radio = gr.Radio(
                label="Run Mode",
                choices=RUN_MODE_CHOICES,
                value="Single",
                info="Single uses one model and one attention setting. Multi-Pass runs multiple combinations and can merge them.",
            )
            multi_model_checkbox = gr.Checkbox(
                label="Multi-Model",
                value=False,
                info="When enabled in Multi-Pass mode, replace the single model dropdown with a model multi-select list.",
            )
            multi_atten_checkbox = gr.Checkbox(
                label="Multi-Atten",
                value=False,
                info="When enabled in Multi-Pass mode, replace the single attention selector with an attention multi-select list.",
            )

        with gr.Row():
            advanced_dual_load_checkbox = gr.Checkbox(
                label="Advanced: Dual-Load up to 2 models",
                value=False,
                visible=False,
                info="Optional advanced feature. Attempts to keep up to 2 models in memory. If it fails, the app falls back to sequential loading.",
            )
            enable_merge_checkbox = gr.Checkbox(
                label="Enable Merge Stage",
                value=True,
                visible=False,
                info="In Multi-Pass mode, merge successful raw captions into one final training caption.",
            )

    with gr.Accordion("Model Settings", open=True):
        with gr.Row():
            model_dropdown = gr.Dropdown(
                label="Model",
                choices=AVAILABLE_MODELS,
                value=DEFAULT_MODEL_ID,
                allow_custom_value=False,
                visible=True,
                interactive=True,
                info="Pick a Hugging Face Transformers caption model for the main single-model workflow or as fallback when Multi-Model is off.",
            )
            model_multiselect = gr.Dropdown(
                label="Models",
                choices=[m for m in AVAILABLE_MODELS if m != "Custom..."],
                value=[DEFAULT_MODEL_ID],
                multiselect=True,
                visible=False,
                interactive=True,
                info="Select multiple caption models to run in Multi-Pass mode.",
            )

        custom_model_box = gr.Textbox(
            label="Custom Model ID",
            placeholder="e.g. Qwen/Qwen3-VL-8B-Instruct",
            visible=False,
            info="Use this only when Model is set to Custom....",
        )

        with gr.Row():
            hf_model_url_input = gr.Textbox(
                label="Download compatible HF model",
                placeholder="e.g. https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct or Qwen/Qwen3-VL-8B-Instruct",
                info="Paste a Hugging Face model URL or repo id. The snapshot is downloaded to the local HF cache and added to the model dropdowns.",
            )
            download_model_button = gr.Button("Download / Add HF Model")

        with gr.Row():
            quant_dropdown = gr.Radio(
                label="Quantization",
                choices=QUANT_CHOICES,
                # choices=["None", "8-bit", "4-bit"],
                value=DEFAULT_QUANT,
                interactive=True,
                info="Lower-bit quantization can reduce VRAM use, but may affect quality or compatibility.",
            )
            attn_dropdown = gr.Radio(
                label="Attention Implementation",
                choices=ATTN_CHOICES,
                value=DEFAULT_ATTN,
                interactive=True,
                visible=True,
                info="Recommended default on Windows is eager for stability.",
            )
            attn_multiselect = gr.Dropdown(
                label="Attention Implementations",
                choices=ATTN_CHOICES,
                value=[DEFAULT_ATTN],
                multiselect=True,
                interactive=True,
                visible=False,
                info="Select multiple attention implementations for Multi-Pass mode.",
            )

        with gr.Row():
            load_button = gr.Button("Load / Reload Model")
            unload_button = gr.Button("Unload / Release VRAM")
            reset_button = gr.Button("Reset Prompt")
            reset_merge_prompt_button = gr.Button(
                "Reset Merge Prompt",
                visible=False,
            )

        with gr.Row():
            save_defaults_button = gr.Button("Save Current as My Favorite Settings")
            load_defaults_button = gr.Button("Load Default Settigns")

    with gr.Accordion("Model Information", open=False):
        model_name_display = gr.Textbox(label="Model Name", interactive=False)
        device_display = gr.Textbox(label="Device", interactive=False)
        vram_display = gr.Textbox(label="VRAM Usage", interactive=False)
        dtype_display = gr.Textbox(label="Torch Dtype", interactive=False)
        config_display = gr.Textbox(label="Model Config", lines=8, interactive=False)

    with gr.Accordion("Folder Processing", open=True):
        folder_input = gr.Textbox(
            label="Folder Path",
            placeholder=r"e.g. D:\Images\ToCaption",
            info="The app scans this folder and subfolders for supported image and video files.",
        )

        prompt_input = gr.Textbox(
            label="Custom Prompt",
            value=DEFAULT_PROMPT,
            lines=4,
            info="Main caption prompt used for raw caption generation.",
        )

        with gr.Row():
            skip_existing_checkbox = gr.Checkbox(
                label="Skip if adjacent final .txt already exists",
                value=False,
                info="Primary skip rule: if image001.png already has image001.txt beside it, the image is skipped unless you turn skipping off.",
            )
            retain_preview_checkbox = gr.Checkbox(
                label="Retain preview when skipping",
                value=True,
                info="Keep the last preview visible when a file is skipped.",
            )

        with gr.Row():
            summary_mode = gr.Checkbox(
                label="Summary Mode",
                value=False,
                info="Ask the model for a shorter summary-style caption.",
            )
            one_sentence_mode = gr.Checkbox(
                label="One-Sentence Mode",
                value=False,
                info="Ask the model to keep the caption to one sentence.",
            )
            resolution_mode = gr.Dropdown(
                label="Image Resolution",
                # choices=["auto", "auto_high", "fast", "high"],
                choices=RESOLUTION_CHOICES,
                value="auto",
                info="Controls the image sizing strategy passed into supported models.",
            )

        max_tokens_slider = gr.Slider(
            label="Max Tokens",
            minimum=304,
            maximum=2048,
            value=512,
            step=16,
            info="Maximum new tokens generated for each caption or merge response.",
        )

        prompt_preview = gr.Textbox(
            label="Final Prompt Preview",
            value=build_final_prompt(DEFAULT_PROMPT, False, False),
            lines=2,
            interactive=False,
            info="Preview of the prompt after Summary Mode and One-Sentence Mode modifiers are applied.",
        )

    with gr.Accordion("Merge Settings", open=True):
        merge_model_dropdown = gr.Dropdown(
            label="Merge Model",
            choices=MERGE_MODEL_CHOICES,
            value=DEFAULT_MERGE_MODEL_ID,
            visible=False,
            interactive=True,
            info="Model used to compare raw captions and produce one final consolidated caption.",
        )

        merge_help_markdown = gr.Markdown(
            "The merge stage combines successful raw captions conservatively, preserves concrete details, removes duplication, and avoids inventing details.",
            visible=False,
        )

        show_merge_prompt_checkbox = gr.Checkbox(
            label="Show editable merge prompt",
            value=False,
            visible=False,
            info="Keep the merge prompt hidden by default, but allow advanced users to inspect or edit it.",
        )

        merge_prompt_box = gr.Textbox(
            label="Merge Prompt",
            value=DEFAULT_MERGE_PROMPT,
            lines=14,
            visible=False,
            info="Rule-based anti-hallucination prompt used only during the merge stage.",
        )

        with gr.Row():
            save_raw_captions_checkbox = gr.Checkbox(
                label="Save raw captions",
                value=True,
                visible=False,
                info="Save raw captions into the _captions_raw subfolder.",
            )
            save_audit_checkbox = gr.Checkbox(
                label="Save audit file",
                value=True,
                visible=False,
                info="Write a text audit file containing sources, failures, and the final merged caption.",
            )

        with gr.Row():
            raw_file_handling_radio = gr.Radio(
                label="Raw File Handling",
                # choices=["Reuse existing raw files", "Overwrite raw files"],
                choices=RAW_FILE_HANDLING_CHOICES,
                value="Reuse existing raw files",
                visible=False,
                interactive=True,
                info="Choose exactly one behavior for existing raw caption files.",
            )

        merge_output_note_markdown = gr.Markdown(
            "Final output rule: `image001.txt` is written beside the image as the main caption. Raw files and optional audit files are saved inside `_captions_raw` in the same folder.",
            visible=False,
        )

    with gr.Row():
        start_button = gr.Button("Start Processing", variant="primary")
        abort_button = gr.Button("Abort", interactive=False)

    status_output = gr.Textbox(
        label="Status",
        interactive=False,
        info="Current action or error message.",
    )
    progressbar = gr.Slider(
        minimum=0,
        maximum=100,
        value=0,
        step=1,
        label="Progress",
        interactive=False,
    )
    time_display = gr.Textbox(
        label="Elapsed Time",
        interactive=False,
        info="Elapsed processing time for the current run.",
    )

    preflight_output = gr.Textbox(
        label="Preflight Summary",
        lines=16,
        interactive=False,
        info="Run summary shown before processing starts in Multi-Pass mode.",
    )

    with gr.Row():
        with gr.Column(scale=1):
            media_output = gr.Image(label="Current Image", interactive=False)
            media_name_markdown = gr.Markdown()
        with gr.Column(scale=1):
            caption_output = gr.Textbox(
                label="Generated / Final Caption", lines=18, interactive=False
            )

        ui_e = {
            "model_dropdown": model_dropdown,
            "quant_dropdown": quant_dropdown,
            "attn_dropdown": attn_dropdown,
            "load_button": load_button,
            "unload_button": unload_button,
            "download_model_button": download_model_button,
            "save_defaults_button": save_defaults_button,
            "load_defaults_button": load_defaults_button,
            "reset_button": reset_button,
            "start_button": start_button,
            "abort_button": abort_button,
            "folder_input": folder_input,
            "prompt_input": prompt_input,
            "skip_existing_checkbox": skip_existing_checkbox,
            "max_tokens_slider": max_tokens_slider,
            "summary_mode": summary_mode,
            "one_sentence_mode": one_sentence_mode,
            "retain_preview_checkbox": retain_preview_checkbox,
            "resolution_mode": resolution_mode,
            "run_mode_radio": run_mode_radio,
            "multi_model_checkbox": multi_model_checkbox,
            "multi_atten_checkbox": multi_atten_checkbox,
            "advanced_dual_load_checkbox": advanced_dual_load_checkbox,
            "model_multiselect": model_multiselect,
            "attn_multiselect": attn_multiselect,
            "enable_merge_checkbox": enable_merge_checkbox,
            "merge_model_dropdown": merge_model_dropdown,
            "merge_prompt_box": merge_prompt_box,
            "show_merge_prompt_checkbox": show_merge_prompt_checkbox,
            "reset_merge_prompt_button": reset_merge_prompt_button,
            "save_raw_captions_checkbox": save_raw_captions_checkbox,
            "save_audit_checkbox": save_audit_checkbox,
            "raw_file_handling_radio": raw_file_handling_radio,
            "hf_model_url_input": hf_model_url_input,
        }

    mode_visibility_outputs = [
        multi_model_checkbox,
        multi_atten_checkbox,
        model_dropdown,
        model_multiselect,
        attn_dropdown,
        attn_multiselect,
        enable_merge_checkbox,
        merge_model_dropdown,
        merge_help_markdown,
        show_merge_prompt_checkbox,
        merge_prompt_box,
        reset_merge_prompt_button,
        save_raw_captions_checkbox,
        save_audit_checkbox,
        raw_file_handling_radio,
        advanced_dual_load_checkbox,
        merge_output_note_markdown,
    ]

    control_outputs = [ui_e[k] for k in control_keys]
    process_outputs = [
        status_output,
        media_output,
        media_name_markdown,
        caption_output,
        progressbar,
        time_display,
        preflight_output,
        *control_outputs,
    ]

    defaults_load_outputs = [
        status_output,
        model_dropdown,
        model_multiselect,
        custom_model_box,
        merge_model_dropdown,
        quant_dropdown,
        attn_dropdown,
        attn_multiselect,
        run_mode_radio,
        multi_model_checkbox,
        multi_atten_checkbox,
        advanced_dual_load_checkbox,
        enable_merge_checkbox,
        show_merge_prompt_checkbox,
        save_raw_captions_checkbox,
        save_audit_checkbox,
        skip_existing_checkbox,
        retain_preview_checkbox,
        summary_mode,
        one_sentence_mode,
        resolution_mode,
        raw_file_handling_radio,
        max_tokens_slider,
        prompt_preview,
    ]

    model_dropdown.change(
        _toggle_custom,
        inputs=[model_dropdown],
        outputs=[custom_model_box],
    )

    model_dropdown.change(
        _update_custom_visibility,
        inputs=[model_dropdown, run_mode_radio, multi_model_checkbox],
        outputs=[custom_model_box],
    )
    run_mode_radio.change(
        _update_custom_visibility,
        inputs=[model_dropdown, run_mode_radio, multi_model_checkbox],
        outputs=[custom_model_box],
    )
    multi_model_checkbox.change(
        _update_custom_visibility,
        inputs=[model_dropdown, run_mode_radio, multi_model_checkbox],
        outputs=[custom_model_box],
    )

    run_mode_radio.change(
        _update_mode_visibility,
        inputs=[
            run_mode_radio,
            multi_model_checkbox,
            multi_atten_checkbox,
            enable_merge_checkbox,
            show_merge_prompt_checkbox,
        ],
        outputs=mode_visibility_outputs,
    )
    multi_model_checkbox.change(
        _update_mode_visibility,
        inputs=[
            run_mode_radio,
            multi_model_checkbox,
            multi_atten_checkbox,
            enable_merge_checkbox,
            show_merge_prompt_checkbox,
        ],
        outputs=mode_visibility_outputs,
    )
    multi_atten_checkbox.change(
        _update_mode_visibility,
        inputs=[
            run_mode_radio,
            multi_model_checkbox,
            multi_atten_checkbox,
            enable_merge_checkbox,
            show_merge_prompt_checkbox,
        ],
        outputs=mode_visibility_outputs,
    )
    enable_merge_checkbox.change(
        _update_mode_visibility,
        inputs=[
            run_mode_radio,
            multi_model_checkbox,
            multi_atten_checkbox,
            enable_merge_checkbox,
            show_merge_prompt_checkbox,
        ],
        outputs=mode_visibility_outputs,
    )
    show_merge_prompt_checkbox.change(
        _update_mode_visibility,
        inputs=[
            run_mode_radio,
            multi_model_checkbox,
            multi_atten_checkbox,
            enable_merge_checkbox,
            show_merge_prompt_checkbox,
        ],
        outputs=mode_visibility_outputs,
    )

    prompt_input.change(
        build_final_prompt,
        inputs=[prompt_input, summary_mode, one_sentence_mode],
        outputs=[prompt_preview],
    )
    summary_mode.change(
        build_final_prompt,
        inputs=[prompt_input, summary_mode, one_sentence_mode],
        outputs=[prompt_preview],
    )
    one_sentence_mode.change(
        build_final_prompt,
        inputs=[prompt_input, summary_mode, one_sentence_mode],
        outputs=[prompt_preview],
    )

    reset_button.click(
        ui_reset_prompt_and_preview,
        inputs=[summary_mode, one_sentence_mode],
        outputs=[prompt_input, prompt_preview],
    )

    reset_merge_prompt_button.click(
        ui_reset_merge_prompt,
        inputs=[],
        outputs=[merge_prompt_box],
    )

    load_button.click(
        ui_load_model,
        inputs=[model_dropdown, custom_model_box, quant_dropdown, attn_dropdown],
        outputs=[
            status_output,
            model_name_display,
            device_display,
            vram_display,
            dtype_display,
            config_display,
        ],
    )

    unload_button.click(
        ui_unload_model,
        inputs=[],
        outputs=[
            status_output,
            model_name_display,
            device_display,
            vram_display,
            dtype_display,
            config_display,
        ],
    )

    download_model_button.click(
        ui_download_hf_model,
        inputs=[hf_model_url_input, model_multiselect, merge_model_dropdown],
        outputs=[
            status_output,
            hf_model_url_input,
            model_dropdown,
            model_multiselect,
            merge_model_dropdown,
        ],
    )

    save_defaults_button.click(
        ui_save_current_as_defaults,
        inputs=[
            model_dropdown,
            model_multiselect,
            custom_model_box,
            merge_model_dropdown,
            quant_dropdown,
            attn_dropdown,
            attn_multiselect,
            run_mode_radio,
            multi_model_checkbox,
            multi_atten_checkbox,
            advanced_dual_load_checkbox,
            enable_merge_checkbox,
            show_merge_prompt_checkbox,
            save_raw_captions_checkbox,
            save_audit_checkbox,
            skip_existing_checkbox,
            retain_preview_checkbox,
            summary_mode,
            one_sentence_mode,
            resolution_mode,
            raw_file_handling_radio,
            max_tokens_slider,
        ],
        outputs=[status_output],
    )

    load_defaults_button.click(
        ui_load_my_defaults,
        inputs=[prompt_input],
        outputs=defaults_load_outputs,
    ).then(
        _update_mode_visibility,
        inputs=[
            run_mode_radio,
            multi_model_checkbox,
            multi_atten_checkbox,
            enable_merge_checkbox,
            show_merge_prompt_checkbox,
        ],
        outputs=mode_visibility_outputs,
    ).then(
        _update_custom_visibility,
        inputs=[model_dropdown, run_mode_radio, multi_model_checkbox],
        outputs=[custom_model_box],
    )

    start_button.click(
        start_process,
        inputs=[],
        outputs=control_outputs,
        queue=False,
    ).then(
        process_folder_dispatch,
        inputs=[
            folder_input,
            prompt_input,
            skip_existing_checkbox,
            max_tokens_slider,
            summary_mode,
            one_sentence_mode,
            retain_preview_checkbox,
            resolution_mode,
            run_mode_radio,
            multi_model_checkbox,
            multi_atten_checkbox,
            model_dropdown,
            model_multiselect,
            custom_model_box,
            attn_dropdown,
            attn_multiselect,
            quant_dropdown,
            enable_merge_checkbox,
            merge_model_dropdown,
            merge_prompt_box,
            save_raw_captions_checkbox,
            save_audit_checkbox,
            raw_file_handling_radio,
            advanced_dual_load_checkbox,
        ],
        outputs=process_outputs,
    )

    abort_button.click(
        abort_process,
        inputs=[],
        outputs=[status_output, *control_outputs],
        queue=False,
    )

    iface.load(
        fn=get_model_info,
        inputs=[],
        outputs=[
            model_name_display,
            device_display,
            vram_display,
            dtype_display,
            config_display,
        ],
    )

    iface.load(
        fn=ui_load_my_defaults,
        inputs=[prompt_input],
        outputs=defaults_load_outputs,
    ).then(
        _update_mode_visibility,
        inputs=[
            run_mode_radio,
            multi_model_checkbox,
            multi_atten_checkbox,
            enable_merge_checkbox,
            show_merge_prompt_checkbox,
        ],
        outputs=mode_visibility_outputs,
    ).then(
        _update_custom_visibility,
        inputs=[model_dropdown, run_mode_radio, multi_model_checkbox],
        outputs=[custom_model_box],
    )


if __name__ == "__main__":
    iface.queue()
    iface.launch(
        share=False,
        theme=gr.themes.Base(),
        css=css,
    )
