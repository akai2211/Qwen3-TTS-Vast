# coding=utf-8
import os
import sys
import warnings

# Suppress common warnings
warnings.filterwarnings("ignore", message=".*Min value of input waveform.*")
warnings.filterwarnings("ignore", message=".*Max value of input waveform.*")
warnings.filterwarnings("ignore", message=".*Trying to convert audio automatically.*")
warnings.filterwarnings("ignore", message=".*pad_token_id.*")
warnings.filterwarnings("ignore", message=".*Setting `pad_token_id`.*")

# Only attempt flash-attn install on Linux (HF Spaces)
if sys.platform == "linux":
    import subprocess
    subprocess.run('pip install flash-attn --no-build-isolation', env={'FLASH_ATTENTION_SKIP_CUDA_BUILD': "TRUE"}, shell=True)

import gradio as gr
import numpy as np
import torch
from huggingface_hub import snapshot_download, scan_cache_dir

# Whisper model for transcription
whisper_model = None


def get_device():
    """Get the best available device for this platform."""
    if torch.cuda.is_available():
        return "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"


def get_whisper_model():
    """Load Whisper tiny model for transcription."""
    global whisper_model
    if whisper_model is None:
        import whisper
        device = get_device()
        whisper_model = whisper.load_model("tiny", device=device)
    return whisper_model


def unload_whisper():
    """Force unload whisper model from GPU."""
    global whisper_model
    if whisper_model is not None:
        # Move to CPU first, then delete
        try:
            whisper_model.cpu()
        except:
            pass
        whisper_model = None
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        torch.mps.empty_cache()


def transcribe_audio(audio):
    """Transcribe audio using Whisper tiny."""
    global whisper_model
    if audio is None:
        return "Пожалуйста, сначала загрузите аудио."
    
    try:
        sr, wav = audio
        # Convert to float32 and normalize properly
        wav = wav.astype(np.float32)
        
        # Check if audio needs normalization (int16 range is -32768 to 32767)
        max_val = np.abs(wav).max()
        if max_val > 1.0:
            wav = wav / max_val  # Normalize to [-1, 1] range
        
        # Whisper expects 16kHz mono
        if len(wav.shape) > 1:
            wav = wav.mean(axis=1)
        
        if sr != 16000:
            import librosa
            wav = librosa.resample(wav, orig_sr=sr, target_sr=16000)
        
        model = get_whisper_model()
        device = get_device()
        result = model.transcribe(wav, fp16=(device == "cuda"))
        text = result["text"].strip()
        
        # Unload whisper to free GPU memory
        unload_whisper()
        
        return text
    except Exception as e:
        # Still try to unload on error
        unload_whisper()
        return f"Ошибка транскрипции: {str(e)}"

# Global model holders - keyed by (model_type, model_size)
loaded_models = {}

# Model size options
MODEL_SIZES = ["0.6B", "1.7B"]

# Available models configuration
AVAILABLE_MODELS = {
    "VoiceDesign": {
        "sizes": ["1.7B"],
        "description": "Создавайте уникальные голоса с помощью текстовых описаний"
    },
    "Base": {
        "sizes": ["0.6B", "1.7B"],
        "description": "Клонирование голоса на основе эталонного аудио"
    },
    "CustomVoice": {
        "sizes": ["0.6B", "1.7B"],
        "description": "TTS с предустановленными дикторами и инструкциями по стилю"
    }
}


def get_model_repo_id(model_type: str, model_size: str) -> str:
    """Get HuggingFace repo ID for a model."""
    return f"Qwen/Qwen3-TTS-12Hz-{model_size}-{model_type}"


def get_model_path(model_type: str, model_size: str) -> str:
    """Get model path based on type and size."""
    return snapshot_download(get_model_repo_id(model_type, model_size))


def check_model_downloaded(model_type: str, model_size: str) -> bool:
    """Check if a model is already downloaded in the cache."""
    try:
        cache_info = scan_cache_dir()
        repo_id = get_model_repo_id(model_type, model_size)
        for repo in cache_info.repos:
            if repo.repo_id == repo_id:
                return True
        return False
    except Exception:
        return False


def get_downloaded_models_status() -> str:
    """Get status of all available models."""
    lines = ["### Статус загрузки моделей\n"]
    for model_type, info in AVAILABLE_MODELS.items():
        lines.append(f"**{model_type}** - {info['description']}")
        for size in info["sizes"]:
            status = "✅ Загружено" if check_model_downloaded(model_type, size) else "⬜ Не загружено"
            lines.append(f"  - {size}: {status}")
        lines.append("")
    return "\n".join(lines)


def download_model(model_type: str, model_size: str, progress=gr.Progress()):
    """Download a specific model."""
    if model_size not in AVAILABLE_MODELS.get(model_type, {}).get("sizes", []):
        return f"❌ Неверная комбинация: {model_type} {model_size}", get_downloaded_models_status()
    
    repo_id = get_model_repo_id(model_type, model_size)
    
    if check_model_downloaded(model_type, model_size):
        return f"✅ {model_type} {model_size} уже загружена!", get_downloaded_models_status()
    
    try:
        progress(0, desc=f"Загрузка {model_type} {model_size}...")
        snapshot_download(repo_id)
        progress(1, desc="Завершено!")
        return f"✅ {model_type} {model_size} успешно загружена!", get_downloaded_models_status()
    except Exception as e:
        return f"❌ Ошибка при загрузке {model_type} {model_size}: {str(e)}", get_downloaded_models_status()


def get_available_sizes(model_type: str):
    """Get available sizes for a model type."""
    return gr.update(choices=AVAILABLE_MODELS.get(model_type, {}).get("sizes", []), value=AVAILABLE_MODELS.get(model_type, {}).get("sizes", ["1.7B"])[0])


def get_model(model_type: str, model_size: str):
    """Get or load a model by type and size."""
    global loaded_models
    key = (model_type, model_size)
    if key not in loaded_models:
        from qwen_tts import Qwen3TTSModel
        model_path = get_model_path(model_type, model_size)
        device = get_device()
        
        # Use appropriate device mapping and dtype
        if device == "cuda":
            loaded_models[key] = Qwen3TTSModel.from_pretrained(
                model_path,
                device_map="cuda",
                dtype=torch.bfloat16,
            )
        else:
            # For MPS and CPU, use CPU with float32
            # MPS support for large models is experimental and can cause issues
            loaded_models[key] = Qwen3TTSModel.from_pretrained(
                model_path,
                device_map="cpu",
                dtype=torch.float32,
            )
    return loaded_models[key]


def get_loaded_models_status() -> str:
    """Get status of currently loaded models in memory."""
    if not loaded_models:
        return "В памяти нет загруженных моделей."
    
    lines = ["**Сейчас загружены в память:**"]
    for (model_type, model_size) in loaded_models.keys():
        lines.append(f"- {model_type} ({model_size})")
    return "\n".join(lines)


def load_model_manual(model_type: str, model_size: str, progress=gr.Progress()):
    """Manually load a model into memory."""
    if model_size not in AVAILABLE_MODELS.get(model_type, {}).get("sizes", []):
        return f"❌ Неверная комбинация: {model_type} {model_size}", get_loaded_models_status()
    
    key = (model_type, model_size)
    if key in loaded_models:
        return f"✅ {model_type} {model_size} уже загружена!", get_loaded_models_status()
    
    try:
        progress(0, desc=f"Загрузка {model_type} {model_size}...")
        get_model(model_type, model_size)
        progress(1, desc="Завершено!")
        return f"✅ {model_type} {model_size} успешно загружена!", get_loaded_models_status()
    except Exception as e:
        return f"❌ Ошибка при загрузке {model_type} {model_size}: {str(e)}", get_loaded_models_status()


def unload_model(model_type: str, model_size: str):
    """Unload a specific model from memory."""
    global loaded_models
    key = (model_type, model_size)
    
    if key not in loaded_models:
        return f"⚠️ {model_type} {model_size} не загружена.", get_loaded_models_status()
    
    try:
        del loaded_models[key]
        device = get_device()
        if device == "cuda":
            torch.cuda.empty_cache()
        elif device == "mps":
            torch.mps.empty_cache()
        return f"✅ Модель {model_type} {model_size} выгружена, память видеокарты освобождена.", get_loaded_models_status()
    except Exception as e:
        return f"❌ Ошибка при выгрузке: {str(e)}", get_loaded_models_status()


def unload_all_models():
    """Unload all models from memory."""
    global loaded_models
    
    if not loaded_models:
        return "⚠️ В памяти нет загруженных моделей.", get_loaded_models_status()
    
    try:
        count = len(loaded_models)
        loaded_models.clear()
        device = get_device()
        if device == "cuda":
            torch.cuda.empty_cache()
        elif device == "mps":
            torch.mps.empty_cache()
        return f"✅ {count} мод. выгружено, память видеокарты освобождена.", get_loaded_models_status()
    except Exception as e:
        return f"❌ Ошибка при выгрузке: {str(e)}", get_loaded_models_status()


def _normalize_audio(wav, eps=1e-12, clip=True):
    """Normalize audio to float32 in [-1, 1] range."""
    x = np.asarray(wav)

    if np.issubdtype(x.dtype, np.integer):
        info = np.iinfo(x.dtype)
        if info.min < 0:
            y = x.astype(np.float32) / max(abs(info.min), info.max)
        else:
            mid = (info.max + 1) / 2.0
            y = (x.astype(np.float32) - mid) / mid
    elif np.issubdtype(x.dtype, np.floating):
        y = x.astype(np.float32)
        m = np.max(np.abs(y)) if y.size else 0.0
        if m > 1.0 + 1e-6:
            y = y / (m + eps)
    else:
        raise TypeError(f"Unsupported dtype: {x.dtype}")

    if clip:
        y = np.clip(y, -1.0, 1.0)

    if y.ndim > 1:
        y = np.mean(y, axis=-1).astype(np.float32)

    return y


import re

def chunk_text(text: str, max_chars: int = 200) -> list:
    """
    Split text into chunks without cutting words.
    Tries to split on sentence boundaries first, then falls back to word boundaries.
    """
    text = text.strip()
    if not text:
        return []
    
    if len(text) <= max_chars:
        return [text]
    
    # Sentence-ending punctuation patterns
    sentence_endings = re.compile(r'(?<=[.!?。！？])\s+')
    
    # Split into sentences first
    sentences = sentence_endings.split(text)
    
    chunks = []
    current_chunk = ""
    
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
            
        # If single sentence is too long, split by words
        if len(sentence) > max_chars:
            # Flush current chunk first
            if current_chunk:
                chunks.append(current_chunk.strip())
                current_chunk = ""
            
            # Split long sentence by words
            words = sentence.split()
            for word in words:
                if len(current_chunk) + len(word) + 1 <= max_chars:
                    current_chunk = current_chunk + " " + word if current_chunk else word
                else:
                    if current_chunk:
                        chunks.append(current_chunk.strip())
                    current_chunk = word
        else:
            # Try to add sentence to current chunk
            test_chunk = current_chunk + " " + sentence if current_chunk else sentence
            if len(test_chunk) <= max_chars:
                current_chunk = test_chunk
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = sentence
    
    # Don't forget the last chunk
    if current_chunk:
        chunks.append(current_chunk.strip())
    
    return chunks


def _audio_to_tuple(audio):
    """Convert Gradio audio input to (wav, sr) tuple."""
    if audio is None:
        return None

    if isinstance(audio, tuple) and len(audio) == 2 and isinstance(audio[0], int):
        sr, wav = audio
        wav = _normalize_audio(wav)
        return wav, int(sr)

    if isinstance(audio, dict) and "sampling_rate" in audio and "data" in audio:
        sr = int(audio["sampling_rate"])
        wav = _normalize_audio(audio["data"])
        return wav, sr

    return None


# Speaker and language choices for CustomVoice model
SPEAKERS = [
    "Aiden", "Dylan", "Eric", "Ono_anna", "Ryan", "Serena", "Sohee", "Uncle_fu", "Vivian"
]
LANGUAGES = ["Auto", "Italian", "Chinese", "English", "Japanese", "Korean", "French", "German", "Spanish", "Portuguese", "Russian"]


import random

def set_seed(seed):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = get_device()
    if device == "cuda":
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # Force deterministic algorithms
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    elif device == "mps":
        # MPS doesn't need special seed handling
        pass


def generate_voice_design(text, language, voice_description, seed):
    """Generate speech using Voice Design model (1.7B only)."""
    if not text or not text.strip():
        return None, "Error: Text is required."
    if not voice_description or not voice_description.strip():
        return None, "Error: Voice description is required."

    try:
        # Handle seed - if -1 (auto), generate one
        if seed == -1:
            seed = random.randint(0, 2147483647)
        seed = int(seed)
        set_seed(seed)
        
        tts = get_model("VoiceDesign", "1.7B")
        
        print(f"\n{'='*50}")
        print(f"🎨 Voice Design Generation")
        print(f"{'='*50}")
        print(f"🎲 Seed: {seed}")
        print(f"📝 Текст: {len(text)} симв.")
        
        wavs, sr = tts.generate_voice_design(
            text=text.strip(),
            language=language,
            instruct=voice_description.strip(),
            non_streaming_mode=True,
            max_new_tokens=2048,
        )
        
        total_duration = len(wavs[0]) / sr
        print(f"\n{'='*50}")
        print(f"✅ Готово! Длительность: {total_duration:.2f}с")
        print(f"{'='*50}\n")
        
        status = f"Сгенерировано {total_duration:.1f}с аудио | Seed: {seed}"
        return (sr, wavs[0]), status
    except Exception as e:
        print(f"❌ Ошибка: {type(e).__name__}: {e}")
        return None, f"Ошибка: {type(e).__name__}: {e}"


def generate_voice_clone(ref_audio, ref_text, target_text, language, use_xvector_only, model_size, max_chunk_chars, chunk_gap, seed):
    """Generate speech using Base (Voice Clone) model."""
    if not target_text or not target_text.strip():
        return None, "Ошибка: Текст обязателен."

    audio_tuple = _audio_to_tuple(ref_audio)
    if audio_tuple is None:
        return None, "Ошибка: Эталонное аудио обязательно."

    if not use_xvector_only and (not ref_text or not ref_text.strip()):
        return None, "Ошибка: Эталонный текст обязателен, если не включен 'Только x-vector'."

    try:
        from tqdm import tqdm
        
        # Handle seed - if -1 (auto), generate one and use it for all chunks
        if seed == -1:
            seed = random.randint(0, 2147483647)
        seed = int(seed)
        
        tts = get_model("Base", model_size)
        chunks = chunk_text(target_text.strip(), max_chars=int(max_chunk_chars))
        
        print(f"\n{'='*50}")
        print(f"🎭 Voice Clone Generation ({model_size})")
        print(f"{'='*50}")
        print(f"🎲 Seed: {seed}")
        print(f"📝 Text length: {len(target_text)} chars → {len(chunks)} chunk(s)")
        print(f"⏱️ Chunk gap: {chunk_gap}s")
        
        all_wavs = []
        sr = None
        for i, chunk in enumerate(tqdm(chunks, desc="Генерация частей", unit="chunk")):
            # Set seed before each chunk to ensure consistency
            set_seed(seed)
            
            print(f"\n🔊 Часть {i+1}/{len(chunks)} [Seed: {seed}]: \"{chunk[:50]}{'...' if len(chunk) > 50 else ''}\"")
            wavs, sr = tts.generate_voice_clone(
                text=chunk,
                language=language,
                ref_audio=audio_tuple,
                ref_text=ref_text.strip() if ref_text else None,
                x_vector_only_mode=use_xvector_only,
                max_new_tokens=2048,
            )
            all_wavs.append(wavs[0])
            print(f"   ✅ Сгенерировано {len(wavs[0])/sr:.2f}с аудио")
        
        # Concatenate all audio chunks with gap (silence) between them
        if len(all_wavs) > 1 and chunk_gap > 0:
            gap_samples = int(sr * chunk_gap)
            silence = np.zeros(gap_samples, dtype=np.float32)
            chunks_with_gaps = []
            for i, wav in enumerate(all_wavs):
                chunks_with_gaps.append(wav)
                if i < len(all_wavs) - 1:  # Don't add gap after last chunk
                    chunks_with_gaps.append(silence)
            final_wav = np.concatenate(chunks_with_gaps)
        else:
            final_wav = np.concatenate(all_wavs) if len(all_wavs) > 1 else all_wavs[0]
        
        total_duration = len(final_wav) / sr
        print(f"\n{'='*50}")
        print(f"✅ Готово! Общая длительность: {total_duration:.2f}с")
        print(f"{'='*50}\n")
        
        status = f"Сгенерировано {len(chunks)} частей, всего {total_duration:.1f}с | Seed: {seed}" if len(chunks) > 1 else f"Сгенерировано {total_duration:.1f}с аудио | Seed: {seed}"
        return (sr, final_wav), status
    except Exception as e:
        print(f"❌ Ошибка: {type(e).__name__}: {e}")
        return None, f"Ошибка: {type(e).__name__}: {e}"


def generate_custom_voice(text, language, speaker, instruct, model_size, seed):
    """Generate speech using CustomVoice model."""
    if not text or not text.strip():
        return None, "Ошибка: Текст обязателен."
    if not speaker:
        return None, "Ошибка: Диктор обязателен."

    try:
        # Handle seed - if -1 (auto), generate one
        if seed == -1:
            seed = random.randint(0, 2147483647)
        seed = int(seed)
        set_seed(seed)
        
        tts = get_model("CustomVoice", model_size)
        
        print(f"\n{'='*50}")
        print(f"🗣️ Custom Voice Generation ({model_size})")
        print(f"{'='*50}")
        print(f"🎲 Seed: {seed}")
        print(f"👤 Диктор: {speaker}")
        print(f"📝 Текст: {len(text)} симв.")
        
        wavs, sr = tts.generate_custom_voice(
            text=text.strip(),
            language=language,
            speaker=speaker.lower().replace(" ", "_"),
            instruct=instruct.strip() if instruct else None,
            non_streaming_mode=True,
            max_new_tokens=2048,
        )
        
        total_duration = len(wavs[0]) / sr
        print(f"\n{'='*50}")
        print(f"✅ Готово! Длительность: {total_duration:.2f}с")
        print(f"{'='*50}\n")
        
        status = f"Сгенерировано {total_duration:.1f}с аудио | Seed: {seed}"
        return (sr, wavs[0]), status
    except Exception as e:
        print(f"❌ Ошибка: {type(e).__name__}: {e}")
        return None, f"Ошибка: {type(e).__name__}: {e}"


# Build Gradio UI
def build_ui():
    theme = gr.themes.Soft(
        font=[gr.themes.GoogleFont("Inter"), "Arial", "sans-serif"],
        primary_hue="indigo",
        secondary_hue="slate",
    )

    css = """
    .gradio-container {
        max-width: 100% !important;
        padding: 0 2rem !important;
    }
    .header-container {
        text-align: center;
        padding: 2rem 1rem;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 16px;
        margin-bottom: 1.5rem;
    }
    .header-container h1 {
        color: white !important;
        font-size: 2.5rem !important;
        margin-bottom: 0.5rem !important;
    }
    .header-container p {
        color: rgba(255,255,255,0.9) !important;
        font-size: 1.1rem !important;
    }
    .feature-badge {
        display: inline-block;
        background: rgba(255,255,255,0.2);
        padding: 0.3rem 0.8rem;
        border-radius: 20px;
        margin: 0.2rem;
        font-size: 0.85rem;
        color: white;
    }
    .tab-content {
        min-height: 500px !important;
    }
    .tabitem {
        min-height: 500px !important;
    }
    """

    with gr.Blocks(title="Qwen3-TTS Demo") as demo:
        gr.HTML(
            """
            <div class="header-container">
                <h1>🎙️ Qwen3-TTS</h1>
                <p>Высококачественный синтез речи с функциями клонирования и дизайна голоса</p>
                <div style="margin-top: 1rem;">
                    <span class="feature-badge">🎨 Дизайн голоса</span>
                    <span class="feature-badge">🎭 Клонирование</span>
                    <span class="feature-badge">🗣️ Свои голоса</span>
                    <span class="feature-badge">📝 Длинные тексты</span>
                </div>
            </div>
            """
        )

        with gr.Tabs():
            # Tab 0: Model Management (Collapsible sections)
            with gr.Tab("⚙️ Модели"):
                with gr.Accordion("📥 Загрузка моделей", open=True):
                    gr.Markdown("*💡 Совет: Модели можно загрузить здесь или они загрузятся автоматически при генерации в любой вкладке.*")
                    with gr.Row():
                        with gr.Column(scale=1):
                            with gr.Row():
                                download_model_type = gr.Dropdown(
                                    label="Тип",
                                    choices=list(AVAILABLE_MODELS.keys()),
                                    value="CustomVoice",
                                    interactive=True,
                                    scale=2,
                                )
                                download_model_size = gr.Dropdown(
                                    label="Размер",
                                    choices=["0.6B", "1.7B"],
                                    value="1.7B",
                                    interactive=True,
                                    scale=1,
                                )
                            download_btn = gr.Button("Скачать", variant="primary", size="sm")
                            download_status = gr.Textbox(label="Статус", lines=1, interactive=False)
                        with gr.Column(scale=2):
                            models_status = gr.Markdown(value=get_downloaded_models_status)
                
                download_model_type.change(
                    get_available_sizes,
                    inputs=[download_model_type],
                    outputs=[download_model_size],
                )
                
                download_btn.click(
                    download_model,
                    inputs=[download_model_type, download_model_size],
                    outputs=[download_status, models_status],
                )
                
                with gr.Accordion("🚀 Загрузить в GPU", open=False):
                    with gr.Row():
                        with gr.Column(scale=1):
                            with gr.Row():
                                load_model_type = gr.Dropdown(
                                    label="Тип",
                                    choices=list(AVAILABLE_MODELS.keys()),
                                    value="CustomVoice",
                                    interactive=True,
                                    scale=2,
                                )
                                load_model_size = gr.Dropdown(
                                    label="Размер",
                                    choices=["0.6B", "1.7B"],
                                    value="1.7B",
                                    interactive=True,
                                    scale=1,
                                )
                            load_btn = gr.Button("Загрузить в GPU", variant="primary", size="sm")
                            load_status = gr.Textbox(label="Статус", lines=1, interactive=False)
                        with gr.Column(scale=2):
                            load_refresh_btn = gr.Button("🔄 Обновить статус", size="sm")
                            load_loaded_status = gr.Markdown(value=get_loaded_models_status)
                
                load_model_type.change(
                    get_available_sizes,
                    inputs=[load_model_type],
                    outputs=[load_model_size],
                )
                
                load_refresh_btn.click(
                    lambda: get_loaded_models_status(),
                    inputs=[],
                    outputs=[load_loaded_status],
                )
                
                load_btn.click(
                    load_model_manual,
                    inputs=[load_model_type, load_model_size],
                    outputs=[load_status, load_loaded_status],
                )
                
                with gr.Accordion("🗑️ Выгрузить модели", open=False):
                    gr.Markdown("*💡 Совет: Нажмите 'Обновить статус', чтобы увидеть модели, загруженные из других вкладок.*")
                    with gr.Row():
                        with gr.Column(scale=1):
                            with gr.Row():
                                unload_model_type = gr.Dropdown(
                                    label="Тип",
                                    choices=list(AVAILABLE_MODELS.keys()),
                                    value="CustomVoice",
                                    interactive=True,
                                    scale=2,
                                )
                                unload_model_size = gr.Dropdown(
                                    label="Размер",
                                    choices=["0.6B", "1.7B"],
                                    value="1.7B",
                                    interactive=True,
                                    scale=1,
                                )
                            with gr.Row():
                                unload_btn = gr.Button("Выгрузить выбранную", variant="secondary", size="sm")
                                unload_all_btn = gr.Button("Выгрузить все", variant="stop", size="sm")
                            unload_status = gr.Textbox(label="Статус", lines=1, interactive=False)
                        with gr.Column(scale=2):
                            refresh_btn = gr.Button("🔄 Обновить статус", size="sm")
                            loaded_status = gr.Markdown(value=get_loaded_models_status)
                
                unload_model_type.change(
                    get_available_sizes,
                    inputs=[unload_model_type],
                    outputs=[unload_model_size],
                )
                
                refresh_btn.click(
                    lambda: get_loaded_models_status(),
                    inputs=[],
                    outputs=[loaded_status],
                )
                
                unload_btn.click(
                    unload_model,
                    inputs=[unload_model_type, unload_model_size],
                    outputs=[unload_status, loaded_status],
                )
                
                unload_all_btn.click(
                    unload_all_models,
                    inputs=[],
                    outputs=[unload_status, loaded_status],
                )

            # Tab 1: Voice Design
            with gr.Tab("🎨 Дизайн голоса"):
                gr.Markdown("*ℹ️ Дизайн голоса создает уникальные голоса на основе описаний. Макс. ~2048 токенов (рекомендуется ~300-500 симв.). Без разбивки на части — для длинных текстов используйте Клонирование или Свои голоса.*")
                with gr.Row(equal_height=True):
                    with gr.Column(scale=1):
                        design_text = gr.Textbox(
                            label="Текст для синтеза",
                            lines=6,
                            placeholder="Введите текст, который хотите озвучить (до ~500 симв.)...",
                            value="Это в верхнем ящике... подожди, там пусто? Быть не может! Я точно помню, что положил это туда!"
                        )
                        design_instruct = gr.Textbox(
                            label="Описание голоса",
                            lines=3,
                            placeholder="Опишите характер голоса...",
                            value="Говорите недоверчивым тоном, с легким оттенком начинающейся паники в голосе."
                        )
                        with gr.Row():
                            design_language = gr.Dropdown(
                                label="Язык",
                                choices=LANGUAGES,
                                value="Auto",
                                interactive=True,
                            )
                            design_seed = gr.Number(
                                label="Seed (-1 = Авто)",
                                value=-1,
                                precision=0,
                            )
                        design_btn = gr.Button("🎙️ Генерировать", variant="primary", size="lg")

                    with gr.Column(scale=1):
                        design_audio_out = gr.Audio(label="Результат", type="numpy")
                        design_status = gr.Textbox(label="Статус", lines=2, interactive=False)

                design_btn.click(
                    generate_voice_design,
                    inputs=[design_text, design_language, design_instruct, design_seed],
                    outputs=[design_audio_out, design_status],
                )

            # Tab 2: Voice Clone
            with gr.Tab("🎭 Клонирование голоса"):
                with gr.Row(equal_height=True):
                    with gr.Column(scale=1):
                        clone_ref_audio = gr.Audio(
                            label="Эталонное аудио",
                            type="numpy",
                        )
                        with gr.Row():
                            clone_ref_text = gr.Textbox(
                                label="Эталонный текст",
                                lines=2,
                                placeholder="Транскрипция эталонного аудио...",
                                scale=3,
                            )
                            transcribe_btn = gr.Button("🎤 Транскрибировать", scale=1)
                        clone_xvector = gr.Checkbox(
                            label="Только x-vector (текст не нужен, качество ниже)",
                            value=False,
                        )
                        clone_target_text = gr.Textbox(
                            label="Целевой текст",
                            lines=5,
                            placeholder="Текст для озвучки клонированным голосом...",
                        )
                        with gr.Row():
                            clone_language = gr.Dropdown(
                                label="Язык",
                                choices=LANGUAGES,
                                value="Auto",
                                interactive=True,
                            )
                            clone_model_size = gr.Dropdown(
                                label="Размер",
                                choices=MODEL_SIZES,
                                value="1.7B",
                                interactive=True,
                            )
                        with gr.Row():
                            clone_chunk_size = gr.Slider(
                                label="Размер части (симв.)",
                                minimum=50,
                                maximum=500,
                                value=200,
                                step=10,
                            )
                            clone_chunk_gap = gr.Slider(
                                label="Пауза между частями (с)",
                                minimum=0.0,
                                maximum=3.0,
                                value=0.0,
                                step=0.01,
                            )
                        with gr.Row():
                            clone_seed = gr.Number(
                                label="Seed (-1 = Авто)",
                                value=-1,
                                precision=0,
                            )
                        clone_btn = gr.Button("🎙️ Клонировать и озвучить", variant="primary", size="lg")

                    with gr.Column(scale=1):
                        clone_audio_out = gr.Audio(label="Результат", type="numpy")
                        clone_status = gr.Textbox(label="Статус", lines=2, interactive=False)

                transcribe_btn.click(
                    transcribe_audio,
                    inputs=[clone_ref_audio],
                    outputs=[clone_ref_text],
                )
                
                clone_btn.click(
                    generate_voice_clone,
                    inputs=[clone_ref_audio, clone_ref_text, clone_target_text, clone_language, clone_xvector, clone_model_size, clone_chunk_size, clone_chunk_gap, clone_seed],
                    outputs=[clone_audio_out, clone_status],
                )

            # Tab 3: Custom Voice TTS
            with gr.Tab("🗣️ Свои голоса"):
                gr.Markdown("*ℹ️ Эта вкладка использует предустановленных дикторов. Макс. ~2048 токенов (рекомендуется ~300-500 симв.). Для более длинных текстов используйте Клонирование.*")
                with gr.Row(equal_height=True):
                    with gr.Column(scale=1):
                        tts_text = gr.Textbox(
                            label="Текст для синтеза",
                            lines=6,
                            placeholder="Введите текст, который хотите озвучить (до ~500 симв.)...",
                            value="Привет! Добро пожаловать в систему синтеза речи. Это демонстрация наших возможностей TTS."
                        )
                        with gr.Row():
                            tts_speaker = gr.Dropdown(
                                label="Диктор",
                                choices=SPEAKERS,
                                value="Ryan",
                                interactive=True,
                            )
                            tts_language = gr.Dropdown(
                                label="Язык",
                                choices=LANGUAGES,
                                value="English",
                                interactive=True,
                            )
                        tts_instruct = gr.Textbox(
                            label="Инструкция по стилю (опционально, только 1.7B)",
                            lines=2,
                            placeholder="Например: Говорите веселым и энергичным тоном",
                        )
                        with gr.Row():
                            tts_model_size = gr.Dropdown(
                                label="Размер",
                                choices=MODEL_SIZES,
                                value="1.7B",
                                interactive=True,
                            )
                            tts_seed = gr.Number(
                                label="Seed (-1 = Авто)",
                                value=-1,
                                precision=0,
                            )
                        tts_btn = gr.Button("🎙️ Генерировать", variant="primary", size="lg")

                    with gr.Column(scale=1):
                        tts_audio_out = gr.Audio(label="Результат", type="numpy")
                        tts_status = gr.Textbox(label="Статус", lines=2, interactive=False)

                tts_btn.click(
                    generate_custom_voice,
                    inputs=[tts_text, tts_language, tts_speaker, tts_instruct, tts_model_size, tts_seed],
                    outputs=[tts_audio_out, tts_status],
                )

    return demo, theme, css


if __name__ == "__main__":
    demo, theme, css = build_ui()
    demo.launch(theme=theme, css=css)
