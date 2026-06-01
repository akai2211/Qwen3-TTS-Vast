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

# Flash-attn is installed via torch.js during Pinokio install - no runtime install needed

import gradio as gr
import numpy as np
import torch
import json
import shutil
import scipy.io.wavfile as wavfile
from pathlib import Path
from huggingface_hub import snapshot_download, scan_cache_dir
from database import VOICE_DATABASE

# Persistent paths on Vast.ai (/workspace survives container restarts).
WORKSPACE = Path(os.environ.get("WORKSPACE", "/workspace"))
_hf_home = os.environ.get("HF_HOME") or str(WORKSPACE / "cache" / "huggingface")
os.environ.setdefault("HF_HOME", _hf_home)
os.environ.setdefault("HUGGINGFACE_HUB_CACHE", _hf_home)

# Whisper model for transcription
whisper_model = None

# Custom voices storage
VOICES_DIR = WORKSPACE / "custom_voices"
VOICES_JSON = VOICES_DIR / "voices.json"
VOICES_DIR.mkdir(parents=True, exist_ok=True)

if not VOICES_JSON.exists():
    with open(VOICES_JSON, "w", encoding="utf-8") as f:
        json.dump({}, f, ensure_ascii=False, indent=2)

# Designed voices storage
DESIGNED_VOICES_DIR = WORKSPACE / "designed_voices"
DESIGNED_VOICES_JSON = DESIGNED_VOICES_DIR / "voices.json"
DESIGNED_VOICES_DIR.mkdir(parents=True, exist_ok=True)

if not DESIGNED_VOICES_JSON.exists():
    with open(DESIGNED_VOICES_JSON, "w", encoding="utf-8") as f:
        json.dump({}, f, ensure_ascii=False, indent=2)

def get_saved_voices():
    """Get list of saved voices."""
    try:
        with open(VOICES_JSON, "r", encoding="utf-8") as f:
            voices = json.load(f)
        return sorted(list(voices.keys()))
    except Exception:
        return []

def save_voice(name, audio, ref_text, xvector_only, model_size, chunk_size, chunk_gap):
    """Save a voice to storage."""
    if not name or not name.strip():
        return "Ошибка: Имя голоса обязательно.", gr.update(choices=get_saved_voices())
    
    name = name.strip()
    audio_tuple = _audio_to_tuple(audio)
    if audio_tuple is None:
        return "Ошибка: Эталонное аудио обязательно.", gr.update(choices=get_saved_voices())
    
    try:
        # Save audio file
        import scipy.io.wavfile as wavfile
        sr, wav = audio_tuple[1], audio_tuple[0]
        # Convert back to int16 for wav storage if needed, or keep float32
        # Most practical is int16 for compatibility
        wav_int16 = (wav * 32767).astype(np.int16)
        audio_path = VOICES_DIR / f"{name}.wav"
        wavfile.write(audio_path, sr, wav_int16)
        
        # Save metadata
        with open(VOICES_JSON, "r", encoding="utf-8") as f:
            voices = json.load(f)
            
        voices[name] = {
            "ref_audio_path": str(audio_path),
            "ref_text": ref_text,
            "xvector_only": xvector_only,
            "model_size": model_size,
            "chunk_size": chunk_size,
            "chunk_gap": chunk_gap
        }
        
        with open(VOICES_JSON, "w", encoding="utf-8") as f:
            json.dump(voices, f, ensure_ascii=False, indent=2)
            
        return f"✅ Голос '{name}' успешно сохранен!", gr.update(choices=get_saved_voices(), value=name)
    except Exception as e:
        return f"❌ Ошибка сохранения: {str(e)}", gr.update(choices=get_saved_voices())

def load_voice_data(name):
    """Load voice data for the UI."""
    if not name:
        return None, "", False, "1.7B", 200, 0.0
    
    try:
        with open(VOICES_JSON, "r", encoding="utf-8") as f:
            voices = json.load(f)
            
        if name not in voices:
            return None, "", False, "1.7B", 200, 0.0
            
        v = voices[name]
        audio_path = v.get("ref_audio_path")
        
        # Resilient path resolution: 
        # If path doesn't exist, try resolving it as a filename in local VOICES_DIR
        if audio_path and not os.path.exists(audio_path):
            filename = os.path.basename(audio_path)
            local_path = VOICES_DIR / filename
            if local_path.exists():
                audio_path = str(local_path)
            
        ref_text = v.get("ref_text", "")
        xvector_only = v.get("xvector_only", False)
        model_size = v.get("model_size", "1.7B")
        chunk_size = v.get("chunk_size", 200)
        chunk_gap = v.get("chunk_gap", 0.0)
        
        return audio_path, ref_text, xvector_only, model_size, chunk_size, chunk_gap
    except Exception as e:
        print(f"Error loading voice data: {e}")
        return None, "", False, "1.7B", 200, 0.0

def delete_voice(name):
    """Delete a saved voice."""
    if not name:
        return "Ошибка: Выберите голос для удаления.", gr.update(choices=get_saved_voices())
    
    try:
        with open(VOICES_JSON, "r", encoding="utf-8") as f:
            voices = json.load(f)
            
        if name in voices:
            audio_path = Path(voices[name]["ref_audio_path"])
            if audio_path.exists():
                audio_path.unlink()
            del voices[name]
            
            with open(VOICES_JSON, "w", encoding="utf-8") as f:
                json.dump(voices, f, ensure_ascii=False, indent=2)
                
            return f"✅ Голос '{name}' удален.", gr.update(choices=get_saved_voices(), value=None)
        return f"⚠️ Голос '{name}' не найден.", gr.update(choices=get_saved_voices())
    except Exception as e:
        return f"❌ Ошибка удаления: {str(e)}", gr.update(choices=get_saved_voices())


# --- Designed Voices Logic ---

def get_saved_designed_voices():
    """Get list of saved designed voices."""
    try:
        with open(DESIGNED_VOICES_JSON, "r", encoding="utf-8") as f:
            voices = json.load(f)
        return sorted(list(voices.keys()))
    except Exception:
        return []

def save_designed_voice(name, instruct, seed):
    """Save a designed voice metadata."""
    if not name or not name.strip():
        return "Ошибка: Имя обязательно.", gr.update(choices=get_saved_designed_voices())
    if not instruct or not instruct.strip():
        return "Ошибка: Описание обязательно.", gr.update(choices=get_saved_designed_voices())
    
    name = name.strip()
    try:
        with open(DESIGNED_VOICES_JSON, "r", encoding="utf-8") as f:
            voices = json.load(f)
            
        voices[name] = {
            "instruct": instruct.strip(),
            "seed": int(seed)
        }
        
        with open(DESIGNED_VOICES_JSON, "w", encoding="utf-8") as f:
            json.dump(voices, f, ensure_ascii=False, indent=2)
            
        return f"✅ Дизайн '{name}' успешно сохранен!", gr.update(choices=get_saved_designed_voices(), value=name)
    except Exception as e:
        return f"❌ Ошибка сохранения: {str(e)}", gr.update(choices=get_saved_designed_voices())

def load_designed_voice_data(name):
    """Load designed voice data for the UI."""
    if not name:
        return "", -1
    
    try:
        with open(DESIGNED_VOICES_JSON, "r", encoding="utf-8") as f:
            voices = json.load(f)
            
        if name not in voices:
            return "", -1
            
        v = voices[name]
        return v.get("instruct", ""), v.get("seed", -1)
    except Exception:
        return "", -1

def delete_designed_voice(name):
    """Delete a saved designed voice."""
    if not name:
        return "Ошибка: Выберите дизайн для удаления.", gr.update(choices=get_saved_designed_voices())
        
    try:
        with open(DESIGNED_VOICES_JSON, "r", encoding="utf-8") as f:
            voices = json.load(f)
            
        if name in voices:
            del voices[name]
            with open(DESIGNED_VOICES_JSON, "w", encoding="utf-8") as f:
                json.dump(voices, f, ensure_ascii=False, indent=2)
            return f"✅ Дизайн '{name}' удален.", gr.update(choices=get_saved_designed_voices(), value=None)
        return f"⚠️ Дизайн '{name}' не найден.", gr.update(choices=get_saved_designed_voices())
    except Exception as e:
        return f"❌ Ошибка удаления: {str(e)}", gr.update(choices=get_saved_designed_voices())


def get_whisper_model():
    """Load Whisper tiny model for transcription."""
    global whisper_model
    if whisper_model is None:
        import whisper
        whisper_model = whisper.load_model("tiny", device="cuda" if torch.cuda.is_available() else "cpu")
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
        result = model.transcribe(wav, fp16=torch.cuda.is_available())
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
    """Get or load a model by type and size with robust fallback logic."""
    global loaded_models
    key = (model_type, model_size)
    if key not in loaded_models:
        from qwen_tts import Qwen3TTSModel
        model_path = get_model_path(model_type, model_size)
        
        # Determine best settings for the current hardware
        if torch.cuda.is_available():
            device_str = "cuda"
            major, _ = torch.cuda.get_device_capability()
            if major >= 8: # Ampere+
                dtype = torch.bfloat16
                attn_impl = "flash_attention_2"
            else: # Turing/Pascal Fallback
                dtype = torch.float16
                attn_impl = "sdpa"
        else:
            device_str = "cpu"
            dtype = torch.float32
            attn_impl = "eager"

        try:
            print(f"📥 Loading {model_type} ({model_size}) on {device_str} with {attn_impl} / {dtype}...")
            loaded_models[key] = Qwen3TTSModel.from_pretrained(
                model_path,
                device_map=device_str,
                dtype=dtype,
                attn_implementation=attn_impl,
            )
        except Exception as e:
            print(f"⚠️ Failed to load with {attn_impl}. Trying 'eager' fallback. Error: {e}")
            loaded_models[key] = Qwen3TTSModel.from_pretrained(
                model_path,
                device_map=device_str,
                dtype=torch.float32 if device_str == "cpu" else torch.float16,
                attn_implementation="eager",
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
        torch.cuda.empty_cache()
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
        torch.cuda.empty_cache()
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

    # Handle string path (from type="filepath")
    if isinstance(audio, str):
        try:
            import librosa
            wav, sr = librosa.load(audio, sr=None)
            wav = _normalize_audio(wav)
            return wav, int(sr)
        except Exception as e:
            print(f"Error loading audio with librosa: {e}")
            try:
                # Fallback to wavfile just in case
                sr, wav = wavfile.read(audio)
                wav = _normalize_audio(wav)
                return wav, int(sr)
            except Exception as e2:
                print(f"Error loading audio from path {audio}: {e2}")
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
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # Force deterministic algorithms
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def generate_voice_design(text, language, voice_description, seed, num_variants=1):
    """Generate speech using Voice Design model (1.7B only)."""
    if not text or not text.strip():
        return [None] * 5 + ["Error: Text is required."]
    if not voice_description or not voice_description.strip():
        return [None] * 5 + ["Error: Voice description is required."]

    results = []
    seeds = []
    try:
        num_variants = int(num_variants)
        base_seed = int(seed)
        
        tts = get_model("VoiceDesign", "1.7B")
        
        print(f"\n{'='*50}")
        print(f"🎨 Voice Design Generation ({num_variants} variants)")
        print(f"{'='*50}")
        
        for i in range(min(num_variants, 5)):
            # Handle seed - if -1 (auto), generate one. If fixed, increment.
            current_seed = base_seed
            if current_seed == -1:
                current_seed = random.randint(0, 2147483647)
            else:
                current_seed = base_seed + i
            
            seeds.append(current_seed)
            set_seed(current_seed)
            
            print(f"🎲 Variant {i+1}/{num_variants} [Seed: {current_seed}]")
            
            wavs, sr = tts.generate_voice_design(
                text=text.strip(),
                language=language,
                instruct=voice_description.strip(),
                non_streaming_mode=True,
                max_new_tokens=2048,
            )
            results.append((sr, wavs[0]))
            
        # Pad results to 5
        audio_outputs = results + [None] * (5 - len(results))
        
        duration_info = ", ".join([f"{len(r[1])/r[0]:.1f}s" for r in results])
        status = f"Сгенерировано {len(results)} вар. | Seeds: {seeds} | Длит: {duration_info}"
        
        print(f"\n{'='*50}")
        print(f"✅ Готово! Сгенерировано {len(results)} вариантов.")
        print(f"{'='*50}\n")
        
        return audio_outputs + [status]
    except Exception as e:
        print(f"❌ Ошибка: {type(e).__name__}: {e}")
        return [None] * 5 + [f"Ошибка: {type(e).__name__}: {e}"]


def generate_voice_clone(ref_audio, ref_text, target_text, language, use_xvector_only, model_size, max_chunk_chars, chunk_gap, seed, num_variants=1):
    """Generate speech using Base (Voice Clone) model."""
    if not target_text or not target_text.strip():
        return [None] * 5 + ["Ошибка: Текст обязателен."]

    audio_tuple = _audio_to_tuple(ref_audio)
    if audio_tuple is None:
        return [None] * 5 + ["Ошибка: Эталонное аудио обязательно."]

    if not use_xvector_only and (not ref_text or not ref_text.strip()):
        return [None] * 5 + ["Ошибка: Эталонный текст обязателен, если не включен 'Только x-vector'."]

    try:
        from tqdm import tqdm
        num_variants = int(num_variants)
        base_seed = int(seed)
        
        tts = get_model("Base", model_size)
        chunks = chunk_text(target_text.strip(), max_chars=int(max_chunk_chars))
        
        print(f"\n{'='*50}")
        print(f"🎭 Voice Clone Generation ({model_size}, {num_variants} variants)")
        print(f"{'='*50}")
        print(f"📝 Text length: {len(target_text)} chars → {len(chunks)} chunk(s)")
        print(f"⏱️ Chunk gap: {chunk_gap}s")
        
        results = []
        all_seeds = []
        
        for v in range(min(num_variants, 5)):
            # Handle seed
            current_seed = base_seed
            if current_seed == -1:
                current_seed = random.randint(0, 2147483647)
            else:
                current_seed = base_seed + v
            
            all_seeds.append(current_seed)
            print(f"\n✨ Variant {v+1}/{num_variants} [Seed: {current_seed}]")
            
            variant_wavs = []
            sr = None
            for i, chunk in enumerate(tqdm(chunks, desc=f"Variant {v+1} Chunks", unit="chunk")):
                set_seed(current_seed)
                wavs, sr = tts.generate_voice_clone(
                    text=chunk,
                    language=language,
                    ref_audio=audio_tuple,
                    ref_text=ref_text.strip() if ref_text else None,
                    x_vector_only_mode=use_xvector_only,
                    max_new_tokens=2048,
                )
                variant_wavs.append(wavs[0])
            
            # Concatenate chunks
            if len(variant_wavs) > 1 and chunk_gap > 0:
                gap_samples = int(sr * chunk_gap)
                silence = np.zeros(gap_samples, dtype=np.float32)
                chunks_with_gaps = []
                for i, wav in enumerate(variant_wavs):
                    chunks_with_gaps.append(wav)
                    if i < len(variant_wavs) - 1:
                        chunks_with_gaps.append(silence)
                final_wav = np.concatenate(chunks_with_gaps)
            else:
                final_wav = np.concatenate(variant_wavs) if len(variant_wavs) > 1 else variant_wavs[0]
            
            results.append((sr, final_wav))
            
        # Pad results
        audio_outputs = results + [None] * (5 - len(results))
        
        total_info = f"Сгенерировано {len(results)} вар., {len(chunks)} частей | Seeds: {all_seeds}"
        print(f"\n{'='*50}")
        print(f"✅ Готово! {total_info}")
        print(f"{'='*50}\n")
        
        return audio_outputs + [total_info]
    except Exception as e:
        print(f"❌ Ошибка: {type(e).__name__}: {e}")
        return [None] * 5 + [f"Ошибка: {type(e).__name__}: {e}"]


def generate_custom_voice(text, language, speaker, instruct, model_size, seed, num_variants=1):
    """Generate speech using CustomVoice model."""
    if not text or not text.strip():
        return [None] * 5 + ["Ошибка: Текст обязателен."]
    if not speaker:
        return [None] * 5 + ["Ошибка: Диктор обязателен."]

    results = []
    seeds = []
    try:
        num_variants = int(num_variants)
        base_seed = int(seed)
        
        tts = get_model("CustomVoice", model_size)
        
        print(f"\n{'='*50}")
        print(f"🗣️ Custom Voice Generation ({model_size}, {num_variants} variants)")
        print(f"{'='*50}")
        print(f"👤 Диктор: {speaker}")
        
        for i in range(min(num_variants, 5)):
            current_seed = base_seed
            if current_seed == -1:
                current_seed = random.randint(0, 2147483647)
            else:
                current_seed = base_seed + i
            
            seeds.append(current_seed)
            set_seed(current_seed)
            
            print(f"🎲 Variant {i+1}/{num_variants} [Seed: {current_seed}]")
            
            wavs, sr = tts.generate_custom_voice(
                text=text.strip(),
                language=language,
                speaker=speaker.lower().replace(" ", "_"),
                instruct=instruct.strip() if instruct else None,
                non_streaming_mode=True,
                max_new_tokens=2048,
            )
            results.append((sr, wavs[0]))
            
        # Pad results
        audio_outputs = results + [None] * (5 - len(results))
        
        duration_info = ", ".join([f"{len(r[1])/r[0]:.1f}s" for r in results])
        status = f"Сгенерировано {len(results)} вар. | Seeds: {seeds} | Длит: {duration_info}"
        
        print(f"\n{'='*50}")
        print(f"✅ Готово! Сгенерировано {len(results)} вариантов.")
        print(f"{'='*50}\n")
        
        return audio_outputs + [status]
    except Exception as e:
        print(f"❌ Ошибка: {type(e).__name__}: {e}")
        return [None] * 5 + [f"Ошибка: {type(e).__name__}: {e}"]


# Build Gradio UI
def build_ui():
    theme = gr.themes.Soft(
        font=[gr.themes.GoogleFont("Inter"), "Arial", "sans-serif"],
        primary_hue="indigo",
        secondary_hue="slate",
    )

    css = """
    .gradio-container {
        max-width: 1200px !important;
        margin: auto !important;
        padding: 0 1rem !important;
    }
    .header-container {
        text-align: center;
        padding: 3rem 1.5rem;
        background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 50%, #9333ea 100%);
        border-radius: 24px;
        margin-bottom: 2rem;
        box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.1), 0 8px 10px -6px rgba(0, 0, 0, 0.1);
        position: relative;
        overflow: hidden;
    }
    .header-container::before {
        content: '';
        position: absolute;
        top: -50%;
        left: -50%;
        width: 200%;
        height: 200%;
        background: radial-gradient(circle, rgba(255,255,255,0.1) 0%, transparent 80%);
        pointer-events: none;
    }
    .header-container h1 {
        color: white !important;
        font-size: 3rem !important;
        font-weight: 800 !important;
        margin-bottom: 0.75rem !important;
        letter-spacing: -0.025em !important;
        text-shadow: 0 2px 4px rgba(0,0,0,0.2);
    }
    .header-container p {
        color: rgba(255,255,255,0.95) !important;
        font-size: 1.25rem !important;
        max-width: 700px;
        margin: 0 auto !important;
        line-height: 1.6;
    }
    .feature-badge {
        display: inline-flex;
        align-items: center;
        background: rgba(255,255,255,0.15);
        backdrop-filter: blur(8px);
        -webkit-backdrop-filter: blur(8px);
        padding: 0.4rem 1rem;
        border-radius: 9999px;
        margin: 0.35rem;
        font-size: 0.9rem;
        font-weight: 600;
        color: white;
        border: 1px solid rgba(255,255,255,0.2);
        transition: all 0.2s ease;
    }
    .feature-badge:hover {
        background: rgba(255,255,255,0.25);
        transform: translateY(-1px);
    }
    footer {
        display: none !important;
    }
    .custom-footer {
        text-align: center;
        margin-top: 3rem;
        padding: 2rem 1rem;
        border-top: 1px solid rgba(229, 231, 235, 1);
        color: #4b5563;
        font-size: 0.95rem;
    }
    .custom-footer a {
        color: #4f46e5;
        text-decoration: none;
        font-weight: 600;
        transition: color 0.15s ease;
    }
    .custom-footer a:hover {
        color: #4338ca;
        text-decoration: underline;
    }
    @media (max-width: 768px) {
        .header-container {
            padding: 2rem 1rem;
            border-radius: 16px;
        }
        .header-container h1 {
            font-size: 2rem !important;
        }
        .header-container p {
            font-size: 1rem !important;
        }
        .gradio-container {
            padding: 0 0.5rem !important;
        }
        .feature-badge {
            font-size: 0.8rem;
            padding: 0.3rem 0.75rem;
        }
    }
    """

    with gr.Blocks(title="Qwen3-TTS Demo", theme=theme, css=css) as demo:
        gr.HTML(
            """
            <div class="header-container">
                <h1>🎙️ Qwen3-TTS</h1>
                <p>Высококачественный синтез речи с функциями клонирования и дизайна голоса</p>
                <div style="margin-top: 1.5rem; display: flex; justify-content: center; flex-wrap: wrap;">
                    <div class="feature-badge">🎨 Дизайн голоса</div>
                    <div class="feature-badge">🎭 Клонирование</div>
                    <div class="feature-badge">🗣️ Свои голоса</div>
                    <div class="feature-badge">📝 Длинные тексты</div>
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
                        with gr.Accordion("💾 Сохраненные дизайны", open=False):
                            with gr.Row():
                                saved_designs_dropdown = gr.Dropdown(
                                    label="Выберите дизайн",
                                    choices=get_saved_designed_voices(),
                                    value=None,
                                    scale=3
                                )
                                refresh_designs_btn = gr.Button("🔄", scale=1)
                            
                            save_design_name = gr.Textbox(
                                label="Имя для сохранения",
                                placeholder="Введите имя для этого дизайна...",
                            )
                            with gr.Row():
                                save_design_btn = gr.Button("💾 Сохранить дизайн", variant="secondary")
                                delete_design_btn = gr.Button("🗑️ Удалить выбранный", variant="stop")
                            design_save_status = gr.Textbox(label="Статус сохранения", lines=1, interactive=False)

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
                        with gr.Row():
                            design_num_variants = gr.Slider(
                                label="Количество вариантов",
                                minimum=1,
                                maximum=5,
                                value=1,
                                step=1,
                            )
                        design_btn = gr.Button("🎙️ Генерировать", variant="primary", size="lg")

                    with gr.Column(scale=1):
                        design_audio_outputs = []
                        for i in range(5):
                            design_audio_outputs.append(gr.Audio(label=f"Результат {i+1}", type="numpy", visible=(i==0)))
                        design_status = gr.Textbox(label="Статус", lines=2, interactive=False)

                def update_audio_visibility(num):
                    num = int(num)
                    return [gr.update(visible=True) if i < num else gr.update(visible=False, value=None) for i in range(5)]

                design_num_variants.change(
                    update_audio_visibility,
                    inputs=[design_num_variants],
                    outputs=design_audio_outputs
                )

                design_btn.click(
                    generate_voice_design,
                    inputs=[design_text, design_language, design_instruct, design_seed, design_num_variants],
                    outputs=design_audio_outputs + [design_status],
                )

                # Voice Design saved events
                refresh_designs_btn.click(
                    lambda: gr.update(choices=get_saved_designed_voices()),
                    inputs=[],
                    outputs=[saved_designs_dropdown]
                )

                save_design_btn.click(
                    save_designed_voice,
                    inputs=[save_design_name, design_instruct, design_seed],
                    outputs=[design_save_status, saved_designs_dropdown]
                )

                delete_design_btn.click(
                    delete_designed_voice,
                    inputs=[saved_designs_dropdown],
                    outputs=[design_save_status, saved_designs_dropdown]
                )

                saved_designs_dropdown.change(
                    load_designed_voice_data,
                    inputs=[saved_designs_dropdown],
                    outputs=[design_instruct, design_seed]
                )

            # Tab 2: Voice Clone
            with gr.Tab("🎭 Клонирование голоса"):
                with gr.Row(equal_height=True):
                    with gr.Column(scale=1):
                        with gr.Accordion("💾 Сохраненные голоса", open=True):
                            with gr.Row():
                                saved_voices_dropdown = gr.Dropdown(
                                    label="Выберите голос",
                                    choices=get_saved_voices(),
                                    value=None,
                                    scale=3
                                )
                                refresh_voices_btn = gr.Button("🔄", scale=1)
                            
                            save_voice_name = gr.Textbox(
                                label="Имя для сохранения",
                                placeholder="Введите имя нового голоса...",
                            )
                            with gr.Row():
                                save_voice_btn = gr.Button("💾 Сохранить текущий", variant="secondary")
                                delete_voice_btn = gr.Button("🗑️ Удалить выбранный", variant="stop")
                            save_load_status = gr.Textbox(label="Статус сохранения/загрузки", lines=1, interactive=False)

                        clone_ref_audio = gr.Audio(
                            label="Эталонное аудио",
                            type="filepath",
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
                            clone_num_variants = gr.Slider(
                                label="Кол-во вариантов",
                                minimum=1,
                                maximum=5,
                                value=1,
                                step=1,
                            )
                        clone_btn = gr.Button("🎙️ Клонировать и озвучить", variant="primary", size="lg")

                    with gr.Column(scale=1):
                        clone_audio_outputs = []
                        for i in range(5):
                            clone_audio_outputs.append(gr.Audio(label=f"Результат {i+1}", type="numpy", visible=(i==0)))
                        clone_status = gr.Textbox(label="Статус", lines=2, interactive=False)

                clone_num_variants.change(
                    update_audio_visibility,
                    inputs=[clone_num_variants],
                    outputs=clone_audio_outputs
                )

                transcribe_btn.click(
                    transcribe_audio,
                    inputs=[clone_ref_audio],
                    outputs=[clone_ref_text],
                )
                
                clone_btn.click(
                    generate_voice_clone,
                    inputs=[clone_ref_audio, clone_ref_text, clone_target_text, clone_language, clone_xvector, clone_model_size, clone_chunk_size, clone_chunk_gap, clone_seed, clone_num_variants],
                    outputs=clone_audio_outputs + [clone_status],
                )

                # Saved voices events
                refresh_voices_btn.click(
                    lambda: gr.update(choices=get_saved_voices()),
                    inputs=[],
                    outputs=[saved_voices_dropdown]
                )

                save_voice_btn.click(
                    save_voice,
                    inputs=[save_voice_name, clone_ref_audio, clone_ref_text, clone_xvector, clone_model_size, clone_chunk_size, clone_chunk_gap],
                    outputs=[save_load_status, saved_voices_dropdown]
                )

                delete_voice_btn.click(
                    delete_voice,
                    inputs=[saved_voices_dropdown],
                    outputs=[save_load_status, saved_voices_dropdown]
                )

                saved_voices_dropdown.change(
                    load_voice_data,
                    inputs=[saved_voices_dropdown],
                    outputs=[clone_ref_audio, clone_ref_text, clone_xvector, clone_model_size, clone_chunk_size, clone_chunk_gap]
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
                        with gr.Row():
                            tts_num_variants = gr.Slider(
                                label="Количество вариантов",
                                minimum=1,
                                maximum=5,
                                value=1,
                                step=1,
                            )
                        tts_btn = gr.Button("🎙️ Генерировать", variant="primary", size="lg")

                    with gr.Column(scale=1):
                        tts_audio_outputs = []
                        for i in range(5):
                            tts_audio_outputs.append(gr.Audio(label=f"Результат {i+1}", type="numpy", visible=(i==0)))
                        tts_status = gr.Textbox(label="Статус", lines=2, interactive=False)

                tts_num_variants.change(
                    update_audio_visibility,
                    inputs=[tts_num_variants],
                    outputs=tts_audio_outputs
                )

                tts_btn.click(
                    generate_custom_voice,
                    inputs=[tts_text, tts_language, tts_speaker, tts_instruct, tts_model_size, tts_seed, tts_num_variants],
                    outputs=tts_audio_outputs + [tts_status],
                )

            # Tab 4: Voice Database
            with gr.Tab("📚 База дизайнов голосов"):
                gr.Markdown("### 🔍 Исследуйте библиотеку готовых стилей голосов")
                
                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown("#### 📁 Категории")
                        with gr.Row(variant="compact"):
                            # Create category buttons
                            cat_btns = []
                            for category in VOICE_DATABASE.keys():
                                cat_btns.append(gr.Button(category, size="sm", variant="secondary"))
                                
                        gr.Markdown("#### 🎙️ Голоса")
                        voice_list = gr.Dropdown(
                            label="Выберите голос из категории",
                            choices=[],
                            interactive=True
                        )
                    
                    with gr.Column(scale=1):
                        gr.Markdown("#### 📄 Детали голоса")
                        db_voice_en = gr.Code(label="EN Instruct (для модели)", language="markdown", interactive=False)
                        db_voice_ru = gr.Textbox(label="Описание на русском", interactive=False)
                        
                        apply_to_design_btn = gr.Button("🎨 Использовать в дизайне", variant="primary")
                        apply_status = gr.Markdown()

                # Database Events
                def select_category(cat):
                    voices = list(VOICE_DATABASE.get(cat, {}).keys())
                    return gr.update(choices=voices, value=voices[0] if voices else None)

                for btn in cat_btns:
                    btn.click(
                        select_category,
                        inputs=[btn],
                        outputs=[voice_list]
                    )

                def load_voice_details(cat, voice):
                    if not cat or not voice:
                        return "", ""
                    details = VOICE_DATABASE.get(cat, {}).get(voice, {})
                    return details.get("en", ""), details.get("ru", "")

                # We need to track the current category. Let's use a hidden state or just take it from the button click?
                # Actually, when a button is clicked, it updates the dropdown. The dropdown change will trigger the details update.
                # To get the category in the details update, we can use another state.
                current_db_category = gr.State("")
                
                for btn in cat_btns:
                    btn.click(lambda c: c, inputs=[btn], outputs=[current_db_category])

                voice_list.change(
                    load_voice_details,
                    inputs=[current_db_category, voice_list],
                    outputs=[db_voice_en, db_voice_ru]
                )

                def apply_to_design(instruct):
                    return instruct, "✅ Инструкция скопирована во вкладку 'Дизайн голоса'!"

                apply_to_design_btn.click(
                    apply_to_design,
                    inputs=[db_voice_en],
                    outputs=[design_instruct, apply_status]
                )


        gr.HTML(
            """
            <div class="custom-footer">
                Powered by <a href="https://github.com/QwenLM/Qwen3-TTS" target="_blank">Qwen3-TTS</a> | 
                Модифицированно <a href="https://www.youtube.com/channel/UCLoDL_MJpkrMizBuuXnRYsg" target="_blank">Максимом Юровских</a>
            </div>
            """
        )

        # Refresh dropdowns on page load
        demo.load(
            lambda: [gr.update(choices=get_saved_voices()), gr.update(choices=get_saved_designed_voices())],
            inputs=None,
            outputs=[saved_voices_dropdown, saved_designs_dropdown]
        )

    return demo


if __name__ == "__main__":
    # Fix for 503 error: disable proxy for localhost
    os.environ["NO_PROXY"] = "127.0.0.1,localhost"
    os.environ["PYTHONUNBUFFERED"] = "1"

    server_name = os.environ.get("GRADIO_SERVER_NAME", "0.0.0.0")
    server_port = int(os.environ.get("GRADIO_SERVER_PORT", os.environ.get("QWEN_TTS_PORT", "8000")))

    demo = build_ui()
    demo.launch(
        server_name=server_name,
        server_port=server_port,
        allowed_paths=[
            str(VOICES_DIR.absolute()),
            str(DESIGNED_VOICES_DIR.absolute()),
        ],
    )
