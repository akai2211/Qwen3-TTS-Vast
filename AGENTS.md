# Qwen TTS на Vast.ai — контекст для Cursor

> **Назначение файла:** полный handoff для нового проекта **Qwen TTS** — озвучка (text-to-speech) на GPU через [Vast.ai](https://vast.ai).  
> **Пользователь:** akai2211 — общается **по-русски**, уверенно в IT, читает логи.  
> **Связанный опыт:** образ [sd_forge_max](https://github.com/akai2211/sd_forge_max) (Forge на Vast) — оттуда перенесены проверенные паттерны SSH, портов, `/workspace`, Launch Mode.  
> **Этот файл не коммитить** в sd_forge_max — переносится в отдельный репозиторий проекта.

---

## Цель проекта

Собрать **удобный для Vast.ai** способ запуска **Qwen3-TTS** (озвучка текста):

- синтез речи (CustomVoice, VoiceDesign, Voice Clone);
- Web UI и/или HTTP API (OpenAI-compatible — опционально);
- persistent storage моделей и выходных аудио на `/workspace`;
- SSH для отладки и пакетной обработки;
- шаблон Vast с понятным доступом (Open Ports / Portal).

**Не путать с:** SD Forge / image generation — это **отдельный** проект, другой Docker-образ и порты.

---

## Qwen3-TTS — что это (озвучка)

Официальная open-source линейка TTS от команды Qwen (Alibaba Cloud), релиз **январь 2026**, лицензия **Apache 2.0**.

### Возможности

| Режим | Модель | Описание |
|---|---|---|
| **CustomVoice** | `Qwen3-TTS-12Hz-{0.6B,1.7B}-CustomVoice` | 9 preset-голосов + управление стилем через `instruct` |
| **Voice Design** | `Qwen3-TTS-12Hz-1.7B-VoiceDesign` | Голос из текстового описания |
| **Voice Clone** | `Qwen3-TTS-12Hz-{0.6B,1.7B}-Base` | Клон из ~3 с reference audio + transcript |
| **Streaming** | все 12Hz модели | latency до первого аудио-пакета ~97 ms (по заявлению upstream) |
| **Языки** | 10 | Chinese, English, Japanese, Korean, German, French, Russian, Portuguese, Spanish, Italian |

### Модели на Hugging Face

| Модель | HF id |
|---|---|
| Tokenizer | `Qwen/Qwen3-TTS-Tokenizer-12Hz` |
| CustomVoice 1.7B | `Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice` |
| VoiceDesign 1.7B | `Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign` |
| Base (clone) 1.7B | `Qwen/Qwen3-TTS-12Hz-1.7B-Base` |
| CustomVoice 0.6B | `Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice` |
| Base (clone) 0.6B | `Qwen/Qwen3-TTS-12Hz-0.6B-Base` |

Коллекция: <https://huggingface.co/collections/Qwen/qwen3-tts>

### VRAM (ориентиры)

| Модель | VRAM (ориентир) | Примечание |
|---|---|---|
| 0.6B | ~4–6 GB | легче для дешёвых GPU |
| 1.7B | ~6–8 GB | community-серверы: ~4.4 GB bfloat16 + flash-attn на RTX 4080 SUPER |
| 1.7B + ASR рядом | до ~28 GB | бенчмарк на RTX 5090 с co-located ASR |

Рекомендуемый минимум на Vast: **8 GB** (0.6B / лёгкие сценарии), **12–16 GB** (1.7B комфортно), **24 GB+** если TTS + ASR на одной карте.

### Установка (upstream)

```bash
# Python 3.12 — рекомендация upstream
conda create -n qwen3-tts python=3.12 -y
conda activate qwen3-tts
pip install -U qwen-tts

# FlashAttention 2 — меньше VRAM (нужен GPU + float16/bfloat16)
pip install -U flash-attn --no-build-isolation
```

Или из исходников: <https://github.com/QwenLM/Qwen3-TTS>

### Локальный Web UI (официальный demo)

```bash
qwen-tts-demo Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice --ip 0.0.0.0 --port 8000
# → http://<IP>:8000
```

Варианты моделей: `CustomVoice`, `VoiceDesign`, `Base` — см. `--help`.

### Preset-спикеры (CustomVoice)

| Speaker | Описание | Родной язык |
|---|---|---|
| Vivian, Serena, Uncle_Fu, Dylan, Eric | китайские профили | Chinese / диалекты |
| Ryan, Aiden | английские | English |
| Ono_Anna | японский | Japanese |
| Sohee | корейский | Korean |

### Облачный API (без self-host)

DashScope / Model Studio — managed Qwen TTS (realtime, clone, design):

- Realtime: <https://www.alibabacloud.com/help/en/model-studio/qwen-tts-realtime>
- Voice cloning: <https://www.alibabacloud.com/help/en/model-studio/qwen-tts-voice-cloning>
- Voice design: <https://www.alibabacloud.com/help/en/model-studio/qwen-tts-voice-design>

Для проекта на Vast фокус — **self-hosted**; API — запасной вариант без GPU.

### vLLM

Upstream: day-0 поддержка через **vLLM-Omni** — пока в основном offline inference; online serving в разработке.  
Документация: <https://docs.vllm.ai/projects/vllm-omni/en/latest/getting_started/quickstart/#installation>

---

## Готовые Docker-решения для Qwen TTS (community)

Официального Docker-образа от Qwen **нет** — смотреть community или собирать свой.

| Репозиторий / образ | Что даёт | Порт | CUDA | Заметки |
|---|---|---|---|---|
| [malaiwah/qwen3-tts-server](https://github.com/malaiwah/qwen3-tts-server) | OpenAI-compatible API, streaming PCM, 9 голосов, clone | **8001** | **12.8**, cu128 | `ghcr.io/malaiwah/qwen3-tts-server:latest`; **протестирован на RTX 5090 на Vast** |
| [mosstnslv/qwen-tts-service](https://hub.docker.com/r/mosstnslv/qwen-tts-service) | FastAPI, stream, clone, design | **8188** | CUDA + flash_attn_2 | веса монтировать в `/app/models` |
| [solotrode/EchoFleet-Qwen3-TTS](https://github.com/solotrode/EchoFleet-Qwen3-TTS) | FastAPI + Gradio + Redis workers, multi-GPU | API **8000**, UI **7860** | torch **2.7.0+cu128** | стек для production-like нагрузки |

### malaiwah/qwen3-tts-server — детали для Vast

Полезен как **референс** (уже валидирован на Vast 5090):

```bash
docker run -d --name qwen3-tts \
  --gpus all \
  -p 8001:8001 \
  -v qwen3-hf-cache:/root/.cache/huggingface \
  ghcr.io/malaiwah/qwen3-tts-server:latest
```

- `POST /v1/audio/speech` — OpenAI-совместимый синтез
- `POST /v1/audio/speech/pcm-stream` — streaming PCM
- Env: `QWEN3_TTS_MODEL`, `QWEN3_TTS_ASR_URL` (опционально для транскрипции reference)
- VRAM ~4.4 GB (tier-1, bfloat16)
- На **RTX 5090** (Blackwell): нужен **cu128** образ — этот сервер уже на CUDA 12.8

---

## Vast.ai — платформа (обязательно знать)

Документация (индекс для AI): <https://docs.vast.ai/llms.txt>

### Как устроен инстанс

1. Выбираешь **offer** (GPU + цена).
2. Шаблон (**template**) задаёт Docker-образ, порты, env, Launch Mode.
3. Vast создаёт контейнер на хосте; **внешние порты** случайные → смотри **Open Ports** на карточке.
4. **`/workspace`** — persistent volume (переживает stop/start и смену образа **на том же инстансе**).
5. Оплата **почасовая** — Destroy когда не нужен.

### Launch Mode — критично

| Режим | Поведение |
|---|---|
| **Docker ENTRYPOINT** | Запускается **ENTRYPOINT/CMD образа** — наш сценарий для кастомного Docker |
| **Jupyter + SSH** | Vast **подменяет** entrypoint → Jupyter :8080 + SSH :22; сервисы из образа **сами не стартуют** |
| **Interactive shell, SSH** | Только SSH, entrypoint образа **не выполняется** |

**Для кастомного Qwen-образа:** предпочтительно **Docker ENTRYPOINT** + openssh внутри образа (как sd_forge_max).  
Jupyter mode имеет смысл только для **прототипа** через `PROVISIONING_SCRIPT` / on-start script — не для production-образа.

Источник: [Template Settings — Launch Mode](https://docs.vast.ai/guides/templates/template-settings)

### Порты и доступ

**Open Ports** на карточке: `IP:ВНЕШНИЙ -> ВНУТРЕННИЙ/tcp`

Пример для TTS (если сервис слушает **8000** внутри):

```
http://<IP>:<VAST_TCP_PORT_8000>/
ssh -p <VAST_TCP_PORT_22> -i ~/.ssh/id_ed25519 root@<IP>
ssh -p <VAST_TCP_PORT_22> -i ~/.ssh/id_ed25519 root@<IP> -L 8000:localhost:8000
```

**Правило для AI:** если пользователь кидает блок Open Ports — **сразу** выдать отдельными строками ссылки SSH / TTS / localhost tunnel.

### OPEN_BUTTON_PORT и кнопка Open

- Переменная **`OPEN_BUTTON_PORT`** — какой **внутренний** порт Vast считает «главным» для кнопки **Open**.
- Задаётся **только в шаблоне** (Docker Options или Environment Variables), **не** через `ENV` в Dockerfile (Vast UI env образа не читает).
- Для TTS: `-e OPEN_BUTTON_PORT=8000` (или `1111` если через Instance Portal).
- Env vars применяются при **`docker create`** — после Edit → **Apply**, не при простом Stop/Start если менялся конфиг.

Без `OPEN_BUTTON_PORT` Forge/TTS может работать по прямой ссылке, но кнопки Open не будет.

### Instance Portal (порт 1111)

На образах `vastai/base-image` / `vastai/pytorch` есть стек Portal (Caddy + cloudflared).  
Документация: [Instance Portal](https://docs.vast.ai/guides/instances/connect/instance-portal)

- Внутренний portal: **1111**
- `PORTAL_CONFIG` — список приложений (`localhost:8000:8000:/:Qwen TTS|...`)
- Токен: `OPEN_BUTTON_TOKEN` в `/proc/1/environ` (в SSH `echo $...` часто пустой — читать из pid 1)
- Quick tunnels (trycloudflare) бывают нестабильны; прямой Open Ports надёжнее

### Persistent storage

**Всё важное — в `/workspace`:**

```
/workspace/models/          # веса HF (если качаем сами)
/workspace/outputs/         # wav/mp3
/workspace/cache/huggingface/
/workspace/log/
```

При сборке образа: entrypoint симлинкует или копирует только **недостающее** — не затирать пользовательские данные.

Установки через pip/ apt вне `/workspace` **теряются** при пересоздании контейнера (Edit Apply), но `/workspace` сохраняется.

### SSH

1. Ключ в **Account → Keys** на vast.ai
2. Порт **22** в template: `-p 22:22`
3. Launch Mode = **Docker ENTRYPOINT** (не «SSH mode»)
4. `SSH_PUBLIC_KEY` инжектится в контейнер при create

```bash
ssh -p <VAST_TCP_PORT_22> -i ~/.ssh/id_ed25519 root@<IP>
```

Env в SSH-сессии может быть пустым — переменные у **pid 1** (`tr '\0' '\n' < /proc/1/environ`).

### PROVISIONING_SCRIPT (быстрый старт без своего Dockerfile)

Vast может при старте выполнить bash по URL — см. [Advanced Setup](https://docs.vast.ai/guides/templates/advanced-setup).

Подходит для **эксперимента**, не для финального образа:

- установить `qwen-tts`, скачать модель в `/workspace`
- запустить `qwen-tts-demo` или uvicorn
- прописать в **on-start** (Jupyter/SSH mode) или внутри скрипта

### Extra Filters (GPU)

| Фильтр | Зачем |
|---|---|
| `cuda_max_good>=12.1` | минимум CUDA на хосте |
| `compute_cap>=750` | RTX 20xx+ |
| `compute_cap<=900` | **только до Ada (40xx)** — если образ на torch **cu121 / без sm_120** |
| *(без `<=900`)* | нужен для **RTX 5000** — образ на **PyTorch cu128** |

### GPU: RTX 5000 (Blackwell, sm_120)

Из опыта sd_forge_max (май 2026):

- **PyTorch 2.3.1+cu121** поддерживает arch до **sm_90** → на 5060 Ti / 5090 inference **падает** (`no kernel image`).
- Для Blackwell нужен **torch ≥2.7 с cu128** (CUDA 12.8+).
- Vast base tags: `vastai/pytorch:2.11.0-cu128-cuda-12.9-mini-py310-*` и аналоги — см. [Docker Hub vastai/pytorch](https://hub.docker.com/r/vastai/pytorch/tags)
- Community Qwen server (malaiwah) уже на **cu128** — предпочтительнее для 5090

| Серия | sm | Образ torch |
|---|---|---|
| RTX 3060–4090 | 86–90 | cu121 / cu126 OK |
| RTX 5070–5090, 5060 Ti | 120 | **cu128 обязателен** |

### Типичные ошибки Vast

| Симптом | Причина | Лечение |
|---|---|---|
| `unknown flag: --runtime` | баг хоста Vast | Destroy → другая машина; **не** добавлять `--runtime` в Docker Options |
| Connecting… forever | Portal/tunnel | прямой Open Ports; или setup_portal (см. sd_forge_max) |
| Open пропала после Edit | выпал `OPEN_BUTTON_PORT` | восстановить env + Apply |
| Permission denied SSH | ключ не в контейнере | Keys в Account → Edit Apply |
| Модели пропали | не в `/workspace` | перенос путей в entrypoint |

---

## TTS на Vast — существующие шаблоны Vast.ai

Vast уже продвигает TTS-шаблоны (другие модели, но полезно как reference UX):

| Шаблон | Модель | VRAM | Документация |
|---|---|---|---|
| **Dia 1.6B TTS** | Nari Labs Dia | ~8 GB | [TTS with Nari Labs Dia](https://docs.vast.ai/tts-with-nari-labs-dia) |
| Model Library | разные audio | — | [Popular templates](https://vast.ai/article/popular-templates-for-ai-workloads-on-vast-ai) |

Flow Dia на Vast:

1. Выбрать template → GPU → rent
2. **Open** → Instance Portal → логи установки
3. **Dia TTS Interface** → Web UI
4. Файлы в `/workspace/dia` (SSH / Jupyter terminal)

Для **Qwen TTS** UX можно сделать аналогично: Portal → «Qwen TTS» → `:8000`.

---

## Архитектура проекта Qwen TTS (предложение)

### Вариант A — свой Docker-образ (рекомендуется)

```
FROM vastai/pytorch:2.11.0-cu128-cuda-12.9-mini-py310-...   # или py312
  → pip install qwen-tts flash-attn
  → entrypoint: symlink /workspace, setup_ssh, (optional portal)
  → CMD: qwen-tts-demo ... --ip 0.0.0.0 --port 8000
  → GHCR: ghcr.io/akai2211/qwen_tts:latest
```

Плюсы: воспроизводимость, быстрый cold start, один Launch Mode.  
Минусы: время сборки, pin версий torch/flash-attn.

### Вариант B — форк community server

База: `ghcr.io/malaiwah/qwen3-tts-server` — OpenAI API уже есть, 5090 проверен.

Плюсы: быстрее старт проекта.  
Минусы: меньше контроля, зависимость от upstream.

### Вариант C — PROVISIONING_SCRIPT без образа

Базовый `vastai/pytorch` + скрипт установки при первом старте.

Плюсы: быстрый прототип.  
Минусы: каждый новый инстанс качает заново; хуже для «продукта».

### Порты (черновик шаблона)

| Порт | Сервис |
|---|---|
| **8000** | Qwen Web UI (`qwen-tts-demo`) или FastAPI |
| **8001** | альтернатива (OpenAI API server) |
| **1111** | Instance Portal (optional) |
| **22** | SSH |

**Docker Options (черновик):**

```
-p 8000:8000 -p 1111:1111 -p 22:22 -e OPEN_BUTTON_PORT=8000
```

**Environment Variables (пример):**

| Key | Value |
|---|---|
| `OPEN_BUTTON_PORT` | `8000` |
| `HF_TOKEN` | Hugging Face token (если gated models) — лучше Account env, не public template |
| `QWEN3_TTS_MODEL` | `Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice` |

**Launch Mode:** Docker ENTRYPOINT  
**On-start:** пусто  
**Disk:** 40–60 GB (модели ~3–6 GB каждая + cache)  
**Extra filters:** для cu128 / 5090 — **без** `compute_cap<=900`; для cu121 — **с** `compute_cap<=900`

---

## Workflow разработки на Vast

1. Локально: Dockerfile / scripts в репо `qwen_tts` (имя TBD).
2. Push → GHA → GHCR.
3. Template на Vast → image tag.
4. Тест: **новый инстанс** (не полагаться на старый `/workspace` при смене архитектуры).
5. Проверка:
   - `curl -sf http://127.0.0.1:8000/` или API health
   - синтез 1 фразы → wav без CUDA error
   - `nvidia-smi`, `python -c "import torch; print(torch.__version__, torch.cuda.get_device_capability())"`

### Чеклист диагностики на инстансе

```bash
# GPU
nvidia-smi --query-gpu=name,compute_cap,driver_version --format=csv

# PyTorch / arch
python3 -c "import torch; print(torch.__version__, torch.version.cuda); print(torch.cuda.get_arch_list()); print(torch.cuda.get_device_capability())"

# Env Vast (источник истины — pid 1)
tr '\0' '\n' < /proc/1/environ | grep -E '^(OPEN_|VAST_|HF_)' | sort

# Сервис
ss -tlnp | grep -E '8000|8001|1111|22'
curl -sf -o /dev/null -w '%{http_code}\n' http://127.0.0.1:8000/
```

---

## Открытые задачи проекта (TODO)

- [ ] Создать репозиторий `qwen_tts` (или `qwen-tts-vast`) и ветку `main`
- [ ] Выбрать вариант A/B/C и целевую модель (0.6B vs 1.7B, CustomVoice vs Base)
- [ ] Dockerfile: база **cu128** если нужна 50-series; отдельный тег `:latest-cu121` опционально для 30xx/40xx
- [ ] entrypoint: `/workspace` для HF cache, outputs, logs
- [ ] setup_ssh.sh (можно адаптировать из sd_forge_max)
- [ ] setup_portal.sh (optional — кнопка Open; учесть опыт Connecting… из sd_forge_max)
- [ ] GHA build → `ghcr.io/akai2211/qwen_tts:latest`
- [ ] README + шаблон Vast (порты, env, filters)
- [ ] Тест: RTX 3060/4090 + RTX 5090 (если cu128)
- [ ] (Опционально) OpenAI-compatible API поверх qwen-tts
- [ ] (Опционально) auto-download моделей при старте (аналог provision_models.sh)

---

## Anti-patterns

- ❌ Launch Mode **Jupyter** для финального образа с entrypoint — сервис не поднимется сам
- ❌ `ENV OPEN_BUTTON_PORT=...` в Dockerfile вместо шаблона Vast
- ❌ Модели в `/root/.cache` без volume — пропадут или не переживут migrate
- ❌ torch cu121 на RTX 5090 / 5060 Ti — UI может стартовать, **синтез упадёт**
- ❌ `--runtime=nvidia` / `--gpus` в Docker Options — Vast сам выставляет GPU
- ❌ Секреты (`HF_TOKEN`, API keys) в **public** template

---

## Полезные ссылки

### Qwen3-TTS

- GitHub: <https://github.com/QwenLM/Qwen3-TTS>
- Hugging Face collection: <https://huggingface.co/collections/Qwen/qwen3-tts>
- Blog: <https://qwen.ai/blog?id=qwen3tts-0115>
- HF Demo: <https://huggingface.co/spaces/Qwen/Qwen3-TTS>
- ModelScope Demo: <https://modelscope.cn/studios/Qwen/Qwen3-TTS>

### Community Docker / API

- qwen3-tts-server: <https://github.com/malaiwah/qwen3-tts-server>
- qwen-tts-service: <https://hub.docker.com/r/mosstnslv/qwen-tts-service>
- EchoFleet-Qwen3-TTS: <https://github.com/solotrode/EchoFleet-Qwen3-TTS>

### Vast.ai

- Docs index: <https://docs.vast.ai/llms.txt>
- Quickstart: <https://docs.vast.ai/>
- Template Settings: <https://docs.vast.ai/guides/templates/template-settings>
- Creating Templates: <https://docs.vast.ai/guides/templates/creating-templates>
- Advanced Setup: <https://docs.vast.ai/guides/templates/advanced-setup>
- Instance Portal: <https://docs.vast.ai/guides/instances/connect/instance-portal>
- Managing Instances: <https://docs.vast.ai/guides/instances/manage-instances>
- Dia TTS guide (reference UX): <https://docs.vast.ai/tts-with-nari-labs-dia>
- vastai/pytorch tags: <https://hub.docker.com/r/vastai/pytorch/tags>

### PyTorch / Blackwell

- PyTorch get started: <https://pytorch.org/get-started/locally/>
- cu128 wheels: `pip install torch --index-url https://download.pytorch.org/whl/cu128`
- Forum sm_120: <https://discuss.pytorch.org/t/pytorch-support-for-sm120/216099>

### Связанный проект (опыт Vast)

- sd_forge_max: <https://github.com/akai2211/sd_forge_max>
- CURSOR.md sd_forge_max — паттерны SSH, portal, Open Ports, GPU sm_120

---

## Как начать новый чат Cursor

Скажи ассистенту:

> Прочитай `qwen.md`. Проект — **Qwen3-TTS на Vast.ai** (озвучка). Цель — Docker + шаблон Vast. Учитывай Launch Mode ENTRYPOINT, `/workspace`, Open Ports, cu128 для RTX 5000. Не путай с sd_forge_max. Общаемся по-русски.

---

## Контекст для AI

- Пользователь хочет **озвучку** (TTS), не генерацию картинок.
- Деньги на Vast тратятся почасово — давать **готовые команды** для копипасты.
- Объяснять **почему**, не только «сделай Apply».
- **Не коммитить** без явной просьбы.
- При Open Ports — **сначала** блок ссылок SSH / UI / localhost tunnel **отдельными строками**.

---

*Последнее обновление handoff: май 2026.*
