# Qwen3-TTS-Vast — handoff для Cursor

> Handoff для AI. Пользователь: **akai2211**, по-русски, читает логи.  
> Для людей: **README.md**. Не путать с [sd_forge_max](https://github.com/akai2211/sd_forge_max) (другой образ и порты).

## Статус

| | |
|---|---|
| **Репо** | <https://github.com/akai2211/Qwen3-TTS-Vast> |
| **Образ** | `ghcr.io/akai2211/qwen3-tts-vast:latest` (`cu128`) |
| **UI** | Gradio 5 — `src/app/app.py`, порт **8000** |
| **База** | `vastai/base-image:cuda-12.9-mini-py312-2026-04-15` + torch cu128 |
| **Лицензия** | MIT (репо); модели Qwen — Apache 2.0 |
| **UI-источник** | <https://www.youtube.com/watch?v=P-oMObPeA18> |

Проверено на Vast: **RTX 5060 Ti**. После правок образа — только **новый инстанс** (не патчить running-контейнер).

## Структура репозитория

```text
Dockerfile, requirements.txt
scripts/entrypoint.sh, setup_ssh.sh, setup_portal.sh
src/app/app.py, database.py
.github/workflows/build.yml
```

## Qwen3-TTS (кратко)

| Режим в UI | HF (пример) |
|---|---|
| Custom Voice | `Qwen/Qwen3-TTS-12Hz-{0.6B,1.7B}-CustomVoice` |
| Voice Design | `Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign` |
| Voice Clone | `Qwen/Qwen3-TTS-12Hz-{0.6B,1.7B}-Base` |

VRAM: **0.6B** ~4–6 GB, **1.7B** ~6–8 GB (+ Whisper при клоне — лучше 12–16 GB).  
Коллекция: <https://huggingface.co/collections/Qwen/qwen3-tts>  
**HF_TOKEN** для публичных Qwen-моделей обычно **не нужен**.

## Образ Docker

```
vastai/base-image:cuda-12.9-mini-py312-2026-04-15
  → python3 -m pip install torch (index cu128) + qwen-tts + gradio 5
  → flash-attn: только --only-binary (иначе skip)
  → CMD: python3 /app/app.py
```

**CI:** не `pip install --upgrade pip` (deb Ubuntu → `RECORD file not found`); `PIP_BREAK_SYSTEM_PACKAGES=1`, `python3 -m pip`.

**Gradio 5:** `theme`/`css` в `gr.Blocks(...)`, не в `demo.launch()` — иначе `TypeError` и **Exited**.

**Portal:** в `PORTAL_CONFIG` есть `|` — не `sed` с разделителем `|`.

### `/workspace`

| Путь | Назначение |
|---|---|
| `cache/huggingface` | `HF_HOME`, веса моделей (диск) |
| `custom_voices`, `designed_voices` | сохранённые голоса (клон / дизайн) |
| `exports` | ZIP бэкапы и временные файлы импорта |
| `outputs` | служебное (Gradio / экспорт) |
| `log` | зарезервировано |

### Модели на диск

| | |
|---|---|
| **Куда** | `HF_HOME` = `/workspace/cache/huggingface` |
| **Скачать** | вкладка «Модели» → «Скачать», или при первой генерации |
| **VRAM** | только «Загрузить в GPU» или синтез |

### UI: бэкап и скачивание

| Вкладка | Функция |
|---|---|
| **💾 Бэкап голосов** | ZIP всех `custom_voices` + `designed_voices`; импорт на другой инстанс |
| **Результат 1…5** | `show_download_button=True` — скачать сгенерированный WAV |
| Перенос без UI | `scp` папки `/workspace/custom_voices/` (см. README) |

Локальная копия для разработки: `qwen3-voices/` в gitignore, не коммитить.

## Шаблон Vast.ai

| Поле | Значение |
|---|---|
| Image | `ghcr.io/akai2211/qwen3-tts-vast:latest` |
| **Launch Mode** | **Docker ENTRYPOINT** (не Jupyter) |
| Disk | 50–70 GB |
| Docker options | `-p 8000:8000 -p 1111:1111 -p 22:22 -e OPEN_BUTTON_PORT=8000` |
| Env | `OPEN_BUTTON_PORT=8000` |

**Extra filters (50xx):** `cuda_max_good>=12.8 compute_cap>=890` — **без** `compute_cap<=900`.  
**40xx на том же образе:** `cuda_max_good>=12.1 compute_cap>=750 compute_cap<=900`.

| Порт | Сервис |
|---|---|
| 8000 | Gradio |
| 1111 | Instance Portal |
| 22 | SSH (ключ в Account → Keys) |

`OPEN_BUTTON_PORT` — только в **шаблоне** Vast, не `ENV` в Dockerfile.

### Доступ

**Open Ports** → `http://<IP>:<внешний_8000>/`  
При «Connecting…» — прямой порт 8000, не ждать Portal.

Если пользователь прислал Open Ports — сразу отдельными строками: SSH, UI, `ssh -L 8000:localhost:8000`.

```bash
ssh -p <VAST_TCP_PORT_22> -i ~/.ssh/id_ed25519 root@<IP>
curl -sf -o /dev/null -w '%{http_code}\n' http://127.0.0.1:8000/
python3 -c "import torch; print(torch.__version__, torch.cuda.get_device_capability())"
```

Env Vast в SSH часто пустой — смотреть `tr '\0' '\n' < /proc/1/environ`.

## GPU

| GPU | torch |
|---|---|
| RTX 5070–5090, 5060 Ti (sm_120) | **cu128** обязателен |
| RTX 3060–4090 | cu128 в образе тоже ок; фильтр с `<=900` |

cu121 на 50xx: синтез падает `no kernel image is available`.

## Workflow

1. Правки → коммит в **`main`** (по просьбе пользователя) → пользователь сам **`git push origin main`**.  
2. GHA собирает образ только с **push в `main`**.  
3. **Новый инстанс** с `:latest` (не патчить running).  
4. Проверка: HTTP 8000, вкладка «Модели», синтез, нет Traceback в LOG.

**Git для AI:** не предлагать PR / ветки `cursor/*`, если пользователь не просил. Схема: сделал фичу → закоммитил в `main` → пользователь пушит. Push и merge делает пользователь.

## Сбои (LOG)

| Симптом | Причина |
|---|---|
| Exited, `unexpected keyword argument 'theme'` | Gradio 5 — theme в Blocks |
| Connecting… | процесс упал или Portal; прямой :8000 |
| `no kernel image` | cu121 на 50xx |
| `CUDA out of memory` | несколько моделей в GPU; 1.7B + Whisper; выгрузить на «Модели», взять 0.6B |
| CI `Cannot uninstall pip` | не апгрейдить pip в Dockerfile |
| `sed: unknown option` | portal + `\|` в PORTAL_CONFIG |
| Нет кнопки Open | нет `OPEN_BUTTON_PORT` в шаблоне |
| Модели «Не загружено» | «Скачать» на «Модели» или дождаться скачивания при синтезе |

## Anti-patterns

- Jupyter Launch Mode — entrypoint образа не выполняется  
- `--gpus` / `--runtime=nvidia` в Docker Options  
- Модели только в `/root/.cache`  
- `HF_TOKEN` в public template  
- Патч `/app` на текущем инстансе  

## Чеклист проекта

### Выполнено

- [x] Репозиторий [Qwen3-TTS-Vast](https://github.com/akai2211/Qwen3-TTS-Vast), ветка `main`
- [x] Dockerfile: `vastai/base-image` + torch **cu128**, `qwen-tts`, Gradio 5
- [x] Кастомный UI `src/app/app.py` (CustomVoice / Design / Clone, база стилей)
- [x] Entrypoint: `/workspace`, HF cache, SSH, Instance Portal
- [x] GHA → `ghcr.io/akai2211/qwen3-tts-vast:latest` (теги `latest`, `cu128`)
- [x] Фиксы: pip в CI, Gradio 5 `theme` в Blocks, portal `sed` + `|`
- [x] README.md, LICENSE (MIT), ссылка на видео-источник
- [x] Шаблон Vast: ENTRYPOINT, порты 8000/1111/22, `OPEN_BUTTON_PORT`
- [x] AGENTS.md — актуальный handoff
- [x] Тест на Vast: **RTX 5060 Ti**, UI в браузере (после фикса Exited)
- [x] Бэкап голосов: экспорт/импорт ZIP (`💾 Бэкап голосов`)
- [x] Скачивание сгенерированного WAV (`show_download_button` на «Результат»)
### Планы

- [ ] Подтвердить на **RTX 4090** (Ada) и **5090** (отдельные инстансы)
- [ ] Отдельный тег образа `:cu121` для дешёвых 30xx/40xx (опционально)
- [ ] OpenAI-compatible API (`/v1/audio/speech`) — свой слой или образ malaiwah
- [ ] Автовыгрузка предыдущей модели из VRAM при смене типа (меньше OOM на 16 GB)
- [ ] Публичный шаблон Vast в каталоге (сейчас private — ок)
- [ ] Сжатие/квантование моделей для 8 GB GPU (0.6B по умолчанию в UI)

## Ссылки

- Qwen3-TTS: <https://github.com/QwenLM/Qwen3-TTS>  
- Vast docs: <https://docs.vast.ai/llms.txt>  
- Instance Portal: <https://docs.vast.ai/guides/instances/connect/instance-portal>  

## Новый чат

> Прочитай `README.md` и `AGENTS.md`. **Qwen3-TTS-Vast**, образ `ghcr.io/akai2211/qwen3-tts-vast:latest`, ENTRYPOINT, :8000, cu128, после правок — новый инстанс. По-русски.

**AI:** TTS, не Forge; по-русски; команды для копипасты; коммит — по просьбе, **push только пользователь**; не путать скачивание на диск с «Загрузить в GPU»; не навязывать PR/`cursor/*` ветки; Open Ports — ссылки первым блоком.

*Июнь 2026.*
