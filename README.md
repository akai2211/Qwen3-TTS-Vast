# Qwen3-TTS-Vast

Docker-образ и шаблон для запуска **[Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS)** на [Vast.ai](https://vast.ai): Gradio UI в браузере, GPU-синтез, persistent `/workspace`, SSH и Instance Portal.

Основано на разборе и доработке UI из видео-инструкции (см. [источник](#источник) в конце).

## Возможности

- **Custom Voice** — 9 preset-голосов + `instruct` по стилю  
- **Voice Design** — голос из текстового описания  
- **Voice Clone** — клон по ~3 с reference audio + транскрипт (Whisper)  
- **База стилей** — пресеты в `src/app/database.py`  
- Сохранение своих голосов в `/workspace/custom_voices`  
- **RTX 50xx (Blackwell)** — PyTorch **cu128**, compute capability sm_120  

## Образ

```text
ghcr.io/akai2211/qwen3-tts-vast:latest
```

Сборка при push в `main` (GitHub Actions). Теги: `latest`, `cu128`.

## Быстрый старт на Vast.ai

### Шаблон

| Поле | Значение |
|------|----------|
| **Image** | `ghcr.io/akai2211/qwen3-tts-vast:latest` |
| **Launch Mode** | **Docker ENTRYPOINT** (не Jupyter) |
| **Disk** | 50–70 GB |
| **Docker options** | `-p 8000:8000 -p 1111:1111 -p 22:22 -e OPEN_BUTTON_PORT=8000` |
| **Environment** | `OPEN_BUTTON_PORT=8000` |

**Extra filters (RTX 5060 Ti / 5090):**

```text
cuda_max_good>=12.8 compute_cap>=890
```

На 50xx **не** используйте `compute_cap<=900`.

### Доступ

После **Running** на карточке инстанса → **Open Ports**:

- UI: `http://<IP>:<внешний_порт_8000>/`
- SSH: `ssh -p <VAST_TCP_PORT_22> -i ~/.ssh/id_ed25519 root@<IP>`

Первый синтез скачивает модели в `/workspace/cache/huggingface` (несколько минут).

### Проверка на инстансе

```bash
curl -sf -o /dev/null -w '%{http_code}\n' http://127.0.0.1:8000/
python3 -c "import torch; print(torch.__version__, torch.cuda.get_device_capability())"
```

## Локальная сборка

```bash
git clone https://github.com/akai2211/Qwen3-TTS-Vast.git
cd Qwen3-TTS-Vast
docker build -t qwen3-tts-vast:local .
docker run --gpus all -p 8000:8000 -v "$(pwd)/workspace:/workspace" qwen3-tts-vast:local
```

Откройте http://localhost:8000/

## Структура репозитория

```text
Dockerfile              # vastai/base-image + torch cu128 + qwen-tts
scripts/entrypoint.sh   # /workspace, SSH, Portal
scripts/setup_ssh.sh
scripts/setup_portal.sh
src/app/app.py          # Gradio UI
src/app/database.py     # пресеты стилей
requirements.txt
.github/workflows/build.yml
```

`AGENTS.md` — расширенный handoff для разработки (не обязателен для запуска).

## VRAM (ориентир)

| Модель | VRAM |
|--------|------|
| 0.6B | ~4–6 GB |
| 1.7B | ~6–8 GB (+ Whisper при клоне — лучше 12–16 GB) |

## Hugging Face token

Для официальных моделей **Qwen/Qwen3-TTS-12Hz-*** токен обычно **не нужен** (публичные веса, Apache 2.0). Добавляйте `HF_TOKEN` только при gated/private моделях.

## Типичные проблемы

| Симптом | Решение |
|---------|---------|
| **Exited** / нет UI | Launch Mode = Docker ENTRYPOINT; смотреть LOG |
| **Connecting…** | Прямой порт 8000; portal опционален |
| `no kernel image` на 50xx | Образ с **cu128**, не cu121 |
| Кнопки Open нет | `OPEN_BUTTON_PORT=8000` в **шаблоне** Vast |

После правок кода — **новый инстанс** с актуальным тегом образа (не патчить контейнер вручную).

## Ссылки

- [Qwen3-TTS (GitHub)](https://github.com/QwenLM/Qwen3-TTS)  
- [Модели на Hugging Face](https://huggingface.co/collections/Qwen/qwen3-tts)  
- [Документация Vast.ai](https://docs.vast.ai/)  

## Лицензия

Код этого репозитория (Docker, скрипты, правки деплоя) — **[MIT](LICENSE)**.

Модели и библиотека **Qwen3-TTS** — [Apache License 2.0](https://github.com/QwenLM/Qwen3-TTS/blob/main/LICENSE).

## Источник

UI и идея развёртывания опираются на видео-разбор:

**[YouTube — основа проекта](https://www.youtube.com/watch?v=P-oMObPeA18)**

Канал автора UI: [Максим Юровских](https://www.youtube.com/channel/UCLoDL_MJpkrMizBuuXnRYsg)
