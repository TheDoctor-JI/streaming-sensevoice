"""
Auther: ISJDOG

## Cli

```bash
python realtime_ws_server_demo.py --help
```

## Debug with vscode:

```
{
  // Use IntelliSense to learn about possible attributes.
  // Hover to view descriptions of existing attributes.
  // For more information, visit: https://go.microsoft.com/fwlink/?linkid=830387
  "version": "0.2.0",
  "configurations": [
    {
      "name": "Python Debugger: Current File",
      "type": "debugpy",
      "request": "launch",
      "program": "${file}",
      "console": "integratedTerminal",
      "env": {
        "SENSEVOICE_MODEL_PATH": "iic/SenseVoiceSmall",
        "DEVICE": "cuda",
      }
    }
  ]
}
```
"""

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request, HTTPException
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware

from urllib.parse import parse_qs

from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings

from pysilero import VADIterator

from loguru import logger

import numpy as np

import sys, uuid

import soundfile as sf
import io


class Config(BaseSettings, cli_parse_args=True, cli_use_class_docs_for_groups=True):
    HOST: str = Field("127.0.0.1", description="Host")
    PORT: int = Field(8000, description="Port")
    DEBUG: bool = Field(False, description="Debug mode")
    SENSEVOICE_MODEL_PATH: str = Field(
        "iic/SenseVoiceSmall", description="SenseVoice model path"
    )
    DEVICE: str = Field("cpu", description="Device (cpu, cuda)")
    CUDA_DEVICE_INDEX: int | None = Field(
        None, description="CUDA device index (e.g. 0 for cuda:0)"
    )
    LANGUAGE: str = Field("auto", description="Default language (auto, zh, en, ja, ko, yue)")
    SILEROVAD_VERSION: str = Field("v5", description="SileroVAD version, v4 or v5")
    SAMPLERATE: int = Field(16000, description="Sample rate")
    CHUNK_DURATION: float = Field(0.1, description="Chunk duration (s)")
    VAD_MIN_SILENCE_DURATION_MS: int = Field(
        550, description="VAD min slience duration (ms)"
    )
    VAD_THRESHOLD: float = Field(0.5, description="VAD threshold")


config = Config()

device_raw = config.DEVICE.strip()
device_lower = device_raw.lower()

if device_lower.startswith("cuda"):
    sanitized_device = device_raw.replace(" ", "")
    explicit_index: int | None = None

    if ":" in sanitized_device:
        _, suffix = sanitized_device.split(":", 1)
        suffix = suffix.strip()
        if suffix:
            try:
                explicit_index = int(suffix)
            except ValueError as exc:
                raise ValueError(
                    "CUDA device index provided in DEVICE must be an integer"
                ) from exc

    chosen_index: int | None = (
        explicit_index if explicit_index is not None else config.CUDA_DEVICE_INDEX
    )

    if chosen_index is not None:
        if chosen_index < 0:
            raise ValueError("CUDA device index must be non-negative")
        normalized_device = f"cuda:{chosen_index}"
    else:
        normalized_device = "cuda"

    config.DEVICE = normalized_device
    config.CUDA_DEVICE_INDEX = chosen_index

    if chosen_index is not None:
        try:
            import torch

            torch.cuda.set_device(chosen_index)
        except ImportError:
            logger.warning(
                "PyTorch with CUDA support is not available; unable to set CUDA device index"
            )
        except Exception as exc:
            raise RuntimeError(
                f"Failed to select CUDA device index {chosen_index}: {exc}"
            ) from exc
else:
    config.DEVICE = "cpu"
    config.CUDA_DEVICE_INDEX = None

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

from streaming_sensevoice import StreamingSenseVoice

# load model on startup
StreamingSenseVoice.load_model(model=config.SENSEVOICE_MODEL_PATH, device=config.DEVICE)


class TranscriptionChunk(BaseModel):
    timestamps: list[int]
    raw_text: str
    final_text: str | None = None
    spk_id: int | None = None


class TranscriptionResponse(BaseModel):
    type: str = "TranscriptionResponse"
    id: int
    begin_at: float
    end_at: float | None
    data: TranscriptionChunk
    is_final: bool
    session_id: str | None = None


class VADEvent(BaseModel):
    type: str = "VADEvent"
    is_active: bool


@app.get("/")
async def clientHost():
    return FileResponse("realtime_ws_client.html", media_type="text/html")


@app.websocket("/api/realtime/ws")
async def websocket_endpoint(websocket: WebSocket):
    try:
        await websocket.accept()

        session_id = str(uuid.uuid4())
        logger.info(f"Session {session_id} opened")

        query_params = parse_qs(websocket.scope["query_string"].decode())
        chunk_duration = float(
            query_params.get("chunk_duration", config.CHUNK_DURATION)
        )
        vad_threshold = float(query_params.get("vad_threshold", config.VAD_THRESHOLD))
        vad_min_silence_duration_ms = int(
            query_params.get(
                "vad_min_silence_duration_ms", config.VAD_MIN_SILENCE_DURATION_MS
            )
        )
        requested_language = query_params.get("language", [config.LANGUAGE])[0]
        if requested_language:
            requested_language = requested_language.lower()
        else:
            requested_language = config.LANGUAGE

        available_languages: set[str] | None = None
        try:
            base_model, _ = StreamingSenseVoice.load_model(
                model=config.SENSEVOICE_MODEL_PATH, device=config.DEVICE
            )
            available_languages = set(base_model.lid_dict.keys())
        except Exception as exc:
            logger.error(f"Failed to inspect available languages: {exc}")

        if available_languages and requested_language not in available_languages:
            message = {
                "type": "Error",
                "detail": (
                    "Unsupported language. Available options: "
                    f"{sorted(available_languages)}"
                ),
            }
            await websocket.send_json(message)
            await websocket.close(code=1003)
            logger.warning(
                f"Session {session_id} closed: unsupported language '{requested_language}'"
            )
            return

        try:
            sensevoice_model = StreamingSenseVoice(
                model=config.SENSEVOICE_MODEL_PATH,
                device=config.DEVICE,
                language=requested_language,
            )
        except KeyError:
            message = {
                "type": "Error",
                "detail": (
                    "Unsupported language. Available options: "
                    f"{sorted(available_languages) if available_languages else 'auto, zh, en, ja, ko, yue'}"
                ),
            }
            await websocket.send_json(message)
            await websocket.close(code=1003)
            logger.warning(
                f"Session {session_id} closed during model init: unsupported language '{requested_language}'",
            )
            return

        logger.info(f"Session {session_id} configured with language='{requested_language}'")
        vad_iterator = VADIterator(
            version=config.SILEROVAD_VERSION,
            threshold=vad_threshold,
            min_silence_duration_ms=vad_min_silence_duration_ms,
        )

        audio_buffer = np.array([], dtype=np.float32)
        chunk_size = int(chunk_duration * config.SAMPLERATE)

        speech_count = 0
        currentAudioBeginTime = 0.0

        asrDetected = False

        transcription_response: TranscriptionResponse = None
        while True:
            data = await websocket.receive_bytes()

            # mp3 decode
            buffer = io.BytesIO(data)
            try:
                buffer.name = "a.mp3"

                samples, sr = sf.read(buffer, dtype="float32")
                audio_buffer = np.concatenate((audio_buffer, samples))
            except sf.LibsndfileError as e:
                continue
            finally:
                buffer.close()

            if sr != config.SAMPLERATE:
                raise ValueError("Sample rate mismatch")

            while len(audio_buffer) >= chunk_size:
                chunk = audio_buffer[:chunk_size]
                audio_buffer = audio_buffer[chunk_size:]

                for speech_dict, speech_samples in vad_iterator(chunk):
                    if "start" in speech_dict:
                        sensevoice_model.reset()

                        currentAudioBeginTime: float = (
                            speech_dict["start"] / config.SAMPLERATE
                        )

                        if asrDetected:
                            logger.debug(
                                f"{speech_count}: VAD *NOT* end: \n{transcription_response.data.raw_text}\n{str(transcription_response.data.timestamps)}"
                            )
                            speech_count += 1
                        asrDetected = False

                        logger.debug(
                            f"{speech_count}: VAD start: {currentAudioBeginTime}"
                        )
                        await websocket.send_json(VADEvent(is_active=True).model_dump())

                    is_last = "end" in speech_dict

                    for res in sensevoice_model.streaming_inference(
                        speech_samples, is_last
                    ):

                        if len(res["text"]) > 0:
                            asrDetected = True

                        if asrDetected:
                            transcription_response = TranscriptionResponse(
                                id=speech_count,
                                begin_at=currentAudioBeginTime,
                                end_at=None,
                                data=TranscriptionChunk(
                                    timestamps=res["timestamps"], raw_text=res["text"]
                                ),
                                is_final=False,
                                session_id=session_id,
                            )
                            await websocket.send_json(
                                transcription_response.model_dump()
                            )

                    if is_last:
                        if asrDetected:
                            speech_count += 1
                            asrDetected = False

                            transcription_response.is_final = True
                            transcription_response.end_at = (
                                speech_dict["end"] / config.SAMPLERATE
                            )

                            await websocket.send_json(
                                transcription_response.model_dump()
                            )
                            logger.debug(
                                f"{speech_count}: VAD end: {speech_dict['end'] / config.SAMPLERATE}\n{transcription_response.data.raw_text}\n{str(transcription_response.data.timestamps)}"
                            )
                        else:
                            logger.debug(
                                f"{speech_count}: VAD end: {speech_dict['end'] / config.SAMPLERATE}\nNo Speech"
                            )
                        await websocket.send_json(
                            VADEvent(is_active=False).model_dump()
                        )

    except WebSocketDisconnect:
        logger.info("WebSocket disconnected")
    finally:
        sensevoice_model.reset()
        del sensevoice_model
        del vad_iterator
        del audio_buffer
        logger.info(f"Session {session_id} closed")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host=config.HOST, port=config.PORT)
