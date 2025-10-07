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

import json
import uuid
import time

import base64

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


class WordTimestamp(BaseModel):
    word: str
    start_ms: float
    end_ms: float | None = None


class TranscriptionChunk(BaseModel):
    timestamps: list[int]
    raw_text: str
    final_text: str | None = None
    spk_id: int | None = None
    word_timestamps: list[WordTimestamp] | None = None


class TranscriptionResponse(BaseModel):
    type: str = "TranscriptionResponse"
    id: int
    begin_at: float
    end_at: float | None
    data: TranscriptionChunk
    is_final: bool
    session_id: str | None = None
    segment_start_s: float | None = None
    segment_end_s: float | None = None
    session_start_walltime: float | None = None
    aud_seg_indx: int | None = None


class VADEvent(BaseModel):
    type: str = "VADEvent"
    is_active: bool
    segment_start_s: float | None = None
    segment_end_s: float | None = None
    session_start_walltime: float | None = None
    aud_seg_indx: int | None = None

@app.get("/")
async def clientHost():
    return FileResponse("realtime_ws_client.html", media_type="text/html")


@app.get("/pcm-worklet-processor.js")
async def worklet_module() -> FileResponse:
    return FileResponse("pcm-worklet-processor.js", media_type="application/javascript")


def reset_vad_state_preserve_timing(vad_iterator: VADIterator):
    """
    Selectively reset VAD state while preserving timing counters.
    This allows the VAD to start fresh detection without resetting absolute timestamps.
    
    Resets:
    - VAD model internal state (h, c for v4 or state, context for v5)
    - Speech detection flags (triggered, temp_end)
    - Speech samples buffer
    - FrameQueue remained_samples buffer
    
    Preserves:
    - FrameQueue.current_sample (absolute sample counter)
    - FrameQueue.cache_start (absolute cache position)
    - FrameQueue.cached_samples (maintains timing consistency)
    """
    # Reset VAD model state (version-dependent)
    if vad_iterator.version == "v4":
        vad_iterator.h = np.zeros((2, 1, 64), dtype=np.float32)
        vad_iterator.c = np.zeros((2, 1, 64), dtype=np.float32)
    else:  # v5
        vad_iterator.state = np.zeros((2, 1, 128), dtype=np.float32)
        vad_iterator.context = np.zeros((1, vad_iterator.context_size), dtype=np.float32)
    
    # Reset VAD detection flags
    vad_iterator.triggered = False
    vad_iterator.temp_end = 0
    vad_iterator.speech_samples = np.empty(0, dtype=np.float32)
    
    # Clear the remained_samples buffer in the queue to avoid stale data
    vad_iterator.queue.remained_samples = np.empty(0, dtype=np.float32)
    
    # Reset resampler state if present to avoid carryover artifacts
    if vad_iterator.queue.resampler is not None:
        vad_iterator.queue.resampler = soxr.ResampleStream(
            int(vad_iterator.sample_rate),
            int(vad_iterator.model_sample_rate),
            num_channels=1
        )
    
    logger.debug(
        f"VAD state reset (timing preserved): current_sample={vad_iterator.queue.current_sample}, "
        f"cache_start={vad_iterator.queue.cache_start}"
    )


@app.websocket("/api/realtime/ws")
async def websocket_endpoint(websocket: WebSocket):
    try:
        await websocket.accept()

        session_start_walltime = time.time()
        
        session_id = str(uuid.uuid4())
        logger.info(f"Session {session_id} opened")

        query_params = parse_qs(websocket.scope["query_string"].decode())

        for key in query_params:
            if len(query_params[key]) == 0:
                query_params[key] = None
            elif len(query_params[key]) == 1:
                query_params[key] = query_params[key][0] ## url parse will always return a list, we just want the first one if only one value

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
        in_speech = False  # Track if currently inside an IPU
        current_seg_idx = -1  # Track current audio message segment index
        force_cutoff_pending = False  # Track if force cutoff is pending
        force_cutoff_target_seg = -1  # Track target segment for force cutoff

        samplerate = config.SAMPLERATE
        encoding = "s16le"
        
        transcription_response: TranscriptionResponse = None
        while True:
            message = await websocket.receive()

            if message.get("type") == "websocket.disconnect":
                raise WebSocketDisconnect()

            ## Due to the sequentialized operation here, this idx won't change asynchronously
            payload_bytes = message.get("bytes")
            payload_text = message.get("text")
            aud_seg_indx = -1
            samples_i16: np.ndarray | None = None ## Reset the audio storage


            if payload_bytes is not None:
                if len(payload_bytes) % 2 != 0:
                    logger.warning("Dropping trailing byte from odd-length audio payload")
                    payload_bytes = payload_bytes[:-1]

                samples_i16 = np.frombuffer(payload_bytes, dtype=np.int16)
            elif payload_text is not None:
                try:
                    structured_payload = json.loads(payload_text)
                except json.JSONDecodeError:
                    logger.warning("Received non-JSON text payload; ignoring")
                    continue

                if isinstance(structured_payload, dict):
                    # Check for force cutoff command first
                    command = structured_payload.get("command")
                    if command == "force_vad_offset":
                        target_seg_idx_str = structured_payload.get("cutoff_target_seg_idx")
                        try:
                            force_cutoff_target_seg = int(target_seg_idx_str)
                            force_cutoff_pending = True
                            logger.info(f"Force VAD cutoff command received for seg_idx={force_cutoff_target_seg}")

                        except (TypeError, ValueError):
                            logger.warning(f"Invalid seg_idx in force_vad_offset: {target_seg_idx_str}")
                            continue
                    else:##Skip audio extraction if force cutoff
                        aud_seg_indx_raw = structured_payload.get("seg_idx")
                        if aud_seg_indx_raw is not None:
                            try:
                                aud_seg_indx = int(aud_seg_indx_raw)
                                # Update current segment index
                                if aud_seg_indx >= 0:
                                    current_seg_idx = aud_seg_indx
                            except (TypeError, ValueError):
                                pass

                        audio_field = structured_payload.get("audio")
                        samplerate = structured_payload.get("sr", samplerate)
                        encoding = structured_payload.get("enc", encoding)

                        if samplerate != config.SAMPLERATE:
                            raise ValueError("Sample rate mismatch")

                        if isinstance(audio_field, str):
                            try:
                                audio_field = base64.b64decode(audio_field)
                            except Exception:
                                logger.warning("Failed to decode base64 audio payload; ignoring")
                                continue
                            samples_i16 = np.frombuffer(audio_field, dtype=np.int16)
                        elif isinstance(audio_field, list):
                            samples_i16 = np.asarray(audio_field, dtype=np.int16)
                        elif isinstance(audio_field, (bytes, bytearray)):
                            if len(audio_field) % 2 != 0:
                                logger.warning("Dropping trailing byte from odd-length audio payload")
                                audio_field = audio_field[:-1]
                            samples_i16 = np.frombuffer(audio_field, dtype=np.int16)
                        else:
                            logger.warning("Unsupported audio payload type: {}", type(audio_field))
                            continue

                else:
                    audio_field = structured_payload

                    if samplerate != config.SAMPLERATE:
                        raise ValueError("Sample rate mismatch")

                    if isinstance(audio_field, str):
                        try:
                            audio_field = base64.b64decode(audio_field)
                        except Exception:
                            logger.warning("Failed to decode base64 audio payload; ignoring")
                            continue
                        samples_i16 = np.frombuffer(audio_field, dtype=np.int16)
                    elif isinstance(audio_field, list):
                        samples_i16 = np.asarray(audio_field, dtype=np.int16)
                    elif isinstance(audio_field, (bytes, bytearray)):
                        if len(audio_field) % 2 != 0:
                            logger.warning("Dropping trailing byte from odd-length audio payload")
                            audio_field = audio_field[:-1]
                        samples_i16 = np.frombuffer(audio_field, dtype=np.int16)
                    else:
                        logger.warning("Unsupported audio payload type: {}", type(audio_field))
                        continue
            else:
                logger.warning("Received websocket frame without bytes or text payload; ignoring")
                continue

            # Add audio to buffer if we have samples (skip if force cutoff command)
            if samples_i16 is not None and samples_i16.size > 0:
                if encoding and encoding.lower() not in ("s16le", "pcm16"):
                    raise ValueError(f"Unsupported audio encoding: {encoding}")

                samples = samples_i16.astype(np.float32) / 32768.0
                audio_buffer = np.concatenate((audio_buffer, samples))

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
                        in_speech = True  # Track that we're now in speech

                        logger.debug(
                            f"{speech_count}: VAD start: {currentAudioBeginTime}"
                        )
                        await websocket.send_json(
                            VADEvent(
                                is_active=True,
                                segment_start_s=currentAudioBeginTime,
                                session_start_walltime=session_start_walltime,
                                aud_seg_indx = aud_seg_indx
                            ).model_dump()
                        )

                    is_last = "end" in speech_dict

                    for res in sensevoice_model.streaming_inference(
                        speech_samples, is_last
                    ):

                        if len(res["text"]) > 0:
                            asrDetected = True

                        if asrDetected:
                            word_entries = res.get("word_timestamps") or []
                            segment_end_s: float | None = None
                            if word_entries:
                                last_entry = word_entries[-1]
                                last_end_ms = (
                                    last_entry.get("end_ms")
                                    or last_entry.get("end")
                                )
                                if last_end_ms is not None:
                                    segment_end_s = currentAudioBeginTime + (last_end_ms / 1000.0)

                            if segment_end_s is None:
                                timestamps = res.get("timestamps") or []
                                if timestamps:
                                    segment_end_s = currentAudioBeginTime + (
                                        timestamps[-1] / 1000.0
                                    )

                            if is_last and "end" in speech_dict:
                                segment_end_s = speech_dict["end"] / config.SAMPLERATE

                            transcription_response = TranscriptionResponse(
                                id=speech_count,
                                begin_at=currentAudioBeginTime,
                                end_at=segment_end_s,
                                data=TranscriptionChunk(
                                    timestamps=res["timestamps"],
                                    raw_text=res["text"],
                                    word_timestamps=res.get("word_timestamps"),
                                ),
                                is_final=False,
                                session_id=session_id,
                                segment_start_s=currentAudioBeginTime,
                                segment_end_s=segment_end_s,
                                session_start_walltime=session_start_walltime,
                                aud_seg_indx = aud_seg_indx
                            )
                            await websocket.send_json(
                                transcription_response.model_dump()
                            )

                    if is_last:
                        in_speech = False  # Track that speech has ended
                        if asrDetected:
                            speech_count += 1
                            asrDetected = False

                            transcription_response.is_final = True
                            transcription_response.end_at = (
                                speech_dict["end"] / config.SAMPLERATE
                            )
                            transcription_response.segment_end_s = (
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
                            VADEvent(
                                is_active=False,
                                segment_start_s=currentAudioBeginTime,
                                segment_end_s=speech_dict["end"] / config.SAMPLERATE,
                                session_start_walltime=session_start_walltime,
                                aud_seg_indx = aud_seg_indx
                            ).model_dump()
                        )

            # Execute force cutoff after all audio processing in this iteration
            if force_cutoff_pending and current_seg_idx == force_cutoff_target_seg:
                logger.info(f"Executing force cutoff for seg_idx={force_cutoff_target_seg}, in_speech={in_speech}")
                
                # Only cutoff if we're actually in speech
                if in_speech:
                    # Force ASR to return final hypothesis
                    empty_samples = np.array([], dtype=np.float32)
                    for res in sensevoice_model.streaming_inference(empty_samples, is_last=True):
                        if len(res["text"]) > 0 or asrDetected:
                            # Calculate segment end time from word timestamps
                            word_entries = res.get("word_timestamps") or []
                            segment_end_s: float | None = None
                            
                            if word_entries:
                                last_entry = word_entries[-1]
                                last_end_ms = last_entry.get("end_ms") or last_entry.get("end")
                                if last_end_ms is not None:
                                    segment_end_s = currentAudioBeginTime + (last_end_ms / 1000.0)
                            
                            if segment_end_s is None:
                                timestamps = res.get("timestamps") or []
                                if timestamps:
                                    segment_end_s = currentAudioBeginTime + (timestamps[-1] / 1000.0)
                            
                            # Send partial transcript (is_final=False as requested)
                            transcription_response = TranscriptionResponse(
                                id=speech_count,
                                begin_at=currentAudioBeginTime,
                                end_at=segment_end_s,
                                data=TranscriptionChunk(
                                    timestamps=res["timestamps"],
                                    raw_text=res["text"],
                                    word_timestamps=res.get("word_timestamps"),
                                ),
                                is_final=False,  # Don't force finalize
                                session_id=session_id,
                                segment_start_s=currentAudioBeginTime,
                                segment_end_s=segment_end_s,
                                session_start_walltime=session_start_walltime,
                                aud_seg_indx=force_cutoff_target_seg
                            )
                            await websocket.send_json(transcription_response.model_dump())
                            logger.debug(f"Force cutoff: sent partial transcript for seg_idx={force_cutoff_target_seg}")
                    
                    # Emit VAD offset event
                    await websocket.send_json(
                        VADEvent(
                            is_active=False,
                            segment_start_s=currentAudioBeginTime,
                            segment_end_s=segment_end_s,
                            session_start_walltime=session_start_walltime,
                            aud_seg_indx=force_cutoff_target_seg
                        ).model_dump()
                    )
                    logger.debug(f"Force cutoff: sent VAD offset for seg_idx={force_cutoff_target_seg}")
                    
                    # Reset VAD state while preserving timing counters for consistent timestamps using customized reset method
                    reset_vad_state_preserve_timing(vad_iterator)
        
                    # Reset ASR state as well
                    sensevoice_model.reset()
                    
                    # Update state
                    speech_count += 1
                    asrDetected = False
                    in_speech = False
                    
                    logger.info(f"Force cutoff executed: segment ended at {segment_end_s}s")
                
                # Clear flags and discard remaining audio buffer
                force_cutoff_pending = False
                audio_buffer = np.array([], dtype=np.float32)

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
