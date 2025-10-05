"""Utility to stream pre-encoded audio chunks to the realtime SenseVoice ASR server."""
from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from io import BytesIO
from typing import Iterable, List, Optional

import numpy as np
import soundfile as sf
from pydub import AudioSegment
from websocket import WebSocketTimeoutException, create_connection

DEFAULT_MP3_BITRATE = "32k"

logger = logging.getLogger(__name__)


@dataclass
class TranscriptMessage:
    """Container for the latest transcription state of an utterance."""

    id: int
    begin_at: float
    end_at: Optional[float]
    raw_text: str
    timestamps: List[int]
    is_final: bool
    session_id: Optional[str]

    @classmethod
    def from_payload(cls, payload: dict) -> "TranscriptMessage":
        data = payload["data"]
        return cls(
            id=payload["id"],
            begin_at=payload["begin_at"],
            end_at=payload.get("end_at"),
            raw_text=data.get("raw_text", ""),
            timestamps=data.get("timestamps", []),
            is_final=payload.get("is_final", False),
            session_id=payload.get("session_id"),
        )


class AudioStreamerForASR:
    """Streams encoded audio chunks to the streaming ASR websocket."""

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 9933,
        endpoint: str = "/api/realtime/ws",
        chunk_duration_ms: int = 10,
        sample_rate: int = 16000,
        recv_timeout: float = 0.2,
        tail_timeout: float = 2.0,
    ) -> None:
        self.ws_url = f"ws://{host}:{port}{endpoint}"
        self.chunk_duration_ms = chunk_duration_ms
        self.sample_rate = sample_rate
        self.recv_timeout = recv_timeout
        self.tail_timeout = tail_timeout

    def stream_chunks_to_asr(self, audio_chunks: Iterable[bytes]) -> List[TranscriptMessage]:
        """Stream a sequence of encoded audio chunks and collect ASR transcripts.

        Args:
            audio_chunks: Iterable of encoded chunks (e.g., MP3 frames) at
                ``chunk_duration_ms`` cadence. Chunks are forwarded as-is.

        Returns:
            List of ``TranscriptMessage`` entries ordered by utterance id.
        """

        ws = create_connection(self.ws_url)
        ws.settimeout(self.recv_timeout)
        transcript_map: dict[int, TranscriptMessage] = {}

        try:
            for chunk in audio_chunks:
                ws.send_binary(chunk)
                self._drain_socket(ws, transcript_map)

            # After all chunks, allow extra time for trailing hypotheses
            if self.tail_timeout > 0:
                ws.settimeout(self.tail_timeout)
                try:
                    self._drain_socket(ws, transcript_map)
                except WebSocketTimeoutException:
                    pass
        finally:
            ws.close()

        return [transcript_map[key] for key in sorted(transcript_map)]

    def _drain_socket(self, ws, transcript_map: dict[int, TranscriptMessage]) -> None:
        while True:
            try:
                message = ws.recv()
            except WebSocketTimeoutException:
                break
            except Exception as exc:
                logger.warning("Websocket receive failed: %s", exc)
                break

            if isinstance(message, bytes):
                try:
                    message = message.decode("utf-8")
                except UnicodeDecodeError:
                    logger.debug("Received non-text frame; ignoring")
                    continue

            try:
                payload = json.loads(message)
            except json.JSONDecodeError:
                logger.debug("Non-JSON frame: %s", message)
                continue

            msg_type = payload.get("type")
            if msg_type == "TranscriptionResponse":
                transcript_map[payload["id"]] = TranscriptMessage.from_payload(payload)
            elif msg_type == "VADEvent":
                logger.debug("VAD event: %s", payload.get("is_active"))
            else:
                logger.debug("Unhandled message: %s", payload)


# -----------------------------
# Example CLI usage
# -----------------------------

def _split_audio_into_pcm_chunks(
    samples: np.ndarray,
    sample_rate: int,
    chunk_duration_ms: int,
) -> List[bytes]:
    samples = np.asarray(samples, dtype=np.int16)
    chunk_size = int(round(sample_rate * chunk_duration_ms / 1000.0))
    if chunk_size <= 0:
        raise ValueError("chunk_duration_ms results in non-positive chunk size")

    pad_samples = (-len(samples)) % chunk_size
    if pad_samples:
        samples = np.pad(samples, (0, pad_samples), mode="constant")

    chunks: List[bytes] = []
    for start in range(0, len(samples), chunk_size):
        chunk = samples[start : start + chunk_size]
        chunks.append(chunk.tobytes())
    return chunks


def _encode_pcm_chunks_to_mp3(
    pcm_chunks: Iterable[bytes],
    sample_rate: int,
    bitrate: str = DEFAULT_MP3_BITRATE,
) -> List[bytes]:
    encoded: List[bytes] = []
    for chunk in pcm_chunks:
        segment = AudioSegment(
            data=chunk,
            sample_width=2,
            frame_rate=sample_rate,
            channels=1,
        )
        buffer = BytesIO()
        segment.export(buffer, format="mp3", bitrate=bitrate)
        encoded.append(buffer.getvalue())
    return encoded


def _load_audio_chunks(
    audio_path: str,
    sample_rate: int,
    chunk_duration_ms: int,
) -> List[bytes]:
    samples, sr = sf.read(audio_path, dtype="int16")
    if sr != sample_rate:
        raise ValueError(
            f"Sample rate mismatch: expected {sample_rate}, got {sr}"
        )

    if samples.ndim > 1:
        samples = samples[:, 0]

    pcm_chunks = _split_audio_into_pcm_chunks(samples, sample_rate, chunk_duration_ms)
    return _encode_pcm_chunks_to_mp3(pcm_chunks, sample_rate, DEFAULT_MP3_BITRATE)


def _print_transcripts(messages: List[TranscriptMessage]) -> None:
    if not messages:
        print("No transcripts received.")
        return

    for msg in messages:
        time_range = (
            f"[{msg.begin_at:.3f}, {msg.end_at:.3f}]"
            if msg.end_at is not None
            else f"[{msg.begin_at:.3f}, ...]"
        )
        print(f"{msg.id:02d} {time_range} {msg.raw_text}")
        if msg.timestamps:
            print(f"    timestamps(ms): {msg.timestamps}")


def main() -> None:
    logging.basicConfig(level=logging.INFO)

    audio_path = os.path.join(os.path.dirname(__file__), "audio.wav")
    streamer = AudioStreamerForASR()
    print('Loading audo chunks...')
    chunks = _load_audio_chunks(
        audio_path, streamer.sample_rate, streamer.chunk_duration_ms
    )
    print('Start streaming to ASR...')
    transcripts = streamer.stream_chunks_to_asr(chunks)
    _print_transcripts(transcripts)


if __name__ == "__main__":
    main()
