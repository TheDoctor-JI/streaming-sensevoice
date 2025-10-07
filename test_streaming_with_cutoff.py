import asyncio
import websockets
import json
import wave
import numpy as np
import time
import struct
from pathlib import Path
import argparse
from typing import List, Dict, Any
import base64  # Add this import

class StreamingASRTester:
    def __init__(
        self,
        server_url: str = "ws://127.0.0.1:8000/api/realtime/ws",
        chunk_duration: float = 0.1,
        sample_rate: int = 16000,
        force_offset_timing: float = None,
    ):
        # Add query parameters if needed
        if '?' not in server_url:
            server_url = f"{server_url}?chunk_duration={chunk_duration}"
        
        self.server_url = server_url
        self.chunk_duration = chunk_duration
        self.sample_rate = sample_rate
        self.chunk_size_samples = int(chunk_duration * sample_rate)
        self.chunk_size_bytes = self.chunk_size_samples * 2
        self.force_offset_timing = force_offset_timing
        
        self.vad_events: List[Dict[str, Any]] = []
        self.transcription_updates: List[Dict[str, Any]] = []
        self.streamed_audio: List[np.ndarray] = []
        self.total_audio_duration_sent = 0.0
        
    def load_audio_file(self, filepath: str) -> np.ndarray:
        """Load WAV file and return as int16 numpy array"""
        with wave.open(filepath, 'rb') as wf:
            assert wf.getnchannels() == 1, "Only mono audio supported"
            assert wf.getsampwidth() == 2, "Only 16-bit audio supported"
            assert wf.getframerate() == self.sample_rate, f"Sample rate must be {self.sample_rate}"
            
            frames = wf.readframes(wf.getnframes())
            audio_data = np.frombuffer(frames, dtype=np.int16)
            
        return audio_data
    
    def chunk_audio(self, audio_data: np.ndarray) -> List[np.ndarray]:
        """Chunk audio into fixed-size chunks, padding last chunk if necessary"""
        chunks = []
        num_full_chunks = len(audio_data) // self.chunk_size_samples
        
        for i in range(num_full_chunks):
            start = i * self.chunk_size_samples
            end = start + self.chunk_size_samples
            chunks.append(audio_data[start:end])
        
        remaining_samples = len(audio_data) % self.chunk_size_samples
        if remaining_samples > 0:
            last_chunk = audio_data[num_full_chunks * self.chunk_size_samples:]
            padding = np.zeros(self.chunk_size_samples - remaining_samples, dtype=np.int16)
            padded_chunk = np.concatenate([last_chunk, padding])
            chunks.append(padded_chunk)
        
        return chunks

    async def receive_messages(self, websocket):
        """Continuously receive and store messages from server"""
        message_count = 0
        try:
            print("[RECEIVER] Starting to listen for messages...")
            while True:
                message = await websocket.recv()
                message_count += 1
                
                # Handle both binary and text messages
                if isinstance(message, bytes):
                    try:
                        message = message.decode('utf-8')
                    except UnicodeDecodeError:
                        print(f"[RECEIVER] Received non-text binary message #{message_count}")
                        continue
                
                try:
                    data = json.loads(message)
                except json.JSONDecodeError:
                    print(f"[RECEIVER] Received non-JSON message #{message_count}: {message[:100]}")
                    continue
                
                print(f'[RECEIVER] Received message #{message_count}: {data.get("type", "unknown")}')

                message_type = data.get("type")
                
                if message_type == "VADEvent":
                    self.vad_events.append(data)
                    is_active = data.get("is_active")
                    seg_start = data.get("segment_start_s")
                    seg_end = data.get("segment_end_s")
                    seg_idx = data.get("aud_seg_indx")
                    
                    if is_active:
                        print(f"\n[VAD ONSET] Turn {seg_idx} at {seg_start:.3f}s")
                    else:
                        print(f"[VAD OFFSET] Turn {seg_idx} at {seg_end:.3f}s")
                
                elif message_type == "TranscriptionResponse":
                    self.transcription_updates.append(data)
                    seg_idx = data.get("aud_seg_indx")
                    is_final = data.get("is_final")
                    raw_text = data.get("data", {}).get("raw_text", "")
                    final_marker = "[FINAL]" if is_final else "[PARTIAL]"
                    print(f"[TRANSCRIPT {final_marker}] Turn {seg_idx}: {raw_text}")
                
                elif message_type == "Error":
                    print(f"\n[ERROR] {data.get('detail')}")
                    break
                    
        except websockets.exceptions.ConnectionClosed:
            print("\n[RECEIVER] Connection closed by server")
        except Exception as e:
            print(f"\n[RECEIVER] Error: {e}")
            import traceback
            traceback.print_exc()


    async def stream_turn(
        self,
        websocket,
        turn_idx: int,
        chunks: List[np.ndarray],
    ):
        """Stream audio chunks for a single turn"""
        print(f"\n=== Streaming Turn {turn_idx} ===")
        print(f"Total chunks: {len(chunks)}")
                
        for i, chunk in enumerate(chunks):
            current_audio_time = self.total_audio_duration_sent
            
            # Check for force offset
            if self.force_offset_timing is not None and current_audio_time >= self.force_offset_timing:
                print(f"\n>>> Sending force_vad_offset at audio_time={current_audio_time:.3f}s for turn {turn_idx}")
                cutoff_command = {
                    "command": "force_vad_offset",
                    "cutoff_target_seg_idx": str(turn_idx)
                }
                await websocket.send(json.dumps(cutoff_command))
                self.force_offset_timing = None
                await asyncio.sleep(0.01)
                
                print(f"Stopping stream for turn {turn_idx} after chunk {i}/{len(chunks)}")
                print(f"Total audio duration sent before cutoff: {self.total_audio_duration_sent:.3f}s")
                return True
            
            # Prepare audio payload with segment index (like app.py does)
            audio_bytes = chunk.tobytes()
            
            # Encode audio as base64 and send with seg_idx in JSON format
            payload = {
                "seg_idx": str(turn_idx),  # Send the turn index as segment index
                "audio": base64.b64encode(audio_bytes).decode("ascii"),
            }
            
            if i == 0:
                print(f"[DEBUG] First chunk for turn {turn_idx}:")
                print(f"  - Chunk samples: {len(chunk)}")
                print(f"  - Bytes length: {len(audio_bytes)}")
                print(f"  - Sample rate: {self.sample_rate}")
                print(f"  - First 5 samples: {chunk[:5]}")
                print(f"  - Segment index: {turn_idx}")
            
            # Send JSON payload with both audio and segment index
            await websocket.send(json.dumps(payload))
            
            self.streamed_audio.append(chunk)
            self.total_audio_duration_sent += self.chunk_duration
            
            await asyncio.sleep(0.01)  # Faster than realtime streaming
            
            if (i + 1) % 10 == 0 or (i + 1) == len(chunks):
                print(f"Streamed {i + 1}/{len(chunks)} chunks for turn {turn_idx} (audio_time={self.total_audio_duration_sent:.3f}s)")
        
        print(f"Finished streaming turn {turn_idx} (total audio_time={self.total_audio_duration_sent:.3f}s)")
        return False

    async def run_test(self, audio_files: List[str]):
        """Main test execution"""
        print(f"Connecting to {self.server_url}")
        
        all_turns_chunks = []
        for i, filepath in enumerate(audio_files):
            print(f"\nLoading audio file: {filepath}")
            audio_data = self.load_audio_file(filepath)
            chunks = self.chunk_audio(audio_data)
            all_turns_chunks.append(chunks)
            print(f"Turn {i}: {len(audio_data)} samples -> {len(chunks)} chunks")
        
        try:
            async with websockets.connect(
                self.server_url,
                max_size=10 * 1024 * 1024,  # 10MB max message size
                ping_interval=20,
                ping_timeout=10
            ) as websocket:
                print(f"\n✓ WebSocket connection established to {self.server_url}")
                
                if self.force_offset_timing is not None:
                    print(f"Force offset will be sent at audio_time={self.force_offset_timing:.3f}s")
                
                receive_task = asyncio.create_task(self.receive_messages(websocket))
                
                # Give receiver a moment to start
                await asyncio.sleep(0.1)
                
                for turn_idx, chunks in enumerate(all_turns_chunks):
                    cutoff = await self.stream_turn(websocket, turn_idx, chunks)
                    
                    if cutoff:
                        await asyncio.sleep(0.2)
                
                print("\nWaiting for final messages...")
                await asyncio.sleep(2.0)
                
                await websocket.close()
                receive_task.cancel()
                
                try:
                    await receive_task
                except asyncio.CancelledError:
                    pass
                    
        except Exception as e:
            print(f"\n✗ Connection error: {e}")
            import traceback
            traceback.print_exc()

    def save_streamed_audio(self, output_path: str):
        """Save all streamed audio to a WAV file"""
        if not self.streamed_audio:
            print("No audio was streamed")
            return
        
        full_audio = np.concatenate(self.streamed_audio)
        
        with wave.open(output_path, 'wb') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(self.sample_rate)
            wf.writeframes(full_audio.tobytes())
        
        duration = len(full_audio) / self.sample_rate
        print(f"\nSaved streamed audio to: {output_path}")
        print(f"Duration: {duration:.2f}s, Samples: {len(full_audio)}")
    
    def print_results(self):
        """Print formatted results"""
        print("\n" + "="*80)
        print("RESULTS SUMMARY")
        print("="*80)
        
        print("\n--- VAD Events ---")
        for event in self.vad_events:
            is_active = event.get("is_active")
            seg_idx = event.get("aud_seg_indx")
            seg_start = event.get("segment_start_s")
            seg_end = event.get("segment_end_s")
            
            if is_active:
                print(f"Turn {seg_idx}: VAD ONSET  at {seg_start:.3f}s")
            else:
                print(f"Turn {seg_idx}: VAD OFFSET at {seg_end:.3f}s")
        
        print("\n--- Transcriptions ---")
        turns_transcripts = {}
        for trans in self.transcription_updates:
            seg_idx = trans.get("aud_seg_indx", -1)
            if seg_idx not in turns_transcripts:
                turns_transcripts[seg_idx] = []
            turns_transcripts[seg_idx].append(trans)
        
        for seg_idx in sorted(turns_transcripts.keys()):
            print(f"\nTurn {seg_idx}:")
            transcripts = turns_transcripts[seg_idx]
            last_trans = transcripts[-1]
            
            seg_start_s = last_trans.get("segment_start_s", 0.0)
            is_final = last_trans.get("is_final", False)
            raw_text = last_trans.get("data", {}).get("raw_text", "")
            word_timestamps = last_trans.get("data", {}).get("word_timestamps", [])
            
            print(f"  Text: {raw_text}")
            print(f"  Is Final: {is_final}")
            print(f"  Segment Start: {seg_start_s:.3f}s")
            
            if word_timestamps:
                print(f"  Word Timestamps:")
                for word_info in word_timestamps:
                    word = word_info.get("word", "")
                    start_ms = word_info.get("start_ms", 0)
                    end_ms = word_info.get("end_ms")
                    
                    abs_start_s = seg_start_s + (start_ms / 1000.0)
                    
                    if end_ms is not None:
                        abs_end_s = seg_start_s + (end_ms / 1000.0)
                        print(f"    '{word}': {abs_start_s:.3f}s - {abs_end_s:.3f}s")
                    else:
                        print(f"    '{word}': {abs_start_s:.3f}s - (ongoing)")
        
        print("\n" + "="*80)


async def main():
    parser = argparse.ArgumentParser(description="Test streaming ASR with force VAD cutoff")
    parser.add_argument("--server-url", default="ws://127.0.0.1:9903/api/realtime/ws")
    parser.add_argument("--data-dir", default="./data")
    parser.add_argument("--force-offset-timing", type=float, default=None)
    parser.add_argument("--speed", type=float, default=10.0)
    parser.add_argument("--output-audio", default="./data/test_output_streamed_audio.wav")
    
    args = parser.parse_args()
    
    data_dir = Path(args.data_dir)
    audio_files = [
        str(data_dir / "turn_0.wav"),
        str(data_dir / "turn_1.wav")
    ]
    
    for filepath in audio_files:
        if not Path(filepath).exists():
            print(f"Error: Audio file not found: {filepath}")
            return
    
    tester = StreamingASRTester(
        server_url=args.server_url,
        force_offset_timing=args.force_offset_timing,
    )
    
    await tester.run_test(audio_files)
    tester.save_streamed_audio(args.output_audio)
    tester.print_results()


if __name__ == "__main__":
    asyncio.run(main())