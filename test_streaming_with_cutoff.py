import asyncio
import websockets
import json
import wave
import numpy as np
import time
import base64
from pathlib import Path
import argparse
from typing import List, Dict, Any

class StreamingASRTester:
    def __init__(
        self,
        server_url: str = "ws://127.0.0.1:8000/api/realtime/ws",
        chunk_duration: float = 0.1,
        sample_rate: int = 16000,
        force_offset_timing: float = None,  # Seconds after connection to send force offset
        streaming_speed_multiplier: float = 10.0,  # Stream faster than realtime
    ):
        self.server_url = server_url
        self.chunk_duration = chunk_duration
        self.sample_rate = sample_rate
        self.chunk_size_samples = int(chunk_duration * sample_rate)
        self.chunk_size_bytes = self.chunk_size_samples * 2  # 2 bytes per s16le sample
        self.force_offset_timing = force_offset_timing
        self.streaming_speed_multiplier = streaming_speed_multiplier
        
        # Storage for results
        self.vad_events: List[Dict[str, Any]] = []
        self.transcription_updates: List[Dict[str, Any]] = []
        self.streamed_audio: List[np.ndarray] = []
        self.total_audio_duration_sent = 0.0  # Track audio duration sent to server
        
    def load_audio_file(self, filepath: str) -> np.ndarray:
        """Load WAV file and return as int16 numpy array"""
        with wave.open(filepath, 'rb') as wf:
            # Verify format
            assert wf.getnchannels() == 1, "Only mono audio supported"
            assert wf.getsampwidth() == 2, "Only 16-bit audio supported"
            assert wf.getframerate() == self.sample_rate, f"Sample rate must be {self.sample_rate}"
            
            # Read all frames
            frames = wf.readframes(wf.getnframes())
            audio_data = np.frombuffer(frames, dtype=np.int16)
            
        return audio_data
    
    def chunk_audio(self, audio_data: np.ndarray) -> List[np.ndarray]:
        """Chunk audio into fixed-size chunks, padding last chunk if necessary"""
        chunks = []
        num_full_chunks = len(audio_data) // self.chunk_size_samples
        
        # Process full chunks
        for i in range(num_full_chunks):
            start = i * self.chunk_size_samples
            end = start + self.chunk_size_samples
            chunks.append(audio_data[start:end])
        
        # Handle remaining samples
        remaining_samples = len(audio_data) % self.chunk_size_samples
        if remaining_samples > 0:
            last_chunk = audio_data[num_full_chunks * self.chunk_size_samples:]
            # Pad with zeros (silence)
            padding = np.zeros(self.chunk_size_samples - remaining_samples, dtype=np.int16)
            padded_chunk = np.concatenate([last_chunk, padding])
            chunks.append(padded_chunk)
        
        return chunks

    async def receive_messages(self, websocket):
        """Continuously receive and store messages from server"""
        try:
            while True:
                message = await websocket.recv()
                data = json.loads(message)
                
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
            print("\n[Connection closed by server]")

    async def stream_turn(
        self,
        websocket,
        turn_idx: int,
        chunks: List[np.ndarray],
    ):
        """Stream audio chunks for a single turn"""
        print(f"\n=== Streaming Turn {turn_idx} ===")
        print(f"Total chunks: {len(chunks)}")
        
        chunk_interval = self.chunk_duration / self.streaming_speed_multiplier
        
        for i, chunk in enumerate(chunks):
            # Calculate current audio time based on chunks sent (how server measures time)
            current_audio_time = self.total_audio_duration_sent
            
            # Check if we should send force offset command (based on audio duration sent)
            if self.force_offset_timing is not None and current_audio_time >= self.force_offset_timing:
                print(f"\n>>> Sending force_vad_offset at audio_time={current_audio_time:.3f}s for turn {turn_idx}")
                cutoff_command = {
                    "command": "force_vad_offset",
                    "cutoff_target_seg_idx": str(turn_idx)
                }
                await websocket.send(json.dumps(cutoff_command))
                
                # Clear the timing so we don't send it again
                self.force_offset_timing = None
                
                # Wait a bit for server to process
                await asyncio.sleep(0.1)
                
                # Stop streaming this turn
                print(f"Stopping stream for turn {turn_idx} after chunk {i}/{len(chunks)}")
                print(f"Total audio duration sent before cutoff: {self.total_audio_duration_sent:.3f}s")
                return True  # Indicate cutoff happened
            
            # Prepare payload with segment index (base64 encode audio bytes)
            payload = {
                "audio": base64.b64encode(chunk.tobytes()).decode('utf-8'),
                "sr": self.sample_rate,
                "enc": "s16le",
                "seg_idx": turn_idx
            }
            
            # Send as JSON
            await websocket.send(json.dumps(payload))
            
            # Track streamed audio and update duration
            self.streamed_audio.append(chunk)
            self.total_audio_duration_sent += self.chunk_duration
            
            # Wait before sending next chunk (simulating streaming)
            await asyncio.sleep(chunk_interval)
            
            print(f"Streamed {i + 1}/{len(chunks)} chunks for turn {turn_idx} (audio_time={self.total_audio_duration_sent:.3f}s)")
        
        print(f"Finished streaming turn {turn_idx} (total audio_time={self.total_audio_duration_sent:.3f}s)")
        return False  # No cutoff happened

    async def run_test(self, audio_files: List[str]):
        """Main test execution"""
        print(f"Connecting to {self.server_url}")
        
        # Load and chunk all audio files
        all_turns_chunks = []
        for i, filepath in enumerate(audio_files):
            print(f"\nLoading audio file: {filepath}")
            audio_data = self.load_audio_file(filepath)
            chunks = self.chunk_audio(audio_data)
            all_turns_chunks.append(chunks)
            print(f"Turn {i}: {len(audio_data)} samples -> {len(chunks)} chunks")
        
        # Connect to server
        async with websockets.connect(self.server_url) as websocket:
            print(f"\nConnection established")
            
            if self.force_offset_timing is not None:
                print(f"Force offset will be sent at audio_time={self.force_offset_timing:.3f}s")
            
            # Start receiving messages in background
            receive_task = asyncio.create_task(self.receive_messages(websocket))
            
            # Stream each turn
            for turn_idx, chunks in enumerate(all_turns_chunks):
                # Stream the turn (force offset timing is checked within)
                cutoff = await self.stream_turn(
                    websocket,
                    turn_idx,
                    chunks,
                )
                
                if cutoff:
                    # Small delay before starting next turn
                    await asyncio.sleep(0.2)
            
            # Wait a bit for final messages
            print("\nWaiting for final messages...")
            await asyncio.sleep(2.0)
            
            # Close connection
            await websocket.close()
            receive_task.cancel()
            
            try:
                await receive_task
            except asyncio.CancelledError:
                pass

    def save_streamed_audio(self, output_path: str):
        """Save all streamed audio to a WAV file"""
        if not self.streamed_audio:
            print("No audio was streamed")
            return
        
        # Concatenate all chunks
        full_audio = np.concatenate(self.streamed_audio)
        
        # Save as WAV
        with wave.open(output_path, 'wb') as wf:
            wf.setnchannels(1)  # Mono
            wf.setsampwidth(2)  # 16-bit
            wf.setframerate(self.sample_rate)
            wf.writeframes(full_audio.tobytes())
        
        duration = len(full_audio) / self.sample_rate
        print(f"\nSaved streamed audio to: {output_path}")
        print(f"Duration: {duration:.2f}s, Samples: {len(full_audio)}")
    
    def print_results(self):
        """Print formatted results for comparison with audio"""
        print("\n" + "="*80)
        print("RESULTS SUMMARY")
        print("="*80)
        
        print("\n--- VAD Events (relative to WS start) ---")
        for event in self.vad_events:
            is_active = event.get("is_active")
            seg_idx = event.get("aud_seg_indx")
            seg_start = event.get("segment_start_s")
            seg_end = event.get("segment_end_s")
            
            if is_active:
                print(f"Turn {seg_idx}: VAD ONSET  at {seg_start:.3f}s")
            else:
                print(f"Turn {seg_idx}: VAD OFFSET at {seg_end:.3f}s")
        
        print("\n--- Word-Level Timestamps (relative to WS start) ---")
        # Group transcriptions by turn
        turns_transcripts = {}
        for trans in self.transcription_updates:
            seg_idx = trans.get("aud_seg_indx", -1)
            if seg_idx not in turns_transcripts:
                turns_transcripts[seg_idx] = []
            turns_transcripts[seg_idx].append(trans)
        
        for seg_idx in sorted(turns_transcripts.keys()):
            print(f"\nTurn {seg_idx}:")
            
            # Get the last (most complete) transcription for this turn
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
                    
                    # Convert to absolute time (relative to WS start)
                    abs_start_s = seg_start_s + (start_ms / 1000.0)
                    
                    if end_ms is not None:
                        abs_end_s = seg_start_s + (end_ms / 1000.0)
                        print(f"    '{word}': {abs_start_s:.3f}s - {abs_end_s:.3f}s")
                    else:
                        print(f"    '{word}': {abs_start_s:.3f}s - (ongoing)")
            else:
                print(f"  No word timestamps available")
        
        print("\n" + "="*80)


async def main():
    parser = argparse.ArgumentParser(description="Test streaming ASR with force VAD cutoff")
    parser.add_argument(
        "--server-url",
        default="ws://127.0.0.1:9903/api/realtime/ws",
        help="WebSocket server URL"
    )
    parser.add_argument(
        "--data-dir",
        default="./data",
        help="Directory containing audio files"
    )
    parser.add_argument(
        "--force-offset-timing",
        type=float,
        default=None,
        help="Seconds after connection start to send force offset command"
    )
    parser.add_argument(
        "--speed",
        type=float,
        default=10.0,
        help="Streaming speed multiplier (10.0 = 10x faster than realtime)"
    )
    parser.add_argument(
        "--output-audio",
        default="./data/test_output_streamed_audio.wav",
        help="Path to save streamed audio"
    )
    
    args = parser.parse_args()
    
    # Find audio files
    data_dir = Path(args.data_dir)
    audio_files = [
        str(data_dir / "turn_0.wav"),
        str(data_dir / "turn_1.wav")
    ]
    
    # Verify files exist
    for filepath in audio_files:
        if not Path(filepath).exists():
            print(f"Error: Audio file not found: {filepath}")
            return
    
    # Create tester
    tester = StreamingASRTester(
        server_url=args.server_url,
        force_offset_timing=args.force_offset_timing,
        streaming_speed_multiplier=args.speed
    )
    
    # Run test
    await tester.run_test(audio_files)
    
    # Save results
    tester.save_streamed_audio(args.output_audio)
    tester.print_results()


if __name__ == "__main__":
    asyncio.run(main())