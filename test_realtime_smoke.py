"""Actual websocket VAD/transcription check against an already running server."""
import argparse
import asyncio
import json
from pathlib import Path
import wave

import websockets


async def check(url, fixture):
    with wave.open(str(fixture)) as audio:
        assert (audio.getframerate(), audio.getnchannels(), audio.getsampwidth()) == (16000, 1, 2)
        pcm = audio.readframes(audio.getnframes()) + bytes(32000 * 2)
    messages = []
    async with websockets.connect(url, open_timeout=30, proxy=None) as ws:
        async def receive():
            async for raw in ws:
                message = json.loads(raw)
                messages.append(message)
                print(json.dumps(message, ensure_ascii=False), flush=True)

        reader = asyncio.create_task(receive())
        try:
            for offset in range(0, len(pcm), 3200):
                await ws.send(pcm[offset:offset + 3200])
                await asyncio.sleep(0.1)
            await asyncio.sleep(2)
        finally:
            reader.cancel()
            await asyncio.gather(reader, return_exceptions=True)
    vad = [m for m in messages if m.get('type') == 'VADEvent']
    final = [m for m in messages if m.get('type') == 'TranscriptionResponse'
             and m.get('is_final') and m.get('data', {}).get('raw_text')]
    assert any(m.get('is_active') is True for m in vad), 'No VAD onset'
    assert any(m.get('is_active') is False for m in vad), 'No VAD offset'
    assert final, 'No final nonempty transcript'
    print(f'PASS: {len(vad)} VAD events and {len(final)} final transcript(s)')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--url', default='ws://localhost:9903/api/realtime/ws?language=auto')
    parser.add_argument('--wav', type=Path, default=Path(__file__).parent / 'data/test_16k.wav')
    args = parser.parse_args()
    asyncio.run(check(args.url, args.wav))
