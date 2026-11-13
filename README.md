# streaming-sensevoice

Streaming SenseVoice processes inference in chunks of [SenseVoice](https://github.com/FunAudioLLM/SenseVoice).

## Usage

- transcribe wav file

```bash
$ python main.py
```

![](images/screenshot.png)

## Umbrella launcher and Blackwell

The umbrella launcher's `start_server.sh` entry point applies the existing YAML
GPU selection and two-machine UUID override before choosing an interpreter.
`asr_runtime.py` inspects the actual visible GPU's compute capability. Ada and
older GPUs keep `streaming_asr_env`; Blackwell uses a separate environment.

Create that environment once, on the host running ASR:

```bash
bash setup_blackwell_environment.sh
```

This clones the original environment into `streaming_asr_blackwell`, then installs
matched PyTorch/TorchAudio 2.8.0 CUDA 12.8 wheels **only in the new environment**.
Existing environments are not upgraded. Re-running setup checks the target
without changing packages. Launching never installs packages or switches GPUs.
`STREAMING_ASR_BLACKWELL_PYTHON` may specify a different compatible interpreter.

The server runs an inference warmup before opening its port, so unsupported CUDA
kernels fail startup. After starting the server on port 9903, check actual VAD and
transcription with:

```bash
~/miniconda3/envs/streaming_asr_blackwell/bin/python test_realtime_smoke.py
```

Use `python3 test_asr_runtime.py -q` for lightweight interpreter-selection tests.

- transcribe from microphone

```bash
$ python realtime.py
```

- transcribe from websocket

A basic WebSocket service built with [`Recorder`](https://github.com/xiangyuecn/Recorder) and `FastAPI`; the frontend uses `MP3` format to transmit audio information to reduce latency and increase stability.

```bash
pip install -r requirements-ws-demo.txt
python realtime_ws_server_demo.py

# check cli options
python realtime_ws_server_demo.py --help
```
