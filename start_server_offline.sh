#!/bin/bash
# Launch ASR server with larger beam size for offline batch transcription.
# Larger beam = fewer fused-word artefacts from CTC boundary drops (~1.7× slower).
#
# Usage: bash start_server_offline.sh

kill -9 $(lsof -t -i:9903) 2>/dev/null

clear
source ~/miniconda3/etc/profile.d/conda.sh
conda activate streaming_asr_env
export CUDA_VISIBLE_DEVICES=0,1
python realtime_ws_server_demo.py \
    --host localhost \
    --port 9903 \
    --language auto \
    --chunk_duration 0.1 \
    --device "cuda:0" \
    --beam_size 5
