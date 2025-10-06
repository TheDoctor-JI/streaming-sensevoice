# For releasing ports
kill -9 $(lsof -t -i:9903)

clear
source ~/miniconda3/etc/profile.d/conda.sh
conda activate streaming_asr_env
export CUDA_VISIBLE_DEVICES=0,1 ##Somehow, without this, the code fails to load to the right gppu
python realtime_ws_server_demo.py --host localhost --port 9903 --language auto --chunk_duration 0.1 --device "cuda:0"
