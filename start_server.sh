# For releasing ports
kill -9 $(lsof -t -i:9903)

clear
source ~/miniconda3/etc/profile.d/conda.sh
conda activate intelligence_instance_asr_env

python realtime_ws_server_demo.py --host localhost --port 9903 --language auto --chunk_duration 0.1 --device "cuda:1"
