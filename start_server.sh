clear
source ~/miniconda3/etc/profile.d/conda.sh
conda activate streaming_asr_env
cd "$(dirname "${BASH_SOURCE[0]}")" || exit 1
exec python ../runtime_launch.py --config launch_config.yaml -- python realtime_ws_server_demo.py --host localhost --port 9903 --language auto --chunk_duration 0.1 --device "cuda:0"
