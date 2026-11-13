#!/usr/bin/env bash
# Clone dependencies, then replace CUDA binaries only in a NEW environment.
set -euo pipefail
asr_conda="${CONDA_EXE:-$HOME/miniconda3/bin/conda}"
asr_base="$("$asr_conda" info --base)"
asr_source="$asr_base/envs/streaming_asr_env"
asr_target="$asr_base/envs/streaming_asr_blackwell"
if [[ -e "$asr_target" ]]; then
    echo "Environment already exists: $asr_target; checking only, no package changes."
else
    echo "Cloning $asr_source to $asr_target (several GB; this can take a few minutes)."
    "$asr_conda" create --yes --prefix "$asr_target" --clone "$asr_source"
    echo "Installing Blackwell CUDA wheels into the new environment only."
    "$asr_target/bin/python" -m pip install \
        torch==2.8.0 torchaudio==2.8.0 \
        --index-url https://download.pytorch.org/whl/cu128
fi
"$asr_target/bin/python" - <<'PY'
import sys
from importlib.util import find_spec
import torch, torchaudio
for module in ('funasr', 'pysilero', 'asr_decoder', 'online_fbank', 'modelscope',
               'fastapi', 'pydantic_settings'):
    # pysilero's import checks the model hub. Actual model/VAD imports belong to
    # the server's cached-model startup, rather than this environment check.
    assert find_spec(module) is not None, f'Missing dependency: {module}'
assert torch.__version__.startswith('2.8.0'), torch.__version__
assert torchaudio.__version__.startswith('2.8.0'), torchaudio.__version__
assert torch.version.cuda == '12.8', torch.version.cuda
assert 'sm_120' in torch.cuda.get_arch_list(), torch.cuda.get_arch_list()
print('Blackwell ASR runtime ready:', sys.executable)
print('PyTorch', torch.__version__, 'TorchAudio', torchaudio.__version__)
PY
