"""Choose an ASR interpreter after runtime_launch applies GPU visibility.

No packages are installed at launch. The original interpreter probes the actual
logical device (including UUID overrides) without running unsupported kernels.
"""
import os
from pathlib import Path
import sys
import warnings


def choose_python(capability, legacy_python, environ):
    if capability[0] < 10:
        return Path(legacy_python)
    override = environ.get('STREAMING_ASR_BLACKWELL_PYTHON')
    if override:
        return Path(override).expanduser()
    return Path(legacy_python).resolve().parents[2] / 'streaming_asr_blackwell' / 'bin' / 'python'


def main():
    import torch

    # The server's --device cuda:0 refers to the first *visible* device, not
    # physical GPU 0. runtime_launch has already applied the dashboard override.
    with warnings.catch_warnings():
        # The legacy interpreter only queries properties, then exits via exec.
        # Its known kernel warning does not describe the selected runtime.
        warnings.filterwarnings('ignore', category=UserWarning,
                                message='(?s).*is not compatible with the current PyTorch installation.*')
        capability = torch.cuda.get_device_capability(0)
    interpreter = choose_python(capability, sys.executable, os.environ)
    if not interpreter.is_file() or not os.access(interpreter, os.X_OK):
        raise SystemExit(
            f'ASR GPU capability {capability} requires the isolated Blackwell runtime. '
            f'Missing interpreter: {interpreter}. Run bash setup_blackwell_environment.sh '
            'once, or set STREAMING_ASR_BLACKWELL_PYTHON to a compatible ASR interpreter.'
        )
    print(f'[ASR runtime] visible GPU={torch.cuda.get_device_name(0)} '
          f'capability={capability}; python={interpreter}; '
          f'CUDA_VISIBLE_DEVICES={os.environ.get("CUDA_VISIBLE_DEVICES")}', flush=True)
    # exec preserves the launcher-owned PID and every GPU/network setting.
    os.execv(str(interpreter), [str(interpreter), *sys.argv[1:]])


if __name__ == '__main__':
    main()
