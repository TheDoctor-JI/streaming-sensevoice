import unittest
from pathlib import Path
from unittest.mock import Mock, patch
import sys

import asr_runtime


class RuntimeSelectionTests(unittest.TestCase):
    def test_ada_keeps_original_environment(self):
        self.assertEqual(asr_runtime.choose_python(
            (8, 9), '/conda/envs/streaming_asr_env/bin/python', {}),
            Path('/conda/envs/streaming_asr_env/bin/python'))

    def test_blackwell_uses_sibling_environment(self):
        self.assertEqual(asr_runtime.choose_python(
            (12, 0), '/conda/envs/streaming_asr_env/bin/python', {}),
            Path('/conda/envs/streaming_asr_blackwell/bin/python'))

    def test_override_only_applies_to_blackwell(self):
        env = {'STREAMING_ASR_BLACKWELL_PYTHON': '/custom/bin/python'}
        self.assertEqual(asr_runtime.choose_python((12, 0), '/old/python', env), Path('/custom/bin/python'))
        self.assertEqual(asr_runtime.choose_python((8, 9), '/old/python', env), Path('/old/python'))

    def test_exec_keeps_effective_gpu_override(self):
        torch = Mock()
        torch.cuda.get_device_capability.return_value = (12, 0)
        env = {'CUDA_VISIBLE_DEVICES': 'GPU-selected',
               'STREAMING_ASR_BLACKWELL_PYTHON': sys.executable}
        with patch.dict(sys.modules, {'torch': torch}), patch.dict('os.environ', env), \
                patch.object(sys, 'argv', ['asr_runtime.py', 'server.py', '--device', 'cuda:0']), \
                patch('os.execv') as launch:
            asr_runtime.main()
            self.assertEqual(launch.call_args.args, (sys.executable,
                [sys.executable, 'server.py', '--device', 'cuda:0']))
            import os
            self.assertEqual(os.environ['CUDA_VISIBLE_DEVICES'], 'GPU-selected')
            torch.cuda.get_device_capability.assert_called_once_with(0)

    def test_missing_blackwell_runtime_fails_without_fallback(self):
        torch = Mock()
        torch.cuda.get_device_capability.return_value = (12, 0)
        with patch.dict(sys.modules, {'torch': torch}), \
                patch.dict('os.environ', {'STREAMING_ASR_BLACKWELL_PYTHON': '/missing/asr/python'}), \
                patch('os.execv') as launch:
            with self.assertRaisesRegex(SystemExit, 'setup_blackwell_environment.sh'):
                asr_runtime.main()
            launch.assert_not_called()


if __name__ == '__main__':
    unittest.main()
