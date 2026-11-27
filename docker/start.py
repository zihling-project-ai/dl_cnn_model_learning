"""先確認 NVIDIA GPU 可用，釋放檢查程序，再開啟介面；不開始訓練。"""
import json
import os
from pathlib import Path
import subprocess
import sys

root = Path(__file__).resolve().parents[1]
probe = """
import json
import tensorflow as tf
gpus = tf.config.list_physical_devices('GPU')
if not tf.test.is_built_with_cuda() or not gpus:
    raise SystemExit('找不到可用 NVIDIA CUDA GPU；請檢查驅動與 Docker GPU 設定。')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
with tf.device('/GPU:0'):
    value = tf.linalg.matmul(tf.ones((2, 2)), tf.ones((2, 2)))
if 'GPU:0' not in value.device or float(value[0, 0]) != 2.0:
    raise SystemExit('GPU 計算檢查未通過。')
print(json.dumps({'tensorflow': tf.__version__, 'backend': 'cuda',
                  'label': 'NVIDIA CUDA GPU',
                  'gpu': tf.config.experimental.get_device_details(gpus[0]).get('device_name', 'NVIDIA')}))
"""
result = subprocess.run([sys.executable, '-c', probe], capture_output=True, text=True)
if result.returncode:
    raise SystemExit(result.stderr or result.stdout)
execution = json.loads(result.stdout.strip().splitlines()[-1])
# 代表通過裝置檢查；各模型完整訓練仍須在 NVIDIA 硬體驗證。
sys.path.insert(0, str(root))
from tools.web_app import MODELS
folder = root / 'logs'
folder.mkdir(exist_ok=True)
(folder / 'nvidia_runtime.json').write_text(json.dumps({
    'enabled': True, 'dispatch_models': list(MODELS), 'execution': execution,
    'validation_scope': 'GPU 計算檢查；尚非各模型訓練驗證',
}, ensure_ascii=False, indent=2))
print('GPU 檢查通過。瀏覽器開啟 http://127.0.0.1:8766；按開始才訓練。', flush=True)
os.execv(sys.executable, [sys.executable, '-B', '-u', '-m', 'tools.web_app'])
