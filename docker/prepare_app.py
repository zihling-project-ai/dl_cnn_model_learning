"""建置時適配映像內的 web_app.py，不修改主機上的原始碼。"""
from pathlib import Path


def prepare(source):
    changes = {
        "self.root / 'logs/macos_metal_checks/runtime.json'": "self.root / 'logs/nvidia_runtime.json'",
        "self.root / '.venv-macos-metal/bin/python'": "Path(sys.executable)",
        "ThreadingHTTPServer(('127.0.0.1', port), Handler)": "ThreadingHTTPServer(('0.0.0.0', port), Handler)",
        "{f'127.0.0.1:{port}', f'localhost:{port}'}": "{f'127.0.0.1:{port}', f'localhost:{port}', '127.0.0.1:8766', 'localhost:8766'}",
        "{f'http://127.0.0.1:{port}', f'http://localhost:{port}'}": "{f'http://127.0.0.1:{port}', f'http://localhost:{port}', 'http://127.0.0.1:8766', 'http://localhost:8766'}",
    }
    changes.update({
        "runtime.get('validated_models', [])": "runtime.get('dispatch_models', [])",
        "'Metal GPU（已驗證的後續訓練）'": "'NVIDIA CUDA GPU（裝置檢查；模型待驗證）'",
    })
    for old, new in changes.items():
        if old not in source:
            raise ValueError(f'啟動程式已變更，請重新確認 Docker 適配：{old}')
        source = source.replace(old, new)
    return source


if __name__ == '__main__':
    path = Path('/app/tools/web_app.py')
    path.write_text(prepare(path.read_text(encoding='utf-8')), encoding='utf-8')
