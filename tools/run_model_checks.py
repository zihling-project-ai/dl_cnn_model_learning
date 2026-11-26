"""分階段實驗第一步：每個模型獨立檢查，再完整訓練一輪。"""
import argparse
import csv
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
MODELS = ['lenet', 'alexnet', 'alexnet_lrn', 'vgg16', 'vgg16_bn', 'googlenet', 'inception', 'resnet50', 'mobilenet_v1', 'efficientnet_b0']


def save(path, data):
    """先寫暫存檔再替換，避免讀到只寫一半的紀錄。"""
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2))
    temporary.replace(path)


def preflight(name, folder):
    """用小批次檢查模型、梯度與權重存取；會做一次參數更新。"""
    import numpy as np
    import tensorflow as tf
    from models import get_model
    from tools.dataloader import load_train_val_dataset
    from tools.utils import load_config, build_optimizer
    cfg = load_config(ROOT / 'config_lab.yaml')
    tf.keras.utils.set_random_seed(42)
    model = get_model(name, 6, (160, 160, 3), aux_logits=True)
    train, _ = load_train_val_dataset(cfg['dataset']['train_dir'], cfg['dataset']['val_dir'],
                                    2, [160, 160], cfg['augmentation'], 42)
    x, y = train[0]
    optimizer = build_optimizer(cfg['training'])
    with tf.GradientTape() as tape:
        output = model(x, training=True)
        heads = output if isinstance(output, dict) else {'main': output}
        loss = 0.
        for key, probabilities in heads.items():
            tf.debugging.assert_shapes([(probabilities, (2, 6))])
            tf.debugging.assert_all_finite(probabilities, 'non-finite probabilities')
            np.testing.assert_allclose(np.sum(probabilities, axis=1), 1, atol=1e-4)
            loss += (1. if key == 'main' else .3) * tf.reduce_mean(tf.keras.losses.sparse_categorical_crossentropy(y, probabilities))
    gradients = tape.gradient(loss, model.trainable_variables)
    if any(g is None for g in gradients):
        raise ValueError('存在沒有梯度的可訓練參數')
    for gradient in gradients:
        tf.debugging.assert_all_finite(gradient, 'non-finite gradient')
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    before = model(x, training=False)
    before = before['main'] if isinstance(before, dict) else before
    weight = folder / 'temporary_preflight.h5'
    model.save_weights(str(weight))
    model.load_weights(str(weight))
    after = model(x, training=False)
    after = after['main'] if isinstance(after, dict) else after
    np.testing.assert_allclose(before, after, atol=1e-6)
    weight.unlink()
    save(folder / 'preflight.json', {'passed': True, 'parameters': model.count_params(), 'batch': 2,
                                     'loss': float(loss), 'input': [160, 160, 3]})


def run_process(command, folder, phase, memory_limit):
    """執行獨立工作，記錄耗時與記憶體，超限或取消時停止。"""
    start = time.monotonic()
    peak = 0
    reason = None
    with (folder / (phase + '.log')).open('x') as log:
        process = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        while process.poll() is None:
            usage = subprocess.run(['ps', '-o', 'rss=', '-p', str(process.pid)], capture_output=True, text=True)
            rss = int(usage.stdout.strip() or 0) * 1024
            peak = max(peak, rss)
            if (folder.parent / 'stop.flag').exists():
                reason = '使用者停止'
            elif rss > memory_limit:
                reason = '超過記憶體保護上限'
            elif time.monotonic() - start > (300 if phase == 'preflight' else 7200):
                reason = '超過此階段時間上限'
            if reason:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
                break
            time.sleep(2)
    result = {'exit_code': process.returncode, 'seconds': time.monotonic() - start,
              'peak_sampled_rss_bytes': peak, 'stop_reason': reason}
    save(folder / (phase + '_timing.json'), result)
    return result


def main():
    """依序檢查各模型並訓練一輪，紀錄存到新資料夾。"""
    import yaml
    parser = argparse.ArgumentParser()
    parser.add_argument('--preflight', choices=MODELS)
    parser.add_argument('--folder', type=Path, required=True)
    args = parser.parse_args()
    folder = args.folder.resolve()
    if args.preflight:
        preflight(args.preflight, folder)
        return
    folder.mkdir(parents=True, exist_ok=False)
    physical = int(subprocess.check_output(['sysctl', '-n', 'hw.memsize'], text=True))
    memory_limit = int(physical * .60)
    records = []
    save(folder / 'status.json', {'status': 'running', 'pid': os.getpid(), 'memory_limit_bytes': memory_limit})
    # 先檢查所有模型；某個模型失敗不影響其他模型檢查。
    for name in MODELS:
        if (folder / 'stop.flag').exists():
            break
        output = folder / name
        output.mkdir()
        record = {'model': name}
        record['preflight'] = run_process([sys.executable, '-B', '-u', '-m', 'tools.run_model_checks', '--preflight', name, '--folder', str(output)], output, 'preflight', memory_limit)
        records.append(record)
        save(folder / 'results.json', records)
    # 完整一輪使用全部訓練與驗證圖片；不讀測試集，不做 F1 評分。
    for record in records:
        if (folder / 'stop.flag').exists():
            break
        if record['preflight']['exit_code'] != 0:
            continue
        name = record['model']
        output = folder / name
        cfg = yaml.safe_load((ROOT / 'config_lab.yaml').read_text())
        cfg['model']['name'] = name
        cfg['training']['epochs'] = 1
        cfg['save']['log_dir'] = str(output)
        cfg['save']['checkpoint_dir'] = str(ROOT / 'weights' / 'macos_checks' / folder.name / name)
        config = output / 'config.yaml'
        config.write_text(yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False))
        save(folder / 'status.json', {'status': 'training', 'model': name, 'pid': os.getpid(), 'memory_limit_bytes': memory_limit})
        record['training'] = run_process([sys.executable, '-B', '-u', 'main_train.py', '--config', str(config)], output, 'training', memory_limit)
        history = output / (name + '_history.csv')
        if history.exists():
            with history.open() as stream:
                record['history'] = list(csv.DictReader(stream))
        save(folder / 'results.json', records)
    save(folder / 'status.json', {'status': 'stopped' if (folder / 'stop.flag').exists() else 'finished', 'pid': os.getpid()})


if __name__ == '__main__':
    main()
