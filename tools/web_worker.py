# web_worker.py：執行介面指定的單一工作，回報進度並支援取消。
import argparse
import csv
import hashlib
import json
import math
import os
import shutil
from pathlib import Path
import time
import traceback


class Cancelled(Exception):
    """使用者要求取消目前工作。"""


def write_json(path, content):
    """先寫暫存檔再替換，避免介面讀到只寫一半的 JSON。"""
    path = Path(path)
    temporary = path.with_name(path.name + f'.{os.getpid()}.tmp')
    temporary.write_text(json.dumps(content, ensure_ascii=False, allow_nan=False), encoding='utf-8')
    temporary.replace(path)


def check_cancel(folder):
    """每批工作之間檢查取消，不修改既有實驗。"""
    if (Path(folder) / 'cancel.flag').exists():
        raise Cancelled('已取消；本次已完成的紀錄與權重保留，不列為完成實驗。')



def save_milestone(cfg, budget, best_epoch, elapsed):
    """保存當下最佳權重、設定及截至節點的曲線；舊節點不覆寫。"""
    name = cfg['model']['name']
    weight_root = Path(cfg['save']['checkpoint_dir'])
    log_root = Path(cfg['save']['log_dir'])
    destination = weight_root / 'milestones' / f'epoch_{budget}'
    destination.mkdir(parents=True, exist_ok=False)
    weight = weight_root / f'{name}_best.h5'
    shutil.copy2(weight, destination / weight.name)
    shutil.copy2(Path(str(weight) + '.json'), destination / (weight.name + '.json'))
    # CSVLogger 已在本 callback 前寫完本輪。
    history = log_root / f'{name}_history.csv'
    shutil.copy2(history, destination / history.name)
    record = {'budget': budget, 'best_epoch': best_epoch, 'elapsed_seconds': elapsed,
              'weight': str(destination / weight.name), 'history': str(destination / history.name)}
    write_json(destination / 'milestone.json', record)
    records = json.loads((log_root / 'milestones.json').read_text()) if (log_root / 'milestones.json').exists() else {}
    records[str(budget)] = record
    write_json(log_root / 'milestones.json', records)



def validate_completed_history(cfg):
    """完整跑完指定輪數且每個指標有限，才可標為完成，不以 fit 正常返回判定。"""
    history = Path(cfg['save']['log_dir']) / (cfg['model']['name'] + '_history.csv')
    with history.open(newline='') as stream:
        rows = list(csv.DictReader(stream))
    expected = cfg['training']['epochs']
    if len(rows) != expected:
        raise ValueError(f'訓練提早停止：只記錄 {len(rows)}/{expected} 輪，不能標為完成。')
    for index, row in enumerate(rows):
        if int(float(row['epoch'])) != index or not all(math.isfinite(float(value)) for value in row.values()):
            raise ValueError('訓練紀錄含 NaN／Inf 或輪次不完整，不能標為完成。')


def train(job, report):
    """透過原本的訓練入口執行，另外加入介面進度回報。"""
    import tensorflow as tf
    from main_train import main
    from tools.utils import load_config
    cfg = load_config(job['config'])
    for filename, expected in cfg.get('lab', {}).get('source_sha256', {}).items():
        source = Path(job['project']) / filename
        if not source.is_file() or hashlib.sha256(source.read_bytes()).hexdigest() != expected:
            raise ValueError('比較基準已變更，停止此設定；請建立新的比較排程。')

    class Progress(tf.keras.callbacks.Callback):
        def __init__(self):
            super().__init__()
            self.best_score = -float('inf')
            self.best_epoch = None
            self.started = time.time()

        def on_epoch_begin(self, epoch, logs=None):
            check_cancel(job['folder'])
            self.epoch = epoch + 1
            report(phase='training', epoch=self.epoch, batch=0, batches=self.params.get('steps', 0))

        def on_train_batch_end(self, batch, logs=None):
            check_cancel(job['folder'])
            if not all(math.isfinite(float(value)) for value in (logs or {}).values()):
                raise ValueError('訓練批次出現 NaN／Inf，停止並保留失敗紀錄。')
            if batch == 0 or (batch + 1) % 10 == 0:
                report(phase='training', epoch=self.epoch, batch=batch + 1,
                       batches=self.params.get('steps', 0))

        def on_test_begin(self, logs=None):
            check_cancel(job['folder'])
            report(phase='validation', batch=0)

        def on_test_batch_end(self, batch, logs=None):
            check_cancel(job['folder'])
            if batch % 10 == 0:
                report(phase='validation', batch=batch + 1)

        def on_epoch_end(self, epoch, logs=None):
            check_cancel(job['folder'])
            if not all(math.isfinite(float(value)) for value in (logs or {}).values()):
                raise ValueError('訓練出現非有限指標，請檢查設定與資料。')
            key = 'val_main_accuracy' if cfg['model']['name'] == 'googlenet' and cfg['model'].get('aux_logits') else 'val_accuracy'
            score = float(logs[key])
            if score > self.best_score:
                self.best_score, self.best_epoch = score, epoch + 1
            if cfg.get('lab', {}).get('milestones') and epoch + 1 in {1, 5, 10}:
                # ModelCheckpoint 在本 callback 前執行；複製的是截至此刻最佳，不偷看未來。
                save_milestone(cfg, epoch + 1, self.best_epoch, time.time() - self.started)
            report(epoch=epoch + 1, epochs_completed=epoch + 1)

    check_cancel(job['folder'])
    started = time.time()
    main(job['config'], extra_callbacks=[Progress()])
    check_cancel(job['folder'])
    if cfg.get('lab'):
        validate_completed_history(cfg)
    report(training_completed=True, training_seconds=time.time() - started)
    if cfg.get('lab', {}).get('auto_evaluate_best', False):
        weight = Path(cfg['save']['checkpoint_dir']) / (cfg['model']['name'] + '_best.h5')
        tf.keras.backend.clear_session()
        evaluate_validation(dict(job, weights=str(weight)), report)


def evaluate_validation(job, report):
    """只讀驗證集，沿用既有評分函式並保存逐圖預測。"""
    import tensorflow as tf
    from main_test import evaluate_batches
    from models import get_model
    from tools.dataloader import load_test_dataset
    from tools.utils import validate_config, validate_run_metadata

    folder = Path(job['folder'])
    weight = Path(job['weights'])
    metadata = json.loads(Path(str(weight) + '.json').read_text())
    cfg = metadata['config']
    validate_config(cfg)
    names = validate_run_metadata(cfg, metadata)
    # 即使共用讀圖函式叫 load_test_dataset，這裡明確且只傳入 val_dir。
    validation = Path(cfg['dataset']['val_dir']).resolve()
    expected = Path(job['project']) / 'data_animal_pictures' / 'val'
    if validation != expected.resolve():
        raise ValueError('介面只允許評估去重資料的驗證集。')
    dataset = load_test_dataset(str(validation), cfg['model']['input_size'],
                               cfg['training']['batch_size'], class_names=names)
    check_cancel(folder)
    report(phase='loading_model', batch=0, batches=len(dataset))
    model = get_model(cfg['model']['name'], cfg['model']['num_classes'],
                      tuple(cfg['model']['input_size']) + (3,),
                      aux_logits=cfg['model'].get('aux_logits', False))
    model(tf.zeros((1,) + tuple(cfg['model']['input_size']) + (3,)), training=False)
    model.load_weights(str(weight))

    def progress(batch, total):
        check_cancel(folder)
        if batch == total or batch % 5 == 0:
            report(phase='validation', batch=batch, batches=total)

    result = evaluate_batches(model, dataset, include_predictions=True, on_batch_end=progress)
    check_cancel(folder)
    result.update(split='validation', dataset_dir=str(validation), weights=str(weight),
                  run_id=job['run_id'], evaluated_at=time.time(), settings_verified=True)
    write_json(folder / 'validation.json', result)
    report(result_file=str(folder / 'validation.json'))


def main(job_path):
    """執行已建立的工作，不自行開始下一個實驗。"""
    path = Path(job_path).resolve()
    job = json.loads(path.read_text())
    started = time.time()
    status = dict(job, status='running', pid=os.getpid(), started_at=started,
                  phase='preparing', epochs_completed=0, batch=0)

    def report(**changes):
        status.update(changes, elapsed_seconds=time.time() - started)
        write_json(path, status)

    report()
    try:
        check_cancel(job['folder'])
        if job['kind'] == 'train':
            train(job, report)
        elif job['kind'] == 'validation':
            evaluate_validation(job, report)
        else:
            raise ValueError('不支援的工作')
        check_cancel(job['folder'])
        report(status='completed', phase='completed', finished_at=time.time())
    except Cancelled as error:
        report(status='cancelled', phase='cancelled', message=str(error), finished_at=time.time())
    except BaseException as error:
        traceback.print_exc()
        report(status='failed', phase='failed', message=str(error), finished_at=time.time())
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='執行 CNN Lab 的單一背景工作')
    parser.add_argument('--job', type=Path, required=True)
    main(parser.parse_args().job)
