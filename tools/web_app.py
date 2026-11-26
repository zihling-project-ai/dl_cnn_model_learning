# web_app.py：只在本機提供 CNN Lab，管理實驗、取消與驗證結果。
import argparse
import csv
import fcntl
import hashlib
import itertools
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import io
import json
import math
import os
from pathlib import Path
import re
import secrets
import subprocess
import sys
import threading
import time
from urllib.parse import parse_qs, urlsplit
import uuid

import yaml
from tools.web_worker import write_json

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODELS = {'lenet': 'LeNet', 'alexnet': 'AlexNet', 'alexnet_lrn': 'AlexNet LRN',
          'vgg16': 'VGG16', 'vgg16_bn': 'VGG16 BN', 'googlenet': 'GoogLeNet',
          'inception': 'Inception', 'resnet50': 'ResNet50',
          'mobilenet_v1': 'MobileNet V1', 'efficientnet_b0': 'EfficientNet-B0'}
ACTIVE = {'queued', 'running', 'cancel_requested'}


def inside(path, parent):
    """只接受指定資料夾內的真實路徑。"""
    path, parent = Path(path).resolve(), Path(parent).resolve()
    if path == parent or parent in path.parents:
        return path
    raise ValueError('路徑不在允許範圍')


def read_json(path, default=None):
    """讀取已完成寫入的 JSON，缺檔時回傳預設值。"""
    try:
        return json.loads(Path(path).read_text(encoding='utf-8'))
    except (FileNotFoundError, json.JSONDecodeError):
        return default


def history_rows(path):
    """讀取逐輪紀錄，略過背景工作尚未寫完的一行。"""
    rows = []
    if not Path(path).is_file():
        return rows
    with Path(path).open(newline='') as stream:
        for row in csv.DictReader(stream):
            try:
                values = {key: float(value) for key, value in row.items() if key is not None}
                if not all(math.isfinite(value) for value in values.values()):
                    continue
                if not {'epoch', 'loss'} <= values.keys():
                    continue
                values['epoch'] = int(values['epoch']) + 1
                rows.append(values)
            except (TypeError, ValueError):
                continue
    return rows


def protocol_sources(root):
    """記錄模型、訓練工具、資料清單與基準設定的版本，不讀取測試圖片。"""
    root = Path(root)
    files = list((root / 'models').glob('*.py'))
    files += [root / name for name in ['main_train.py', 'main_test.py', 'tools/utils.py',
              'tools/dataloader.py', 'tools/web_worker.py', 'config_lab.yaml',
              'data_animal_pictures/manifest.csv', 'data_animal_pictures/summary.json']]
    return {str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(files) if path.is_file()}



def training_signature(sources):
    """比較影響數值的程式與資料版本；派工和節點複製不屬於模型訓練變更。"""
    core = {name: digest for name, digest in sources.items() if name != 'tools/web_worker.py'}
    return hashlib.sha256(json.dumps(core, sort_keys=True).encode()).hexdigest()


def training_config(root, values, identifier, mode):
    """驗證使用者選項並產生獨立設定，兩種模式均固定Batch32與相同資料。"""
    if set(values) != {'model', 'epochs', 'batch_size', 'learning_rate', 'optimizer'}:
        raise ValueError('訓練設定欄位不完整或含有不支援欄位')
    if values['model'] not in MODELS or values['optimizer'] not in {'adam', 'sgd', 'adamw', 'rmsprop'}:
        raise ValueError('不支援的模型或優化器')
    if type(values['epochs']) is not int or values['epochs'] not in {1, 5, 10}:
        raise ValueError('Epoch只能選1、5、10')
    if type(values['batch_size']) is not int or values['batch_size'] != 32:
        raise ValueError('Batch size固定32')
    if type(values['learning_rate']) not in {int, float} or values['learning_rate'] not in {.00001, .0001, .001}:
        raise ValueError('學習率只能選0.00001、0.0001、0.001')
    if mode not in {'free', 'comparison'} or (mode == 'comparison' and values['epochs'] != 10):
        raise ValueError('固定比較使用10輪')
    root = Path(root)
    config = yaml.safe_load((root / 'config_lab.yaml').read_text())
    config['model']['name'] = values['model']
    config['training'].update({key: values[key] for key in ('epochs', 'batch_size', 'learning_rate', 'optimizer')})
    config['training']['early_stopping'] = False
    config['dataset'] = {split + '_dir': str(root / 'data_animal_pictures' / split)
                         for split in ['train', 'val', 'test']}
    config['save'] = {'checkpoint_dir': str(root / 'weights/macos_ui' / identifier),
                      'log_dir': str(root / 'logs/macos_ui' / identifier)}
    if not all(Path(config['dataset'][key]).is_dir() for key in ('train_dir', 'val_dir')):
        raise ValueError('找不到去重資料的訓練或驗證資料夾')
    sources = protocol_sources(root)
    protocol_id = training_signature(sources)
    config['lab'] = {'mode': mode, 'protocol_id': protocol_id, 'source_sha256': sources,
                     'milestones': values['epochs'] == 10, 'best_monitor': 'validation_accuracy', 'auto_evaluate_best': values['epochs'] == 10,
                     'note': '從零訓練；學習版架構；測試集保留到最後'}
    return config


class Lab:
    """一次管理一個背景工作，不在伺服器程序內載入 TensorFlow。"""
    def __init__(self, root=PROJECT_ROOT):
        self.root = Path(root).resolve()
        self.folder = self.root / 'logs' / 'macos_ui'
        self.folder.mkdir(parents=True, exist_ok=True)
        self.token = secrets.token_urlsafe(32)
        self.lock = threading.Lock()
        self.processes = {}

    def jobs(self):
        jobs = []
        for path in sorted(self.folder.glob('*/job.json')):
            job = read_json(path)
            if not job:
                continue
            if job['status'] in ACTIVE and not (job['status'] == 'queued' and not job.get('pid') and job['id'] not in self.processes):
                process = self.processes.get(job['id'])
                alive = process is not None and process.poll() is None
                if not alive and process is None and job.get('pid'):
                    try:
                        os.kill(job['pid'], 0)
                        alive = True
                    except (ProcessLookupError, PermissionError):
                        pass
                if not alive:
                    job.update(status='interrupted', message='工作程序已離開，未自動續跑。')
                    write_json(path, job)
            job['cancel_requested'] = (path.parent / 'cancel.flag').exists() and job['status'] in ACTIVE
            jobs.append(job)
        return jobs

    def runs(self):
        jobs = {job['id']: job for job in self.jobs()}
        evaluations = {}
        for job in jobs.values():
            if job['kind'] in {'validation', 'train'} and job['status'] == 'completed':
                result = read_json(Path(job['folder']) / 'validation.json')
                if result:
                    evaluations[job['run_id']] = (job, result)
        runs = []
        for base in (self.root / 'logs/macos_v1', self.folder):
            for history in sorted(base.glob('*/' + '*_history.csv')):
                name = history.name.removesuffix('_history.csv')
                relative = history.parent.relative_to(base)
                weight_base = 'macos_v1' if base.name == 'macos_v1' else 'macos_ui'
                weight = self.root / 'weights' / weight_base / relative / f'{name}_best.h5'
                meta = read_json(Path(str(weight) + '.json'))
                if not meta:
                    continue
                run_id = history.parent.name if weight_base == 'macos_ui' else 'saved-' + hashlib.sha256(str(relative).encode()).hexdigest()[:16]
                cfg = meta['config']
                rows = history_rows(history)
                accuracy_key = 'val_main_accuracy' if cfg['model']['name'] == 'googlenet' and cfg['model'].get('aux_logits') else 'val_accuracy'
                best = max(rows, key=lambda row: row.get(accuracy_key, -1)) if rows else None
                job = jobs.get(run_id)
                timing = read_json(history.parent / 'timing.json', {})
                complete = (job.get('training_completed', False) or job['status'] == 'completed') if job else timing.get('exit_code') == 0
                run = {'id': run_id, 'name': history.parent.name, 'model': name,
                       'training': cfg['training'], 'input_size': cfg['model']['input_size'],
                       'history': rows, 'best': best, 'accuracy_key': accuracy_key,
                       'status': job['status'] if job else ('completed' if complete else 'incomplete'),
                       'elapsed_seconds': job.get('training_seconds', job.get('elapsed_seconds')) if job else timing.get('elapsed_seconds'),
                       'can_evaluate': complete and weight.is_file(),
                       'weight_path': str(weight), 'metadata': meta,
                       'execution': cfg.get('lab', {}).get('execution', {'backend': 'cpu', 'label': 'Mac CPU'}),
                       'mode': cfg.get('lab', {}).get('mode', 'legacy'),
                       'protocol_id': training_signature(cfg['lab']['source_sha256']) if cfg.get('lab', {}).get('source_sha256') else '',
                       'milestones': read_json(history.parent / 'milestones.json', {}),
                       'group_id': job.get('group_id') if job else None}
                if run_id in evaluations:
                    evaluation_job, result = evaluations[run_id]
                    run['evaluation_job'] = evaluation_job['id']
                    if complete: run['status'] = 'completed'
                    run['evaluation'] = {key: value for key, value in result.items() if key not in {'predictions', 'report_text', 'dataset_dir', 'weights'}}
                runs.append(run)
        known = {run['id'] for run in runs}
        for job in jobs.values():
            if job['kind'] != 'train' or job['id'] in known or job['status'] == 'superseded':
                continue
            config_path = Path(job['config'])
            if not config_path.is_file():
                continue
            cfg = yaml.safe_load(config_path.read_text())
            runs.append({'id': job['id'], 'name': job['id'], 'model': job['model'],
                'training': cfg['training'], 'input_size': cfg['model']['input_size'],
                'history': [], 'best': None, 'accuracy_key': 'val_accuracy',
                'status': job['status'], 'message': job.get('message', ''),
                'elapsed_seconds': job.get('elapsed_seconds'), 'can_evaluate': False,
                'execution': job.get('execution', {'backend': 'cpu', 'label': 'Mac CPU'}),
                'mode': job.get('mode', 'free'), 'protocol_id': cfg['lab']['protocol_id'],
                'group_id': job.get('group_id')})
        return runs

    def get_run(self, run_id):
        for run in self.runs():
            if run['id'] == run_id:
                return run
        raise ValueError('找不到這個實驗')

    def state(self):
        runs = self.runs()
        public_runs = [{key: value for key, value in run.items() if key not in {'weight_path', 'metadata'}} for run in runs]
        summary = read_json(self.root / 'data_animal_pictures/summary.json', {})
        return {'token': self.token, 'models': MODELS, 'runs': public_runs,
                'jobs': self.jobs(), 'dataset': summary.get('counts', {}),
                'device': 'Metal GPU（已驗證的後續訓練）' if read_json(self.root / 'logs/macos_metal_checks/runtime.json', {}).get('enabled') else 'CPU（目前本機環境）'}

    def stage_training(self, values, mode, group=None, index=1, total=1):
        """建立待執行設定與工作，不在這一步訓練。"""
        identifier = 'ui-' + time.strftime('%Y%m%d-%H%M%S') + '-' + uuid.uuid4().hex[:8]
        config = training_config(self.root, values, identifier, mode)
        folder = self.folder / identifier
        folder.mkdir(exist_ok=False)
        with (folder / 'config.yaml').open('x', encoding='utf-8') as stream:
            yaml.safe_dump(config, stream, allow_unicode=True, sort_keys=False)
        job = {'id': identifier, 'kind': 'train', 'status': 'queued', 'folder': str(folder),
               'project': str(self.root), 'created_at': time.time(), 'config': str(folder / 'config.yaml'),
               'run_id': identifier, 'epochs': values['epochs'], 'model': values['model'],
               'settings': values, 'mode': mode, 'group_id': group,
               'queue_index': index, 'group_total': total}
        write_json(folder / 'job.json', job)
        return job

    def spawn(self, job):
        """啟動單一背景程序；寫入PID供重新開啟介面時識別。"""
        folder = Path(job['folder'])
        runtime = read_json(self.root / 'logs/macos_metal_checks/runtime.json', {})
        gpu_python = self.root / '.venv-macos-metal/bin/python'
        use_gpu = (job['kind'] == 'train' and runtime.get('enabled')
                   and job.get('model') in runtime.get('validated_models', []) and gpu_python.is_file())
        runner = str(gpu_python) if use_gpu else sys.executable
        execution = dict(runtime.get('execution', {})) if use_gpu else {'backend': 'cpu', 'label': 'Mac CPU'}
        job['execution'] = execution
        if job['kind'] == 'train':
            config_path = Path(job['config'])
            cfg = yaml.safe_load(config_path.read_text())
            cfg.setdefault('lab', {})['execution'] = execution
            with config_path.open('w', encoding='utf-8') as stream:
                yaml.safe_dump(cfg, stream, allow_unicode=True, sort_keys=False)
        # 在子程序啟動前寫入執行環境；之後仍由 worker 獨佔進度紀錄。
        write_json(folder / 'job.json', job)
        try:
            with (folder / 'console.log').open('xb') as log:
                process = subprocess.Popen([runner, '-B', '-u', '-m', 'tools.web_worker',
                           '--job', str(folder / 'job.json')], cwd=self.root, stdout=log,
                           stderr=subprocess.STDOUT, start_new_session=True,
                           env={**os.environ, 'PYTHONUNBUFFERED': '1', 'TF_CPP_MIN_LOG_LEVEL': '2', 'TF_NUM_INTRAOP_THREADS': '4', 'TF_NUM_INTEROP_THREADS': '2'})
            self.processes[job['id']] = process
            # Worker owns progress JSON; parent tracks process in memory.
        except Exception as error:
            job.update(status='failed', message=str(error))
            write_json(folder / 'job.json', job)
            raise

    def advance_queue(self):
        """一次只跑一個工作，失敗結果保留並繼續下一個待執行設定。"""
        with self.lock:
            jobs = self.jobs()
            if any(process.poll() is None for process in self.processes.values()):
                return
            if any(job['status'] in {'running', 'cancel_requested'} or
                   (job['status'] == 'queued' and (job.get('pid') or job['id'] in self.processes)) for job in jobs):
                return
            waiting = sorted((job for job in jobs if job['status'] == 'queued'),
                             key=lambda job: job['created_at'])
            if waiting:
                try:
                    self.spawn(waiting[0])
                except Exception:
                    pass  # spawn已記錄失敗，下一次輪詢會繼續。

    def start(self, kind, values):
        """建立單次自由實驗、固定比較排程或指定最佳權重的驗證工作。"""
        with self.lock:
            if any(job['status'] in ACTIVE for job in self.jobs()):
                raise RuntimeError('目前已有工作或排程，請完成或取消後再開始。')
            if kind in {'train', 'comparison', 'grid'}:
                if kind == 'train':
                    job = self.stage_training(values, 'free')
                    self.spawn(job)
                    return {'id': job['id'], 'count': 1}
                if set(values) != {'model', 'epochs', 'batch_size', 'learning_rate', 'optimizer'}:
                    raise ValueError('固定比較設定欄位不完整')
                names = list(MODELS) if values['model'] == 'all' else [values['model']]
                rates = [.00001, .0001, .001] if values['learning_rate'] == 'all' else [values['learning_rate']]
                optimizers = ['adam', 'sgd', 'adamw', 'rmsprop'] if values['optimizer'] == 'all' else [values['optimizer']]
                group = kind + '-' + uuid.uuid4().hex[:12]
                plans = [dict(values, model=name, learning_rate=rate, optimizer=optimizer)
                         for optimizer, rate, name in itertools.product(optimizers, rates, names)]
                if kind == 'grid':
                    plans = [dict(plan, epochs=10) for plan in plans]
                mode = 'free' if kind == 'grid' else 'comparison'
                # 整批先驗證，避免欄位填錯卻留下半份排程。
                for plan in plans:
                    training_config(self.root, plan, 'preview', mode)
                reused = []
                existing = self.runs()
                pending = []
                for plan in plans:
                    candidate = training_config(self.root, plan, 'preview', mode)
                    match = next((run for run in existing if run['status'] == 'completed'
                        and run['can_evaluate'] and len(run['history']) == plan['epochs']
                        and (plan['epochs'] != 10 or run.get('evaluation'))
                        and run.get('protocol_id') == candidate['lab']['protocol_id']
                        and run['model'] == plan['model']
                        and all(run['training'].get(key) == plan[key] for key in
                                ['epochs', 'batch_size', 'learning_rate', 'optimizer'])), None)
                    if match:
                        reused.append(match['id'])
                    else:
                        pending.append(plan)
                plans = pending
                if not plans:
                    return {'id': None, 'count': 0, 'reused': reused, 'group_id': group}
                created = []
                try:
                    for index, plan in enumerate(plans, 1):
                        created.append(self.stage_training(plan, mode, group, index, len(plans)))
                except Exception:
                    for job in created:
                        job.update(status='cancelled', message='排程建立未完成，未開始訓練。')
                        write_json(Path(job['folder']) / 'job.json', job)
                    raise
                self.spawn(created[0])
                return {'id': created[0]['id'], 'count': len(created), 'group_id': group, 'reused': reused}
            if kind != 'validation' or set(values) != {'run_id'}:
                raise ValueError('不支援的工作或欄位')
            run = self.get_run(values['run_id'])
            if not run['can_evaluate']:
                raise ValueError('實驗尚未完成或沒有最佳權重，無法評估')
            weight = inside(run['weight_path'], self.root / 'weights')
            expected = (self.root / 'data_animal_pictures/val').resolve()
            if Path(run['metadata']['config']['dataset']['val_dir']).resolve() != expected:
                raise ValueError('只允許評估去重資料的驗證集')
            identifier = 'validation-' + uuid.uuid4().hex[:12]
            folder = self.folder / identifier
            folder.mkdir(exist_ok=False)
            job = {'id': identifier, 'kind': 'validation', 'status': 'queued', 'folder': str(folder),
                   'project': str(self.root), 'created_at': time.time(), 'run_id': run['id'],
                   'weights': str(weight)}
            write_json(folder / 'job.json', job)
            self.spawn(job)
            return {'id': identifier, 'count': 1}

    def cancel(self, identifier):
        """取消目前工作及同一比較群組的後續排程，保留已產生紀錄。"""
        with self.lock:
            jobs = self.jobs()
            target = next((job for job in jobs if job['id'] == identifier), None)
            if not target or target['status'] not in ACTIVE:
                raise ValueError('這個工作已經結束')
            for job in jobs:
                if job['id'] != identifier and not (target.get('group_id') and job.get('group_id') == target['group_id']):
                    continue
                if job['status'] not in ACTIVE:
                    continue
                folder = inside(job['folder'], self.folder)
                (folder / 'cancel.flag').touch(exist_ok=True)
                if job['status'] == 'queued' and not job.get('pid') and job['id'] not in self.processes:
                    job.update(status='cancelled', message='已取消待執行排程，沒有開始訓練。')
                    write_json(folder / 'job.json', job)
            return {'message': '已取消待執行排程；目前工作將於批次之間停止，紀錄保留。'}

    def image(self, run_id, index):
        run = self.get_run(run_id)
        if 'evaluation_job' not in run:
            raise ValueError('尚未計算逐圖預測')
        result = read_json(self.folder / run['evaluation_job'] / 'validation.json')
        predictions = result.get('predictions', [])
        if index < 0 or index >= len(predictions):
            raise ValueError('圖片編號不正確')
        path = inside(self.root / 'data_animal_pictures/val' / predictions[index]['filename'], self.root / 'data_animal_pictures/val')
        from PIL import Image, ImageOps
        with Image.open(path) as picture:
            picture = ImageOps.exif_transpose(picture).convert('RGB')
            picture.thumbnail((400, 300))
            output = io.BytesIO()
            picture.save(output, 'JPEG', quality=85)
        return output.getvalue()

    def errors(self, run_id, true_id, pred_id):
        run = self.get_run(run_id)
        if 'evaluation_job' not in run:
            return []
        result = read_json(self.folder / run['evaluation_job'] / 'validation.json')
        return [dict(index=index, **row) for index, row in enumerate(result['predictions']) if row['true_id'] == true_id and row['pred_id'] == pred_id][:12]


def create_server(lab, port=8765):
    """綁定 loopback，拒絕跨站寫入與任意路徑存取。"""
    class Handler(BaseHTTPRequestHandler):
        def send(self, content, kind='application/json; charset=utf-8', status=200):
            if not isinstance(content, bytes):
                content = json.dumps(content, ensure_ascii=False, allow_nan=False).encode()
            self.send_response(status)
            self.send_header('Content-Type', kind)
            self.send_header('Content-Length', str(len(content)))
            self.send_header('X-Content-Type-Options', 'nosniff')
            self.send_header('Cache-Control', 'no-store')
            self.send_header('Content-Security-Policy', "default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'; img-src 'self'; connect-src 'self'; frame-ancestors 'none'")
            self.end_headers()
            self.wfile.write(content)

        def allowed(self):
            port = self.server.server_address[1]
            return self.headers.get('Host') in {f'127.0.0.1:{port}', f'localhost:{port}'}

        def do_GET(self):
            if not self.allowed():
                return self.send({'error': '不接受此 Host'}, status=403)
            parsed = urlsplit(self.path)
            args = parse_qs(parsed.query)
            try:
                if parsed.path == '/api/state':
                    return self.send(lab.state())
                if parsed.path == '/api/errors':
                    return self.send(lab.errors(args['run'][0], int(args['true'][0]), int(args['pred'][0])))
                if parsed.path == '/api/image':
                    return self.send(lab.image(args['run'][0], int(args['index'][0])), 'image/jpeg')
                files = {'/': ('index.html', 'text/html; charset=utf-8'), '/app.js': ('app.js', 'text/javascript; charset=utf-8'), '/style.css': ('style.css', 'text/css; charset=utf-8')}
                # 只提供固定模型圖清單，避免任意路徑存取。
                model_images = {
                    '/model-images/' + name: lab.root / 'docs/images' / name
                    for name in ['lenet-reference.svg', 'alexnet-reference.svg', 'alexnet_lrn-reference.svg',
                                 'vgg16-reference.svg', 'vgg16_bn-reference.svg', 'googlenet-reference.svg',
                                 'inception-reference.svg', 'resnet50-reference.svg', 'mobilenet_v1-reference.svg',
                                 'efficientnet_b0-reference.svg']
                }
                if parsed.path in model_images:
                    return self.send(model_images[parsed.path].read_bytes(), 'image/svg+xml')
                if parsed.path not in files:
                    return self.send({'error': '找不到頁面'}, status=404)
                filename, kind = files[parsed.path]
                return self.send((lab.root / 'webs' / filename).read_bytes(), kind)
            except (ValueError, KeyError, IndexError, FileNotFoundError) as error:
                self.send({'error': str(error)}, status=400)
            except Exception:
                self.send({'error': '無法讀取結果，請查看本機紀錄。'}, status=500)

        def do_POST(self):
            port = self.server.server_address[1]
            if not self.allowed() or self.headers.get('Origin') not in {f'http://127.0.0.1:{port}', f'http://localhost:{port}'}:
                return self.send({'error': '僅接受本機介面的操作'}, status=403)
            if self.headers.get('X-Lab-Token') != lab.token or self.headers.get('Content-Type') != 'application/json':
                return self.send({'error': '缺少操作驗證'}, status=403)
            try:
                length = int(self.headers.get('Content-Length', 0))
                if not 0 < length <= 4096:
                    raise ValueError('請求大小不正確')
                data = json.loads(self.rfile.read(length))
                if not isinstance(data, dict):
                    raise ValueError('請傳入設定物件')
                routes = {'/api/train': lambda: lab.start('train', data), '/api/comparison': lambda: lab.start('comparison', data), '/api/grid': lambda: lab.start('grid', data), '/api/validate': lambda: lab.start('validation', data), '/api/cancel': lambda: lab.cancel(data['id'])}
                if self.path not in routes:
                    return self.send({'error': '不支援的操作'}, status=404)
                return self.send(routes[self.path]())
            except RuntimeError as error:
                self.send({'error': str(error)}, status=409)
            except (ValueError, KeyError, TypeError) as error:
                self.send({'error': str(error)}, status=400)
            except Exception:
                self.send({'error': '工作無法啟動，請查看本機紀錄。'}, status=500)

        def log_message(self, format, *args):
            pass

    server = ThreadingHTTPServer(('127.0.0.1', port), Handler)
    server.service_actions = lab.advance_queue
    return server


def main():
    """啟動本機介面；開啟頁面不會啟動訓練。"""
    parser = argparse.ArgumentParser(description='CNN Lab 本機介面')
    parser.add_argument('--port', type=int, default=8765)
    args = parser.parse_args()
    lab = Lab()
    lock_file = (lab.folder / 'server.lock').open('a')
    try:
        fcntl.flock(lock_file, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise SystemExit('CNN Lab 已在執行，請使用已開啟的介面。')
    server = create_server(lab, args.port)
    print(f'CNN Lab：http://127.0.0.1:{server.server_address[1]}', flush=True)
    print('開啟頁面不會訓練；按 Ctrl+C 關閉介面伺服器。', flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print('\n介面已關閉；進行中的背景工作仍可保留，重開後可查看或取消。')
    finally:
        server.server_close()
        lock_file.close()


if __name__ == '__main__':
    main()
