"""只匯出完成實驗的匿名數值；不載入 TensorFlow、權重或照片。"""
import argparse
import csv
from datetime import datetime, timezone
import json
import math
from pathlib import Path

MODELS = {'lenet':'LeNet', 'alexnet':'AlexNet', 'alexnet_lrn':'AlexNet LRN',
          'vgg16':'VGG16', 'vgg16_bn':'VGG16 BN', 'googlenet':'GoogLeNet',
          'inception':'Inception v2', 'resnet50':'ResNet50',
          'mobilenet_v1':'MobileNet v1', 'efficientnet_b0':'EfficientNet-B0'}
LABELS = {'cat','dog','fox','leopard','lion','wolf'}
QUARANTINED_MODELS = {'inception': '數值異常待查；暫不納入展示比較，原始紀錄保留。'}


def valid_evaluation(result):
    """檢查分類比例、張數與混淆矩陣是否一致，不把合理範圍當成全面正確保證。"""
    try:
        samples = result['samples']
        matrix = result['confusion_matrix']
        if type(samples) is not int or samples <= 0 or len(matrix) != 6 or any(len(row) != 6 for row in matrix):
            return False
        if any(type(v) is not int or v < 0 for row in matrix for v in row):
            return False
        if sum(map(sum, matrix)) != samples:
            return False
        if not math.isclose(sum(matrix[i][i] for i in range(6))/samples, result['accuracy'], abs_tol=1e-6):
            return False
        if not math.isfinite(result['loss']) or result['loss'] < 0:
            return False
        for key in ['accuracy', 'top3_accuracy']:
            if key in result and (not math.isfinite(result[key]) or not 0 <= result[key] <= 1):
                return False
        for name, row in result['report'].items():
            if name in LABELS | {'macro avg','weighted avg'} and isinstance(row, dict):
                if any(not math.isfinite(row[k]) or not 0 <= row[k] <= 1 for k in ['precision','recall','f1-score']):
                    return False
                if not math.isfinite(row['support']) or row['support'] < 0:
                    return False
        return True
    except (KeyError, TypeError, ValueError):
        return False


def export(source, output):
    """讀取完整十輪紀錄並輸出經欄位篩選的匿名展示數值。"""
    source, output = Path(source), Path(output)
    runs = []
    job_files = sorted((source / 'logs/macos_ui').glob('*/job.json'))
    jobs = [json.loads(path.read_text()) for path in job_files]
    completed_cases = {(job.get('model'), job.get('settings', {}).get('learning_rate'), job.get('settings', {}).get('optimizer'))
        for job in jobs if job.get('model') in MODELS and job.get('status') == 'completed' and job.get('epochs_completed') == 10}
    review_cases = []
    for optimizer in ['adam','sgd','adamw','rmsprop']:
        for rate in [.00001,.0001,.001]:
            matches = [job for job in jobs if job.get('model') == 'inception'
                and job.get('settings', {}).get('optimizer') == optimizer
                and job.get('settings', {}).get('learning_rate') == rate]
            completed = any(job.get('status') == 'completed' and job.get('epochs_completed') == 10 for job in matches)
            status = 'completed_under_review' if completed else 'failed' if any(job.get('status') == 'failed' for job in matches) else 'not_run'
            review_cases.append({'model':'inception', 'optimizer':optimizer, 'learning_rate':rate,
                'epochs':10, 'status':status, 'included_in_comparison':False,
                'reason':QUARANTINED_MODELS['inception']})
    for job_path in sorted((source / 'logs/macos_ui').glob('*/job.json')):
        job = json.loads(job_path.read_text())
        model = job.get('model')
        if job.get('status') != 'completed' or job.get('epochs_completed') != 10 or model not in MODELS:
            continue
        if model in QUARANTINED_MODELS:
            continue
        folder = job_path.parent
        result_path = folder / 'validation.json'
        history_path = folder / (model + '_history.csv')
        if not result_path.is_file() or not history_path.is_file():
            continue
        with history_path.open() as stream:
            history = [{k: float(v) for k, v in row.items() if k in {
                'epoch','accuracy','loss','top3_acc','val_accuracy','val_loss','val_top3_acc',
                'main_accuracy','main_top3_acc','main_loss','aux1_loss','aux2_loss',
                'val_main_accuracy','val_main_top3_acc','val_main_loss','val_aux1_loss','val_aux2_loss'
            }} for row in csv.DictReader(stream)]
        if len(history) != 10 or [row['epoch'] for row in history] != list(range(10)):
            continue
        if not all(math.isfinite(value) for row in history for value in row.values()):
            continue
        if any(('accuracy' in key or 'top3' in key) and not 0 <= value <= 1 for row in history for key, value in row.items()):
            continue
        for row in history:
            row['epoch'] = int(row['epoch']) + 1
        result = json.loads(result_path.read_text())
        if not valid_evaluation(result):
            continue
        if set(result.get('class_indices', {})) != LABELS:
            continue
        settings = job['settings']
        if settings.get('optimizer') not in {'adam','adamw','sgd','rmsprop'}:
            continue
        # 所有輸出採明確允許清單：不複製整個 job、config、metadata 或 predictions。
        evaluation = {key: result[key] for key in ['samples','accuracy','loss','top3_accuracy',
                      'class_indices','confusion_matrix','report'] if key in result}
        # 報告只保留預定六類與三種彙整；拒絕任意字串／檔名。
        evaluation['report'] = {name: {key: float(row[key]) for key in
            ['precision','recall','f1-score','support'] if key in row}
            for name, row in result['report'].items()
            if name in LABELS | {'macro avg','weighted avg'} and isinstance(row, dict)}
        assert math.isfinite(evaluation['accuracy'])
        assert set(evaluation['class_indices'].values()) == set(range(6))
        assert len(evaluation['confusion_matrix']) == 6
        assert all(len(row) == 6 for row in evaluation['confusion_matrix'])
        backend = job.get('execution', {}).get('backend', 'cpu')
        backend = backend if backend in {'cpu','metal','cuda'} else 'cpu'
        execution = {'backend': backend, 'label': {'cpu':'Mac CPU','metal':'Apple Metal GPU','cuda':'NVIDIA CUDA GPU'}[backend]}
        accuracy_key = 'val_main_accuracy' if model == 'googlenet' and 'val_main_accuracy' in history[0] else 'val_accuracy'
        best = max(history, key=lambda row: row[accuracy_key])
        runs.append({'id': 'showcase-' + str(len(runs)+1).zfill(3), 'name':'展示實驗',
            'source':'showcase', 'model':model, 'training':{
                'epochs':10, 'batch_size':int(settings['batch_size']),
                'learning_rate':float(settings['learning_rate']), 'optimizer':settings['optimizer'], 'seed':42},
            'input_size':[160,160], 'history':history, 'best':best, 'accuracy_key':accuracy_key,
            'status':'completed', 'can_evaluate':False, 'evaluation':evaluation,
            'execution':execution, 'protocol_id':'showcase-baseline', 'milestones':{},
            'elapsed_seconds':job.get('training_seconds'),
            'timing_note':'歷史牆鐘時間，部分可能含休眠或暫停；不作速度排名'})
    summary = json.loads((source / 'data_afhq_deduplicated/summary.json').read_text())
    counts = {split:{label:int(summary['counts'][split][label]) for label in sorted(LABELS)}
              for split in ['train','val','test']}
    data = {'schema_version':1, 'mode':'showcase', 'models':MODELS, 'runs':runs, 'jobs':[],
            'dataset':counts, 'device':'展示模式，不執行訓練',
            'snapshot':{'exported_at':datetime.now(timezone.utc).isoformat(), 'completed_runs':len(runs),
                'training_completed_runs':len(completed_cases),
                'quarantined_completed_runs':sum(case[0] in QUARANTINED_MODELS for case in completed_cases),
                'review_cases':review_cases, 'quarantined_models':QUARANTINED_MODELS,
                'planned_runs':120, 'final_results':False, 'weights_included':False,
                'note':'驗證集探索快照；Inception 暫停採信且排除比較；抽查通過不代表全部結果已驗證。非最終測試成績，不含照片或權重。'}}
    text = json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False)
    assert not any(marker in text for marker in ['/Users/', '/Volumes/', '.local', 'weight_path', 'dataset_dir', 'predictions', 'token'])
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix('.tmp')
    temporary.write_text(text, encoding='utf-8')
    temporary.replace(output)
    return len(runs)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True, help='本機訓練專案根目錄（唯讀）')
    parser.add_argument('--output', type=Path, default=Path('results/showcase.json'))
    args = parser.parse_args()
    print('已匯出', export(args.source, args.output), '組完成結果')
