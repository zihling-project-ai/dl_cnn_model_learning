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


def export(source, output):
    """讀取完整十輪紀錄並輸出經欄位篩選的匿名展示數值。"""
    source, output = Path(source), Path(output)
    runs = []
    for job_path in sorted((source / 'logs/macos_ui').glob('*/job.json')):
        job = json.loads(job_path.read_text())
        model = job.get('model')
        if job.get('status') != 'completed' or job.get('epochs_completed') != 10 or model not in MODELS:
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
        for row in history:
            row['epoch'] = int(row['epoch']) + 1
        result = json.loads(result_path.read_text())
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
                'planned_runs':120, 'final_results':False, 'weights_included':False,
                'note':'進行中的驗證實驗快照，非最終測試成績；不含照片、權重或個別預測。'}}
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
