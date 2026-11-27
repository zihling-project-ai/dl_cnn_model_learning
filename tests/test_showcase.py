import csv
import json
from pathlib import Path
import tempfile
import unittest
from tools.export_showcase import export


class ShowcaseExportTests(unittest.TestCase):
    def fixture(self, root, status='completed', epochs=10):
        folder=root/'logs/macos_ui/private-run-id';folder.mkdir(parents=True)
        labels=['cat','dog','fox','leopard','lion','wolf']
        (folder/'job.json').write_text(json.dumps({'status':status,'epochs_completed':epochs,'model':'lenet',
            'settings':{'batch_size':32,'learning_rate':0.001,'optimizer':'adam'},
            'folder':'/Users/private/person','token':'private-secret','training_seconds':12}))
        with (folder/'lenet_history.csv').open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=['epoch','accuracy','loss','val_accuracy','val_loss']);writer.writeheader()
            for i in range(epochs):writer.writerow(dict(epoch=i,accuracy=.5,loss=1,val_accuracy=.5,val_loss=1))
        report={name:{'precision':.5,'recall':.5,'f1-score':.5,'support':1} for name in labels}
        (folder/'validation.json').write_text(json.dumps({'accuracy':.5,'loss':1,'samples':6,
            'class_indices':dict(zip(labels,range(6))),'report':report,
            'confusion_matrix':[[int(j==(i if i<3 else (i+1)%6)) for j in range(6)] for i in range(6)],
            'weights':'/Users/private/model.h5','predictions':[{'filename':'private.jpg'}]}))
        data=root/'data_afhq_deduplicated';data.mkdir()
        (data/'summary.json').write_text(json.dumps({'counts':{s:{n:1 for n in labels} for s in ['train','val','test']}}))

    def test_exports_numbers_without_paths_secrets_or_predictions(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);self.fixture(root);out=root/'public.json'
            self.assertEqual(export(root,out),1)
            text=out.read_text();data=json.loads(text)
            for secret in ['private-run-id','private-secret','/Users/','private.jpg','model.h5']:
                self.assertNotIn(secret,text)
            self.assertEqual(len(data['runs'][0]['history']),10)
            self.assertFalse(data['runs'][0]['can_evaluate'])

    def test_incomplete_run_is_not_exported(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);self.fixture(root,epochs=5)
            self.assertEqual(export(root,root/'public.json'),0)

    def test_failed_run_is_not_exported(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);self.fixture(root,status='failed')
            self.assertEqual(export(root,root/'public.json'),0)

    def test_quarantined_inception_stays_out_of_comparison(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);self.fixture(root)
            folder=root/'logs/macos_ui/private-run-id'
            job=json.loads((folder/'job.json').read_text());job['model']='inception'
            (folder/'job.json').write_text(json.dumps(job))
            (folder/'lenet_history.csv').rename(folder/'inception_history.csv')
            out=root/'public.json'
            self.assertEqual(export(root,out),0)
            data=json.loads(out.read_text())
            self.assertEqual(data['snapshot']['training_completed_runs'],1)
            self.assertEqual(data['snapshot']['quarantined_completed_runs'],1)
            self.assertEqual(len(data['snapshot']['review_cases']),12)
            self.assertFalse(any(row['included_in_comparison'] for row in data['snapshot']['review_cases']))

    def test_out_of_range_history_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);self.fixture(root)
            history=root/'logs/macos_ui/private-run-id/lenet_history.csv'
            history.write_text(history.read_text().replace('0.5,1,0.5,1','-0.1,1,0.5,1',1))
            self.assertEqual(export(root,root/'public.json'),0)

    def test_inconsistent_confusion_matrix_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);self.fixture(root)
            report=root/'logs/macos_ui/private-run-id/validation.json'
            result=json.loads(report.read_text());result['accuracy']=.9
            report.write_text(json.dumps(result))
            self.assertEqual(export(root,root/'public.json'),0)
