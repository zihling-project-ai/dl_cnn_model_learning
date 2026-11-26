"""使用暫存資料測試介面安全與評估流程，不進行實際訓練。"""
import copy
import csv
import hashlib
import http.client
import json
from pathlib import Path
import sys
import tempfile
import threading
import unittest
from unittest.mock import patch, Mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import yaml
from tools.web_app import Lab, create_server, inside, history_rows
from tools.web_worker import check_cancel, Cancelled, train, write_json


class WebLabTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        original = Path(__file__).resolve().parents[1]
        self.template = (original / 'config_lab.yaml').read_text()
        (self.root / 'config_lab.yaml').write_text(self.template)
        for split in ['train', 'val', 'test']:
            (self.root / 'data_animal_pictures' / split).mkdir(parents=True)
        self.lab = Lab(self.root)
        self.values = {'model': 'efficientnet_b0', 'epochs': 1, 'batch_size': 32,
                       'learning_rate': .001, 'optimizer': 'adam'}
        self.process = Mock()
        self.process.pid = 99999999
        self.process.poll.return_value = None

    def test_new_training_config_and_exclusive_job(self):
        before = (self.root / 'config_lab.yaml').read_bytes()
        with patch('tools.web_app.subprocess.Popen', return_value=self.process) as start:
            result = self.lab.start('train', self.values)
            folder = self.lab.folder / result['id']
            config = yaml.safe_load((folder / 'config.yaml').read_text())
            self.assertEqual(config['training']['epochs'], 1)
            self.assertEqual(Path(config['dataset']['val_dir']), (self.root / 'data_animal_pictures/val').resolve())
            self.assertIn(result['id'], config['save']['checkpoint_dir'])
            self.assertEqual(start.call_count, 1)
            self.assertTrue(start.call_args.kwargs["start_new_session"])
            with self.assertRaises(RuntimeError):
                self.lab.start('train', self.values)
            self.lab.cancel(result['id'])
            self.assertTrue((folder / 'cancel.flag').is_file())
            self.assertTrue(self.lab.jobs()[0]['cancel_requested'])
        self.assertEqual((self.root / 'config_lab.yaml').read_bytes(), before)

    def test_bad_parameters_never_spawn(self):
        bad = [dict(self.values, epochs=True), dict(self.values, batch_size=999),
               dict(self.values, learning_rate=float('nan')), dict(self.values, model='../other'),
               dict(self.values, optimizer='shell'), dict(self.values, extra='test_dir')]
        with patch('tools.web_app.subprocess.Popen') as start:
            for values in bad:
                with self.subTest(values=values), self.assertRaises(ValueError):
                    self.lab.start('train', values)
            start.assert_not_called()
        self.assertEqual(list(self.lab.folder.glob('*/job.json')), [])

    def test_failed_worker_is_interrupted_not_completed(self):
        with patch('tools.web_app.subprocess.Popen', return_value=self.process):
            job = self.lab.start('train', self.values)
        self.process.poll.return_value = 1
        self.assertEqual(self.lab.jobs()[0]['status'], 'interrupted')

    def saved_run(self, val_split='val'):
        folder = self.root / 'logs/macos_v1/example'
        folder.mkdir(parents=True)
        (folder / 'efficientnet_b0_history.csv').write_text('epoch,accuracy,loss,val_accuracy,val_loss\n0,0.5,1,0.6,0.9\n')
        write_json(folder / 'timing.json', {'exit_code': 0, 'elapsed_seconds': 4})
        weight = self.root / 'weights/macos_v1/example/efficientnet_b0_best.h5'
        weight.parent.mkdir(parents=True)
        weight.write_bytes(b'not a real model; never loaded')
        config = yaml.safe_load(self.template)
        config['dataset']['val_dir'] = str(self.root / 'data_animal_pictures' / val_split)
        write_json(Path(str(weight) + '.json'), {'config': config})
        return self.lab.runs()[0]

    def test_validation_rejects_test_split(self):
        run = self.saved_run('test')
        with patch('tools.web_app.subprocess.Popen') as start, self.assertRaisesRegex(ValueError, '驗證集'):
            self.lab.start('validation', {'run_id': run['id']})
        start.assert_not_called()

    def test_validation_uses_fixed_weight_and_saved_run(self):
        run = self.saved_run()
        with patch('tools.web_app.subprocess.Popen', return_value=self.process):
            job = self.lab.start('validation', {'run_id': run['id']})
        data = json.loads((self.lab.folder / job['id'] / 'job.json').read_text())
        self.assertEqual(data['weights'], run['weight_path'])
        self.assertEqual(data['kind'], 'validation')
        self.assertEqual(run['best']['epoch'], 1)

    def test_path_and_image_traversal_rejected(self):
        with self.assertRaises(ValueError):
            inside(self.root / '../outside.png', self.root)
        (self.root / 'link').symlink_to('/private/tmp')
        with self.assertRaises(ValueError):
            inside(self.root / 'link/outside.png', self.root)

    def test_progress_callback_cancels_without_training(self):
        folder = self.root / 'job'
        folder.mkdir()
        job = {'folder': str(folder), 'config': str(self.root / 'config_lab.yaml')}
        messages = []
        def fake_main(config_path, extra_callbacks):
            callback = extra_callbacks[0]
            callback.set_params({'steps': 2})
            callback.on_epoch_begin(0)
            callback.on_train_batch_end(0)
            (folder / 'cancel.flag').touch()
            callback.on_train_batch_end(1)
        with patch('main_train.main', side_effect=fake_main), self.assertRaises(Cancelled):
            train(job, lambda **data: messages.append(data))
        self.assertTrue(any(data.get('batch') == 1 for data in messages))

    def test_http_origin_token_and_static_routes(self):
        server = create_server(self.lab, 0)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        self.addCleanup(server.server_close)
        self.addCleanup(server.shutdown)
        port = server.server_address[1]
        connection = http.client.HTTPConnection('127.0.0.1', port)
        self.addCleanup(connection.close)
        with patch.object(self.lab, 'start', return_value={'id':'mocked-no-worker'}) as start:
            connection.request('POST', '/api/train', json.dumps(self.values), headers={'Content-Type':'application/json','Origin':'http://evil.example','X-Lab-Token':self.lab.token})
            response = connection.getresponse();response.read();self.assertEqual(response.status, 403)
            start.assert_not_called()
            connection.request('POST', '/api/train', json.dumps(self.values), headers={'Content-Type':'application/json','Origin':f'http://127.0.0.1:{port}','X-Lab-Token':self.lab.token})
            response = connection.getresponse();response.read();self.assertEqual(response.status, 200)
            self.assertEqual(start.call_count, 1)
        connection.request('GET', '/../../config.yaml')
        response = connection.getresponse();response.read();self.assertEqual(response.status, 404)
        connection.request('GET', '/api/state', headers={'Host':f'evil.example:{port}'})
        response = connection.getresponse();response.read();self.assertEqual(response.status, 403)

    def test_prediction_rows_and_metrics_are_aligned(self):
        import numpy as np
        from main_test import evaluate_batches
        class Dataset:
            shuffle = False
            class_indices = {'cat': 0, 'dog': 1}
            filenames = ['cat/a.png', 'dog/b.png']
            def __len__(self): return 1
            def __getitem__(self, index): return np.zeros((2,32,32,3)), np.array([0,1])
        model = lambda images, training: np.array([[.9,.1],[.7,.3]])
        progress=[]
        result=evaluate_batches(model, Dataset(), include_predictions=True, on_batch_end=lambda *args:progress.append(args))
        self.assertEqual(result['confusion_matrix'], [[1,0],[1,0]])
        self.assertEqual(result['predictions'][1]['filename'], 'dog/b.png')
        self.assertEqual(result['predictions'][1]['true_id'], 1)
        self.assertEqual(result['predictions'][1]['pred_id'], 0)
        self.assertEqual(progress, [(1,1)])
        self.assertEqual(result['accuracy'], .5)


    def test_validation_worker_only_reads_validation_and_saves_predictions(self):
        import numpy as np
        from PIL import Image
        import tensorflow as tf
        from tools.web_worker import evaluate_validation
        from tools.utils import make_run_metadata
        from tools.dataloader import load_test_dataset
        folder = self.root / 'validation-job'
        folder.mkdir()
        for name in ['cat', 'dog']:
            directory = self.root / 'data_animal_pictures/val' / name
            directory.mkdir()
            Image.fromarray(np.full((32,32,3),100,dtype=np.uint8)).save(directory/'a.png')
        config = {'model':{'name':'lenet','num_classes':2,'input_size':[32,32],'aux_logits':False},
                  'training':{'batch_size':2,'epochs':1,'learning_rate':.001,'optimizer':'adam','seed':42},
                  'augmentation':{'use_aug':False},
                  'dataset':{split+'_dir':str(self.root/'data_animal_pictures'/split) for split in ['train','val','test']},
                  'save':{'checkpoint_dir':str(folder),'log_dir':str(folder)}}
        model = tf.keras.Sequential([tf.keras.layers.Input((32,32,3)),tf.keras.layers.GlobalAveragePooling2D(),tf.keras.layers.Dense(2,activation='softmax',kernel_initializer='zeros',bias_initializer='zeros')])
        weight=folder/'toy.h5';model.save_weights(str(weight))
        write_json(Path(str(weight)+'.json'),make_run_metadata(config,{'cat':0,'dog':1}))
        job={'project':str(self.root),'folder':str(folder),'weights':str(weight),'run_id':'toy-only'}
        updates=[]
        with patch('models.get_model',return_value=model), patch('tools.dataloader.load_test_dataset',wraps=load_test_dataset) as loader, patch.object(tf.keras.Model,'fit',side_effect=AssertionError('no training')):
            evaluate_validation(job,lambda **values:updates.append(values))
        self.assertEqual(Path(loader.call_args.args[0]),(self.root/'data_animal_pictures/val').resolve())
        result=json.loads((folder/'validation.json').read_text())
        self.assertEqual(result['split'],'validation')
        self.assertEqual(result['samples'],2)
        self.assertEqual(len(result['predictions']),2)
        self.assertEqual(result['confusion_matrix'],[[1,0],[1,0]])
        self.assertTrue(result['settings_verified'])
        self.assertTrue(updates[-1]['result_file'].endswith('validation.json'))
        tf.keras.backend.clear_session()

    def test_error_image_cannot_read_test_folder(self):
        folder=self.lab.folder/'fake-validation';folder.mkdir()
        write_json(folder/'validation.json',{'predictions':[{'filename':'../test/cat/a.png','true_id':0,'pred_id':1}]})
        with patch.object(self.lab,'get_run',return_value={'evaluation_job':'fake-validation'}):
            with self.assertRaisesRegex(ValueError,'路徑'):
                self.lab.image('anything',0)


if __name__ == '__main__':
    unittest.main()
