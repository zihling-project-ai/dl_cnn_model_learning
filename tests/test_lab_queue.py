"""驗證排程與重用規則；使用假程序，不跑大型訓練。"""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch
import yaml
from tools.web_app import Lab, training_config
from tools.web_worker import train, write_json


class QueueTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        original = Path(__file__).resolve().parents[1]
        (self.root/'config_lab.yaml').write_text((original/'config_lab.yaml').read_text())
        for split in ['train','val']:
            (self.root/'data_animal_pictures'/split).mkdir(parents=True)
        self.lab=Lab(self.root)
        self.values={'model':'lenet','epochs':10,'batch_size':32,'learning_rate':.0001,'optimizer':'adam'}
        self.process=Mock(pid=99999999)
        self.process.poll.return_value=None

    def test_no_weights_needed_and_lr_rules(self):
        self.assertFalse((self.root/'weights').exists())
        with patch('tools.web_app.subprocess.Popen',return_value=self.process):
            result=self.lab.start('train',dict(self.values,epochs=1,learning_rate=.00001))
        cfg=yaml.safe_load((self.lab.folder/result['id']/'config.yaml').read_text())
        self.assertFalse(cfg['training']['early_stopping'])
        self.assertFalse(cfg['lab']['auto_evaluate_best'])
        self.assertEqual(len(self.lab.jobs()),1)
        for bad in [dict(self.values,batch_size=16),dict(self.values,batch_size=64),dict(self.values,learning_rate=.01)]:
            with self.assertRaises(ValueError):training_config(self.root,bad,'x','free')

    def test_120_unique_jobs_only_one_process_then_continue_after_failure(self):
        values=dict(self.values,model='all',learning_rate='all',optimizer='all')
        with patch('tools.web_app.subprocess.Popen',return_value=self.process) as spawn:
            result=self.lab.start('grid',values)
            jobs=self.lab.jobs()
            self.assertEqual(result['count'],120)
            self.assertEqual(spawn.call_count,1)
            self.lab.advance_queue()
            self.assertEqual(spawn.call_count,1)
            configs=[yaml.safe_load(Path(j['config']).read_text()) for j in jobs]
            keys={(c['model']['name'],c['training']['epochs'],c['training']['learning_rate'],c['training']['optimizer']) for c in configs}
            self.assertEqual(len(keys),120)
            first=next(j for j in jobs if j['id']==result['id'])
            first.update(status='failed',message='保留失敗例子')
            write_json(Path(first['folder'])/'job.json',first)
            self.process.poll.return_value=1
            self.lab.advance_queue()
            self.assertEqual(spawn.call_count,2)
            self.lab.cancel(next(j['id'] for j in self.lab.jobs() if j['status']=='queued'))
            self.assertEqual(sum(j['status']=='failed' for j in self.lab.jobs()),1)
            self.assertFalse(any(j['status']=='queued' and j['id'] not in self.lab.processes for j in self.lab.jobs()))
            self.assertTrue(any(r.get('message')=='保留失敗例子' for r in self.lab.runs()))

    def test_matching_ten_epoch_free_run_reused_without_spawning(self):
        cfg=training_config(self.root,self.values,'prior','free')
        candidate={'id':'prior','status':'completed','model':'lenet','can_evaluate':True,
                   'evaluation':{'accuracy':.5},'protocol_id':cfg['lab']['protocol_id'],
                   'history':[{}]*10,'training':cfg['training']}
        with patch.object(self.lab,'runs',return_value=[candidate]),patch('tools.web_app.subprocess.Popen') as spawn:
            result=self.lab.start('comparison',self.values)
            self.assertEqual(result['count'],0)
            self.assertEqual(result['reused'],['prior'])
            spawn.assert_not_called()
        candidate['training']['seed']=99
        # Config version mismatch must never be called equivalent.
        candidate['protocol_id']='different'
        with patch.object(self.lab,'runs',return_value=[candidate]),patch('tools.web_app.subprocess.Popen',return_value=self.process):
            self.assertEqual(self.lab.start('comparison',self.values)['count'],1)

    def test_best_eval_only_after_successful_ten_epochs(self):
        folder=self.root/'work';folder.mkdir()
        cfg=training_config(self.root,self.values,'work','free')
        config=folder/'config.yaml';config.write_text(yaml.safe_dump(cfg))
        job={'config':str(config),'project':str(self.root),'folder':str(folder),'run_id':'work'}
        updates=[]
        with patch('main_train.main') as training,patch('tools.web_worker.validate_completed_history'),patch('tools.web_worker.evaluate_validation') as evaluation:
            train(job,lambda **data:updates.append(data))
            training.assert_called_once()
            evaluation.assert_called_once()
            self.assertTrue(any(row.get('training_completed') for row in updates))
        with patch('main_train.main',side_effect=RuntimeError('failed')),patch('tools.web_worker.evaluate_validation') as evaluation:
            with self.assertRaises(RuntimeError):train(job,lambda **data:None)
            evaluation.assert_not_called()

    def test_source_change_stops_before_training(self):
        folder=self.root/'work';folder.mkdir()
        cfg=training_config(self.root,self.values,'work','free')
        config=folder/'config.yaml';config.write_text(yaml.safe_dump(cfg))
        (self.root/'config_lab.yaml').write_text('changed')
        job={'config':str(config),'project':str(self.root),'folder':str(folder),'run_id':'work'}
        with patch('main_train.main') as training,self.assertRaisesRegex(ValueError,'基準'):
            train(job,lambda **data:None)
        training.assert_not_called()


class MilestoneTests(unittest.TestCase):
    def test_snapshots_preserve_best_so_far_without_future_overwrite(self):
        from tools.web_worker import save_milestone
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);weight=root/'weights';logs=root/'logs'
            weight.mkdir();logs.mkdir()
            cfg={'model':{'name':'toy'},'save':{'checkpoint_dir':str(weight),'log_dir':str(logs)}}
            (weight/'toy_best.h5').write_bytes(b'best-at-1')
            (weight/'toy_best.h5.json').write_text('{}')
            (logs/'toy_history.csv').write_text('epoch,loss\n0,1.0\n')
            save_milestone(cfg,1,1,3)
            (weight/'toy_best.h5').write_bytes(b'best-at-4')
            (logs/'toy_history.csv').write_text('epoch,loss\n0,1.0\n1,.9\n2,.8\n3,.7\n4,.8\n')
            save_milestone(cfg,5,4,15)
            self.assertEqual((weight/'milestones/epoch_1/toy_best.h5').read_bytes(),b'best-at-1')
            self.assertEqual((weight/'milestones/epoch_5/toy_best.h5').read_bytes(),b'best-at-4')
            self.assertEqual(json.loads((logs/'milestones.json').read_text())['5']['best_epoch'],4)
            with self.assertRaises(FileExistsError):save_milestone(cfg,1,1,16)
