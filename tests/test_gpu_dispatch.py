"""以假程序檢查 GPU 派工，不啟動實際訓練。"""
import json
from pathlib import Path
import sys,tempfile,unittest
from unittest.mock import Mock,patch
import yaml
from tools.web_app import Lab


class GPUDispatchTests(unittest.TestCase):
    def test_only_validated_models_use_fixed_gpu_environment(self):
        for validated in [True,False]:
            with self.subTest(validated=validated),tempfile.TemporaryDirectory() as temp:
                root=Path(temp)
                original=Path(__file__).resolve().parents[1]
                (root/'config_lab.yaml').write_text((original/'config_lab.yaml').read_text())
                for split in ['train','val']:(root/'data_animal_pictures'/split).mkdir(parents=True)
                gpu=root/'.venv-macos-metal/bin/python';gpu.parent.mkdir(parents=True);gpu.touch()
                folder=root/'logs/macos_metal_checks';folder.mkdir(parents=True)
                (folder/'runtime.json').write_text(json.dumps({'enabled':True,'python':'/untrusted/ignored','validated_models':['lenet'] if validated else [],'execution':{'backend':'metal','label':'Apple Metal GPU'}}))
                lab=Lab(root);process=Mock(pid=99999999);process.poll.return_value=None
                values={'model':'lenet','epochs':10,'batch_size':32,'learning_rate':.0001,'optimizer':'adam'}
                with patch('tools.web_app.subprocess.Popen',return_value=process) as spawn:
                    job=lab.start('train',values)
                self.assertEqual(spawn.call_args.args[0][0],str(gpu.resolve()) if validated else sys.executable)
                cfg=yaml.safe_load((lab.folder/job['id']/'config.yaml').read_text())
                self.assertEqual(cfg['lab']['execution']['backend'],'metal' if validated else 'cpu')
