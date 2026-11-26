"""提早結束與非有限紀錄不能冒充完成實驗。"""
from pathlib import Path
import tempfile,unittest
from tools.web_worker import validate_completed_history

class CompletionGuardTests(unittest.TestCase):
    def test_rejects_short_nonfinite_and_wrong_epoch_history(self):
        with tempfile.TemporaryDirectory() as temp:
            cfg={'save':{'log_dir':temp},'model':{'name':'toy'},'training':{'epochs':2}}
            p=Path(temp)/'toy_history.csv'
            for content in ['epoch,loss\n0,1\n','epoch,loss\n0,1\n1,nan\n','epoch,loss\n0,1\n2,.5\n']:
                p.write_text(content)
                with self.assertRaises(ValueError):validate_completed_history(cfg)
            p.write_text('epoch,loss\n0,1\n1,.5\n')
            validate_completed_history(cfg)
