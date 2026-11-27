"""只檢查容器適配與掛載規則，不宣稱 Docker／NVIDIA 實機通過。"""
from pathlib import Path
import unittest
import yaml
from docker.prepare_app import prepare
ROOT=Path(__file__).resolve().parents[1]

class DockerPreparationTests(unittest.TestCase):
    def test_patch_matches_current_published_app(self):
        original=(ROOT/'tools/web_app.py').read_text()
        adapted=prepare(original)
        compile(adapted,'container-web-app','exec')
        self.assertIn("ThreadingHTTPServer(('0.0.0.0', port)",adapted)
        self.assertIn("runtime.get('dispatch_models', [])",adapted)
        self.assertNotIn("self.root / '.venv-macos-metal/bin/python'",adapted)
        self.assertIn('http://127.0.0.1:8766',adapted)
        self.assertEqual(original,(ROOT/'tools/web_app.py').read_text())

    def test_changed_source_fails_instead_of_silent_patch(self):
        with self.assertRaises(ValueError):prepare('unrecognized source')

    def test_compose_keeps_data_readonly_outputs_persistent_and_port_local(self):
        cfg=yaml.safe_load((ROOT/'compose.yaml').read_text())['services']['cnn-lab']
        self.assertEqual(cfg['ports'],['127.0.0.1:8766:8765'])
        self.assertEqual(cfg['platform'],'linux/amd64')
        self.assertEqual(cfg['restart'],'no')
        data=cfg['volumes'][0]
        self.assertEqual(data['source'],'./data_animal_pictures')
        self.assertTrue(data['read_only'])
        self.assertFalse(data['bind']['create_host_path'])
        self.assertIn('./docker-results/weights:/app/weights',cfg['volumes'])
        self.assertEqual(cfg['deploy']['resources']['reservations']['devices'][0]['capabilities'],['gpu'])

    def test_build_inputs_include_published_paths_and_no_photos(self):
        source=(ROOT/'Dockerfile').read_text()
        self.assertIn('COPY tools/ ./tools/',source)
        self.assertIn('COPY webs/ ./webs/',source)
        self.assertIn('COPY results/ ./results/',source)
        self.assertNotIn('COPY . ',source)
        for name in ['start.py','prepare_app.py']:
            compile((ROOT/'docker'/name).read_text(),name,'exec')
