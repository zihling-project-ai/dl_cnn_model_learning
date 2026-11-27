"""由共用 Lab HTML 產生靜態展示首頁；不需要 TensorFlow。"""
from pathlib import Path


def build(root):
    """由共用前端產生不需後端的展示首頁。"""
    root = Path(root)
    text = (root / 'webs/index.html').read_text(encoding='utf-8')
    text = text.replace('href="/style.css"', 'href="webs/style.css"')
    text = text.replace('src="/app.js"', 'src="webs/app.js"')
    text = text.replace('id="cnn-lab-preview-v2"', 'id="cnn-lab-preview-v2" data-mode="showcase"', 1)
    text = text.replace('/model-images/', 'docs/images/')
    (root / 'index.html').write_text(text, encoding='utf-8')


if __name__ == '__main__':
    build(Path(__file__).resolve().parents[1])
