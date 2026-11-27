# NVIDIA GPU 版本；沿用專案 TensorFlow 2.15，避免直接升級到 Keras 3。
FROM tensorflow/tensorflow:2.15.0-gpu
WORKDIR /app
COPY docker/requirements.txt /tmp/cnn-requirements.txt
RUN python -m pip install --no-cache-dir -r /tmp/cnn-requirements.txt
# 明確列出程式，不複製本機資料集、權重、環境或歷史紀錄。
COPY main_train.py main_test.py config_lab.yaml config.yaml ./
COPY models/ ./models/
COPY tools/ ./tools/
COPY webs/ ./webs/
COPY docker/ ./docker/
COPY docs/images/ ./docs/images/
COPY results/ ./results/
# 只調整映像內的介面連線與 GPU 派工；本機原始檔不變。
RUN python docker/prepare_app.py
ENV PYTHONUNBUFFERED=1 TF_CPP_MIN_LOG_LEVEL=2 TF_FORCE_GPU_ALLOW_GROWTH=true
EXPOSE 8765
CMD ["python", "docker/start.py"]
