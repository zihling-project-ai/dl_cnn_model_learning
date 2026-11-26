# cuda_check.py：手動檢查 TensorFlow 可見裝置與矩陣運算。

import tensorflow as tf


def main():
    """顯示可見 GPU，並執行一個矩陣乘法，不進行模型訓練。"""
    # 1. 顯示裝置並開啟運算位置的顯示。
    print(tf.config.list_physical_devices("GPU"))
    tf.debugging.set_log_device_placement(True)

    # 2. 建立測試矩陣並相乘。
    a = tf.random.normal([1000, 1000])
    b = tf.random.normal([1000, 1000])
    c = tf.matmul(a, b)
    print(c)


if __name__ == "__main__":
    main()
