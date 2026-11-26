'''
GoogLeNet Inception v2（BN-Inception）架構實作
🔹 1. 5×5 分支換成 2 個 3×3；通道數相同且只比較卷積權重時，25C² → 18C²，少約 28%。
🔹 2. 卷積後加入 Batch Normalization，再接 ReLU；這版不建立 Aux 輸出。
🔹 3. 使用 1×1 conv 降通道，搭配多分支，控制計算量。
🔹 4. 對齊 TF-Slim Inception v2 的十個模組與通道配置；速度、穩定性與準確率仍需實測。

參考：https://github.com/tensorflow/models/blob/master/research/slim/nets/inception_v2.py
使用 Keras 重新實作；輸入尺寸、六類輸出與訓練設定依本專案調整。
'''

import tensorflow as tf
from tensorflow.keras import layers, Model

# ============================
#   基本 Block（含 BatchNorm）
# ============================
def conv_bn(x, filters, kernel_size, strides=1, padding="same"):
    x = layers.Conv2D(
        filters, kernel_size, strides=strides,
        padding=padding, use_bias=False,
    )(x)
    x = layers.BatchNormalization()(x)
    return layers.Activation("relu")(x)


# ============================
#   Inception v2 Block
# ============================
def Inception_A(x, f1, f3_reduce, f3, f5_reduce, f5, pool_proj,
                pool_type="avg", name=None):
    # 1x1
    b1 = conv_bn(x, f1, 1)

    # 1x1 → 3x3
    b2 = conv_bn(x, f3_reduce, 1)
    b2 = conv_bn(b2, f3, 3)

    # 1x1 → 5x5 factorized（兩個 3x3）
    b3 = conv_bn(x, f5_reduce, 1)
    b3 = conv_bn(b3, f5, 3)
    b3 = conv_bn(b3, f5, 3)

    # pool → 1x1：大部分區塊用平均池化，最後一個改用最大池化。
    if pool_type == "avg":
        b4 = layers.AveragePooling2D(3, strides=1, padding="same")(x)
    elif pool_type == "max":
        b4 = layers.MaxPool2D(3, strides=1, padding="same")(x)
    else:
        raise ValueError("pool_type 只能是 avg 或 max")
    b4 = conv_bn(b4, pool_proj, 1)

    return layers.Concatenate(name=name)([b1, b2, b3, b4])


# ============================
#   Reduction Block（縮小特徵圖）
# ============================
def Inception_Reduction(x, f3_reduce, f3, f5_reduce, f5, f5_out,
                        name=None):
    # 1x1 → 3x3，最後一步 stride=2，讓高寬縮小。
    b1 = conv_bn(x, f3_reduce, 1)
    b1 = conv_bn(b1, f3, 3, strides=2)

    # 1x1 → 3x3 → 3x3，最後一步同樣縮小高寬。
    b2 = conv_bn(x, f5_reduce, 1)
    b2 = conv_bn(b2, f5, 3)
    b2 = conv_bn(b2, f5_out, 3, strides=2)

    # 池化分支也縮小高寬，才能和兩條卷積分支串接。
    b3 = layers.MaxPool2D(3, strides=2, padding="same")(x)

    return layers.Concatenate(name=name)([b1, b2, b3])


# ============================
#   Inception v2 Model
# ============================
def InceptionV2(input_shape, num_classes):
    inputs = layers.Input(shape=input_shape)

    # Stem：第一層改成深度可分離卷積，接 BN 與 ReLU。
    # RGB 有 3 通道；depth_multiplier=8，先產生 24 個 DW 通道，再組合成 64 通道。
    x = layers.SeparableConv2D(
        64, 7, strides=2, padding="same",
        depth_multiplier=8, use_bias=False,
    )(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.MaxPool2D(3, strides=2, padding="same")(x)

    x = conv_bn(x, 64, 1)
    x = conv_bn(x, 192, 3)
    x = layers.MaxPool2D(3, strides=2, padding="same")(x)

    # Inception A blocks：低層特徵，輸出通道分別為 256、320。
    x = Inception_A(x, 64, 64, 64, 64, 96, 32, name="Mixed_3b")
    x = Inception_A(x, 64, 64, 96, 64, 96, 64, name="Mixed_3c")

    # Reduction：用多分支縮小高寬，輸出 576 通道。
    x = Inception_Reduction(
        x, 128, 160, 64, 96, 96, name="Mixed_4a",
    )

    # Inception A blocks：中層特徵，四個區塊皆輸出 576 通道。
    x = Inception_A(x, 224, 64, 96, 96, 128, 128, name="Mixed_4b")
    x = Inception_A(x, 192, 96, 128, 96, 128, 128, name="Mixed_4c")
    x = Inception_A(x, 160, 128, 160, 128, 160, 96, name="Mixed_4d")
    x = Inception_A(x, 96, 128, 192, 160, 192, 96, name="Mixed_4e")

    # Reduction：再次縮小高寬，輸出 1024 通道。
    x = Inception_Reduction(
        x, 128, 192, 192, 256, 256, name="Mixed_5a",
    )

    # Inception A blocks（high-level）：最後兩個區塊皆輸出 1024 通道。
    x = Inception_A(x, 352, 192, 320, 160, 224, 128, name="Mixed_5b")
    x = Inception_A(
        x, 352, 192, 320, 192, 224, 128,
        pool_type="max", name="Mixed_5c",
    )

    # Global Pool：配合本專案輸入尺寸，使用全域平均池化。
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.4)(x)  # 保留本專案設定；這是正則化強度，不是模組配置。

    outputs = layers.Dense(num_classes, activation="softmax", name="main")(x)

    return Model(inputs, outputs, name="InceptionV2")


def build_model(input_shape, num_classes):
    model = InceptionV2(input_shape, num_classes)
    model(tf.zeros((1,) + input_shape))  # 建立圖，不開始訓練。
    return model


'''
一般 Inception 區塊：
     input
       │
 ┌─────┼──────┬──────┐
 │     │      │      │
1×1   1×1    1×1    pool
       ↓      ↓      ↓
      3×3   3×3×2   1×1
 │     │      │      │
 └─────┴──────┴──────┴────→ concat → output

Reduction 區塊另外使用 stride=2 的卷積／池化，再把三條分支串接。
'''
