'''
MobileNet → 把一個標準卷積拆成 DW + PW
Depthwise（3×3 Conv）: 深度卷積  → → 只抽取「空間特徵」(H,W)
Pointwise（1×1 Conv）: 點卷積 PW（混合資訊） → 決定 model capacity → 決定下一層的 channel 數（寬度）
輸出當成下一層的輸入。串 13 次就是整個 MobileNet v1 backbone。
於邊緣裝置應用極廣:Android / iOS app、監視器 、 CCTV、無人機 Drone、IoT 專用晶片（如 Coral Edge TPU、NPU）
'''
import tensorflow as tf
from tensorflow.keras import layers, Model


def conv_bn_relu6(x, filters, kernel_size, strides=1):
    
    """標準卷積：Conv → BN → ReLU6"""
    x = layers.Conv2D(
        filters,
        kernel_size,
        strides=strides,
        padding="same",
        use_bias=False
    )(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU(max_value=6.0)(x)
    return x


# depthwise + pointwise 
def depthwise_separable_conv(x, pointwise_filters, stride):
    
    # ---- Depthwise ----
    x = layers.DepthwiseConv2D(
        kernel_size=3,
        strides=stride,  # <<< 這裡必須用 strides
        padding="same",
        use_bias=False
    )(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU(6.0)(x)

    # ---- Pointwise ----
    x = layers.Conv2D(
        pointwise_filters,
        kernel_size=1,
        padding="same",
        use_bias=False
    )(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU(6.0)(x)

    return x

def MobileNetV1(input_shape, num_classes, alpha=1.0):
    """
    alpha = width multiplier（控制 model 寬度）
    這裡預設 1.0 = full size，之後你若想做輕量版可以調小
    """
    
    inputs = layers.Input(shape=input_shape)
    
    # ---- Stem：第一層標準卷積 ----
    # 原論文：Conv 3x3, 32 filters, stride=2
    x = conv_bn_relu6(inputs, int(32 * alpha), kernel_size=3, strides=2)

    # ---- 一系列 depthwise separable blocks ----
    # 設計來自 MobileNet v1 論文：(輸出 filters, stride)
    # (dw, pw) 各一層，dw 有些 stride=2 會降解析度
    
    cfg = [
        (64,  1),
        (128, 2),
        (128, 1),
        (256, 2),
        (256, 1),
        (512, 2),
        (512, 1),
        (512, 1),
        (512, 1),
        (512, 1),
        (512, 1),
        (1024, 2),
        (1024, 1),
    ]

    for filters, s in cfg:
        x = depthwise_separable_conv(
            x,
            pointwise_filters=int(filters * alpha),
            stride=s
        )

    # # 等同下方 224x224 -> 112x112
    # x = depthwise_separable_conv(x, _c(64),  stride=1)  # 112x112

    # x = depthwise_separable_conv(x, _c(128), stride=2)  # 56x56
    # x = depthwise_separable_conv(x, _c(128), stride=1)  # 56x56

    # x = depthwise_separable_conv(x, _c(256), stride=2)  # 28x28
    # x = depthwise_separable_conv(x, _c(256), stride=1)  # 28x28

    # x = depthwise_separable_conv(x, _c(512), stride=2)  # 14x14
    # for _ in range(5):                                  # 5 個 512 blocks
    #     x = depthwise_separable_conv(x, _c(512), stride=1)  # 14x14

    # x = depthwise_separable_conv(x, _c(1024), stride=2)     # 7x7
    # x = depthwise_separable_conv(x, _c(1024), stride=1)     # 7x7


# ---- Head：Global Average Pooling + Dense ----
    x = layers.GlobalAveragePooling2D()(x)  # 對每個 channel 取平均 → 變成 (batch, channels)
    x = layers.Dropout(0.2)(x)              # 稍微 regularization 一下
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    return Model(inputs, outputs, name="MobileNetV1")



def build_model(num_classes, input_shape, alpha=1.0):
    """
    統一介面：
    - num_classes：分類數
    - input_shape：(H, W, C)
    """
    model = MobileNetV1(input_shape=input_shape, num_classes=num_classes, alpha=alpha)
    _ = model(tf.zeros((1,) + input_shape))
    return model