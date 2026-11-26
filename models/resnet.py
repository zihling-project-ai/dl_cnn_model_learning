'''
ResNet (2015)通過設計解決深度神經網絡中的梯度消失和梯度爆炸問題
1. 引入“殘差塊”（Residual Blocks）與“捷徑連接”（Shortcut Connection）
2. 殘差學習（Residual Learning）: 殘差塊就是兩三層卷積後就會跟原始的輸入相加 H(x)=F(x)+x 避免因反向傳播梯度消失問題
'''
import tensorflow as tf
from tensorflow.keras import layers, Model

# ------- Conv-BN-ReLU -------
def conv_bn_relu(x, filters, kernel_size, stride=1):
    x = layers.Conv2D(filters, kernel_size, strides=stride,
                      padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    return layers.ReLU()(x)

# ------- Bottleneck Block -------
def bottleneck_block(x, filters, stride=1, shortcut=False):

    shortcut_x = x  # 保存原始 x

    # --- Residual path F(x) ---
    '''
    先用 1×1 降維 256 → 64  
    再用 3×3 在 64 channel 做計算（便宜很多）  
    最後用 1×1 提升回 256 (64 × 4)
    '''
    x = conv_bn_relu(x, filters, 1, stride=stride) # 1×1 降維
    x = conv_bn_relu(x, filters, 3) # 3×3 卷積

    # 1×1 升維（filters*4 = output channels）
    x = layers.Conv2D(filters * 4, 1, use_bias=False)(x)
    x = layers.BatchNormalization()(x)

    # --- Shortcut path ---
    '''
    output = F(x) + x，要能相加，必須滿足shape(F(x)) == shape(x)
    height、width、channels一樣
    輸入維度：256 = x = shortcut_x
    1×1 降維 64 * 4 = 256
    '''
    if shortcut or stride != 1:
        shortcut_x = layers.Conv2D(filters * 4, 1, strides=stride,
                                   use_bias=False)(shortcut_x)
        shortcut_x = layers.BatchNormalization()(shortcut_x)

    # merge
    x = layers.Add()([x, shortcut_x])
    return layers.ReLU()(x)

# ------- Stage Builder -------
def make_stage(x, filters, blocks, stride):

    # 第一個 block 需要降維 → shortcut=True, stride=stride
    x = bottleneck_block(x, filters, stride=stride, shortcut=True)

    # 後續 blocks stride=1
    for _ in range(1, blocks):
        x = bottleneck_block(x, filters, stride=1, shortcut=False)

    return x

# ------- ResNet50 -------
def ResNet50(input_shape, num_classes):
    
    inputs = layers.Input(shape=input_shape)

    # Stem
    x = conv_bn_relu(inputs, 64, 7, stride=2)
    x = layers.MaxPool2D(3, strides=2, padding="same")(x)

    # Stages
    x = make_stage(x, 64,  blocks=3, stride=1)  # conv2_x
    x = make_stage(x, 128, blocks=4, stride=2)  # conv3_x
    x = make_stage(x, 256, blocks=6, stride=2)  # conv4_x
    x = make_stage(x, 512, blocks=3, stride=2)  # conv5_x

    # Head
    x = layers.GlobalAveragePooling2D()(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    return Model(inputs, outputs, name="ResNet50")


def build_model(input_shape, num_classes):
    model = ResNet50(input_shape, num_classes)
    model(tf.zeros((1,) + input_shape))
    return model



'''
         ┌─────────────── shortcut ───────────────┐
         │                                          │
x ──► Conv1x1 ─► Conv3x3 ─► Conv1x1 ─► BN ───► F(x) │
         │                                          │
         └──────────── Add(F(x), x) ────────────────┘
                               │
                             ReLU
                               │
                             output

'''
'''
殘差相加的本質是 element-wise add → 只能在等形狀 tensor 上進行
1.主路徑 F(x) 維持 bottleneck output = filters×4
2.shortcut 會用 1×1 conv 調整到同樣的 channel
3.H,W 若因 stride 改變，也需要 conv 來調整
              (56,56,256)
                    │
                    ▼
            ┌───────Conv1×1 (256→64)
            │
            ├───────Conv3×3 (64→64)
            │
            └───────Conv1×1 (64→256)
                    │
                  F(x)
                    │
shortcut────────────┘  兩者 shape 必須完全一樣
                    ▼
                Add + ReLU
             (56,56,256)

'''