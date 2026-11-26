'''
EfficientNet-B0（2019）架構修正版
EfficientNet 創新：Compound Scaling（複合縮放）
EfficientNet 協調 Depth / Width / Resolution 的縮放；本檔固定使用 B0，不提供 B1～B7。
EfficientNet 的基礎積木：MBConv（Bottleneck + Depthwise）(MobileNet + ResNet + Attention)
MBConv 架構:
輸入
 → 1×1 PW (expand)
 → 3×3 DW（depthwise）
 → SE（注意力機制）
 → 1×1 PW（project）
 → 支路隨機丟棄（僅訓練且可殘差相加時）
 → 殘差連接（if shape match）

參考：https://github.com/tensorflow/tpu/blob/master/models/official/efficientnet/efficientnet_model.py
此實作對齊 SE 寬度與支路遮罩；保留本專案的 BN、初始化與訓練設定。
'''

import tensorflow as tf
from tensorflow.keras import layers, Model

# ------------------------
#   Squeeze-and-Excite Block:
#   讓 CNN 在通道維度（Channel）上有「注意力機制 Attention」，
#   自己學會「哪個 channel 比較重要」，給它更大的權重
# ------------------------
def SE_Block(inputs, in_filters, se_ratio=0.25):
    
    filters = inputs.shape[-1]  # 此時是擴張後的特徵通道數。
    # 修正：SE 中間層以 MBConv「擴張前」的通道計算，不使用 filters。
    reduced = max(1, int(in_filters * se_ratio))
    
    # Squeeze : 全域平均池化，壓成 (B, C)
    se = layers.GlobalAveragePooling2D()(inputs)
    # 變回 (B, 1, 1, C)
    se = layers.Reshape((1, 1, filters))(se)

    # Excitation：先降維再升回來；EfficientNet 在此使用 Swish，再接 Sigmoid。 
    se = layers.Conv2D(reduced, 1, activation="swish", padding="same")(se)
    se = layers.Conv2D(filters, 1, activation="sigmoid", padding="same")(se)

    # Scale: 對每個 channel 乘上權重
    return layers.Multiply()([inputs, se])

# ------------------------
#   MBConv Block
    # MBConv = (1x1 expand) -> (kxk depthwise) -> (SE) -> (1x1 project) (+ residual)
    # - expand_ratio: 擴張倍數
    # - out_filters: 輸出 channel
    # - stride: depthwise 的 stride
    # - kernel_size: 3 或 5
    # - drop_rate: 殘差支路的丟棄機率（每張圖片一個遮罩）
# ------------------------
def MBConv(x, expand_ratio, out_filters, stride, kernel_size=3, se_ratio=0.25, drop_rate=0.0):
    in_filters = x.shape[-1]
    shortcut = x

    # 1. Expand : 先擴展維度 → 提升其表示能力 → 再做 depthwise。
    expanded = in_filters * expand_ratio
    if expand_ratio != 1:
        x = layers.Conv2D(expanded, 1, use_bias=False, padding="same")(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("swish")(x)

    # 2. Depthwise Conv : 與 Conv2D(k×k)不同，不會混合 channel
    x = layers.DepthwiseConv2D(kernel_size=kernel_size, strides=stride, padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("swish")(x)

    # 3. SE block
    x = SE_Block(x, in_filters=in_filters, se_ratio=se_ratio)

    # 4. Project (1x1 Conv)
    x = layers.Conv2D(out_filters, 1, use_bias=False, padding="same")(x)
    x = layers.BatchNormalization()(x)

    # 5. Residual connect + DropConnect（此處指逐樣本丟棄殘差支路輸出）
    '''
    若 stride=1 且 channel 數相同 → 這是 Residual Block
    在相加前，每張圖片各抽一個遮罩，整條殘差支路一起保留或設為 0。
    丟棄時只留下 shortcut；保留時將支路除以存活率，維持期望值。
    這不會跳過前面的卷積計算，只是把算好的支路輸出設為 0。
    推論時關閉隨機丟棄，最後仍做 shortcut + F(x)。
    '''
    if stride == 1 and in_filters == out_filters:
        if drop_rate and drop_rate > 0.0:
            # 修正：(batch, 1, 1, 1)，同一張圖片的 H、W、C 共用一個遮罩。
            # Keras Dropout 自動處理存活率縮放，並在推論時關閉。
            x = layers.Dropout(
                drop_rate, noise_shape=(None, 1, 1, 1),
            )(x)
        x = layers.Add()([shortcut, x])

    return x


# ------------------------
#   EfficientNet-B0 Backbone
# ------------------------
def EfficientNetB0(input_shape, num_classes, drop_connect_rate=0.2):

    inputs = layers.Input(input_shape)
    x = inputs

    # Stem：3x3 conv, 32 filters, stride=2
    x = layers.Conv2D(32, 3, strides=2, padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("swish")(x)

    # 論文 B0 配置： (expand, out, repeats, stride, kernel)
    cfg = [
        (1,  16, 1, 1, 3),
        (6,  24, 2, 2, 3),
        (6,  40, 2, 2, 5),
        (6,  80, 3, 2, 3),
        (6, 112, 3, 1, 5),
        (6, 192, 4, 2, 5),
        (6, 320, 1, 1, 3),
    ]

    # 為了讓 DropConnect 隨深度漸強，先算 block 總數
    total_blocks = sum(r for (_, _, r, _, _) in cfg)
    block_idx = 0

    for expand, out, repeats, stride, k in cfg:
        for i in range(repeats):
            # 第一個 block 用 stride，其餘用 1
            s = stride if i == 0 else 1

            # 線性增加 drop rate，依參考實作使用 block_idx / total_blocks。
            drop_rate = drop_connect_rate * float(block_idx) / float(total_blocks)

            x = MBConv(
                x,
                expand_ratio=expand,
                out_filters=out,
                stride=s,
                kernel_size=k,
                se_ratio=0.25,
                drop_rate=drop_rate
            )
            block_idx += 1

    # Head
    x = layers.Conv2D(1280, 1, use_bias=False, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("swish")(x)

    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.2)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    return Model(inputs, outputs, name="EfficientNetB0")



def build_model(num_classes, input_shape):

    model = EfficientNetB0(input_shape=input_shape, num_classes=num_classes)
    _ = model(tf.zeros((1,) + input_shape))  # 建立圖，不開始訓練。
    return model
