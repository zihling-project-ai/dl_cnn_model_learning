import tensorflow as tf
from tensorflow.keras import layers, Model

# ================= Inception Block =================
def InceptionBlock(x, ch1x1, ch3x3_reduce, ch3x3, ch5x5_reduce, ch5x5, pool_proj):

    # 1x1
    b1 = layers.Conv2D(ch1x1, 1, padding="same", activation="relu")(x)

    # 1x1 → 3x3
    b2 = layers.Conv2D(ch3x3_reduce, 1, padding="same", activation="relu")(x)
    b2 = layers.Conv2D(ch3x3, 3, padding="same", activation="relu")(b2)

    # 1x1 → 5x5
    b3 = layers.Conv2D(ch5x5_reduce, 1, padding="same", activation="relu")(x)
    b3 = layers.Conv2D(ch5x5, 5, padding="same", activation="relu")(b3)

    # maxpool → 1x1
    b4 = layers.MaxPool2D(3, strides=1, padding="same")(x)
    b4 = layers.Conv2D(pool_proj, 1, padding="same", activation="relu")(b4)

    return layers.concatenate([b1, b2, b3, b4])


# ================= Auxiliary Classifier =================
def AuxClassifier(x, num_classes, name):
    x = layers.AvgPool2D(5, strides=3, padding="valid")(x)
    x = layers.Conv2D(128, 1, activation="relu")(x)
    x = layers.Flatten()(x)
    x = layers.Dense(1024, activation="relu")(x)
    x = layers.Dropout(0.7)(x)

    return layers.Dense(num_classes, activation="softmax", name=name)(x)


# ================= GoogLeNet =================
def GoogLeNet(input_shape, num_classes, aux_logits=True):

    inputs = layers.Input(shape=input_shape)

    # Stem
    x = layers.Conv2D(64, 7, strides=2, padding="same", activation="relu")(inputs)
    x = layers.MaxPool2D(3, strides=2, padding="same")(x)

    x = layers.Conv2D(64, 1, activation="relu")(x)
    x = layers.Conv2D(192, 3, padding="same", activation="relu")(x)
    x = layers.MaxPool2D(3, strides=2, padding="same")(x)

    # Inception 3a / 3b
    x = InceptionBlock(x, 64, 96, 128, 16, 32, 32)
    x = InceptionBlock(x, 128, 128, 192, 32, 96, 64)
    x = layers.MaxPool2D(3, strides=2, padding="same")(x)

    # Inception 4a
    x = InceptionBlock(x, 192, 96, 208, 16, 48, 64)
    if aux_logits:
        aux1 = AuxClassifier(x, num_classes, name="aux1")

    # Inception 4b / 4c / 4d
    x = InceptionBlock(x, 160, 112, 224, 24, 64, 64)
    x = InceptionBlock(x, 128, 128, 256, 24, 64, 64)
    x = InceptionBlock(x, 112, 144, 288, 32, 64, 64)

    if aux_logits:
        aux2 = AuxClassifier(x, num_classes, name="aux2")

    # Inception 4e
    x = InceptionBlock(x, 256, 160, 320, 32, 128, 128)
    x = layers.MaxPool2D(3, strides=2, padding="same")(x)

    # Inception 5a / 5b
    x = InceptionBlock(x, 256, 160, 320, 32, 128, 128)
    x = InceptionBlock(x, 384, 192, 384, 48, 128, 128)

    # Final
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.4)(x)
    main_out = layers.Dense(num_classes, activation="softmax", name="main")(x)

    if aux_logits:
        return Model(
            inputs=inputs,
            outputs={"aux1": aux1, "aux2": aux2, "main": main_out},
            name="GoogLeNet"
        )
    else:
        return Model(inputs, main_out, name="GoogLeNet")


def build_model(input_shape, num_classes, aux_logits=True):
    model = GoogLeNet(input_shape, num_classes, aux_logits)
    model(tf.zeros((1,) + input_shape))
    return model
