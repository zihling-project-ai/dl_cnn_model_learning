import tensorflow as tf
from tensorflow.keras import layers

class LRN(tf.keras.layers.Layer):
    def __init__(self, depth_radius=2, bias=1.0, alpha=1e-4, beta=0.75):
        super().__init__()
        self.depth_radius = depth_radius
        self.bias = bias
        self.alpha = alpha
        self.beta = beta

    def call(self, x):
        return tf.nn.local_response_normalization(
            x,
            depth_radius=self.depth_radius,
            bias=self.bias,
            alpha=self.alpha,
            beta=self.beta
        )

class AlexNet(tf.keras.Model):
    def __init__(self, num_classes, input_shape):
        super().__init__()

        self.conv1 = layers.Conv2D(96, 11, strides=4, padding="same", activation="relu")
        self.lrn1 = LRN()
        self.pool1 = layers.MaxPooling2D(3, strides=2)

        self.conv2 = layers.Conv2D(256, 5, padding="same", activation="relu")
        self.lrn2 = LRN()
        self.pool2 = layers.MaxPooling2D(3, strides=2)

        self.conv3 = layers.Conv2D(384, 3, padding="same", activation="relu")
        self.conv4 = layers.Conv2D(384, 3, padding="same", activation="relu")
        self.conv5 = layers.Conv2D(256, 3, padding="same", activation="relu")
        self.pool3 = layers.MaxPooling2D(3, strides=2)

        self.flatten = layers.Flatten()

        self.fc1 = layers.Dense(512, activation="relu")
        self.drop1 = layers.Dropout(0.5)

        self.fc2 = layers.Dense(512, activation="relu")
        self.drop2 = layers.Dropout(0.5)

        self.out_layer = layers.Dense(num_classes, activation="softmax")

    def call(self, x):
        x = self.conv1(x)
        x = self.lrn1(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = self.lrn2(x)
        x = self.pool2(x)

        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.pool3(x)

        x = self.flatten(x)
        x = self.drop1(self.fc1(x))
        x = self.drop2(self.fc2(x))

        return self.out_layer(x)


def build_model(num_classes, input_shape):
    model = AlexNet(num_classes, input_shape)
    # ✔ 正確初始化模型
    dummy = tf.zeros((1,) + input_shape)
    model(dummy)
    
    return model