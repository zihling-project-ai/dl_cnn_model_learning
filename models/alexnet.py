'''
AlexNet（2012）是第一個在 ImageNet 競賽上把 error 從 26% 一口氣降到 15% 的 CNN
它做了四件深度學習史上很大的突破：ReLU、Dropout、Data Augmentation、多 GPU 訓練
'''
import tensorflow as tf
from tensorflow.keras import layers

class AlexNet(tf.keras.Model):
    def __init__(self, num_classes, input_shape):
        super().__init__()
        self.conv1 = layers.Conv2D(96, kernel_size=11, strides=4, padding="same", activation="relu")
        self.pool1 = layers.MaxPooling2D(pool_size=3, strides=2)

        self.conv2 = layers.Conv2D(256, kernel_size=5, padding="same", activation="relu")
        self.pool2 = layers.MaxPooling2D(pool_size=3, strides=2)

        self.conv3 = layers.Conv2D(384, kernel_size=3, padding="same", activation="relu")
        self.conv4 = layers.Conv2D(384, kernel_size=3, padding="same", activation="relu")
        self.conv5 = layers.Conv2D(256, kernel_size=3, padding="same", activation="relu")
        self.pool3 = layers.MaxPooling2D(pool_size=3, strides=2)

        self.flatten = layers.Flatten()
        
        self.fc1 = layers.Dense(512, activation="relu")
        self.drop1 = layers.Dropout(0.5)

        self.fc2 = layers.Dense(512, activation="relu")
        self.drop2 = layers.Dropout(0.5)
        
        self.out_layer = layers.Dense(num_classes, activation="softmax")
    
    def call(self, x):
        x = self.pool1(self.conv1(x))
        x = self.pool2(self.conv2(x))
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
    