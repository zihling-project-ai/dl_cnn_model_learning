'''
VGG網路最大的特色為使用多個非常小size(3×3)的filter
將CNN模型的深度往下推深至16–19層，
其中選擇使用size(3×3)的filter原因，
是因為這是能捕捉周圍訊息(左/右,上/下)的最小大小
'''
import tensorflow as tf
from tensorflow.keras import layers

class VGG16(tf.keras.Model):
    def __init__(self, num_classes, input_shape):
        super().__init__()
        
        # Block 1
        self.conv1_1 = layers.Conv2D(64, (3,3), padding="same", activation="relu")
        self.conv1_2 = layers.Conv2D(64, (3,3), padding="same", activation="relu")
        self.pool1 = layers.MaxPooling2D((2,2))

        # Block 2
        self.conv2_1 = layers.Conv2D(128, (3,3), padding="same", activation="relu")
        self.conv2_2 = layers.Conv2D(128, (3,3), padding="same", activation="relu")
        self.pool2 = layers.MaxPooling2D((2,2))

        # Block 3
        self.conv3_1 = layers.Conv2D(256, (3,3), padding="same", activation="relu")
        self.conv3_2 = layers.Conv2D(256, (3,3), padding="same", activation="relu")
        self.conv3_3 = layers.Conv2D(256, (3,3), padding="same", activation="relu")
        self.pool3 = layers.MaxPooling2D((2,2))

       # Block 4
        self.conv4_1 = layers.Conv2D(512, (3,3), padding="same", activation="relu")
        self.conv4_2 = layers.Conv2D(512, (3,3), padding="same", activation="relu")
        self.conv4_3 = layers.Conv2D(512, (3,3), padding="same", activation="relu")
        self.pool4 = layers.MaxPooling2D((2,2))

        # Block 5
        self.conv5_1 = layers.Conv2D(512, (3,3), padding="same", activation="relu")
        self.conv5_2 = layers.Conv2D(512, (3,3), padding="same", activation="relu")
        self.conv5_3 = layers.Conv2D(512, (3,3), padding="same", activation="relu")
        self.pool5 = layers.MaxPooling2D((2,2))    # Block 1

        # FC layers
        self.flatten = layers.Flatten()
        self.fc1 = layers.Dense(256, activation="relu")
        self.drop1 = layers.Dropout(0.5)
        self.fc2 = layers.Dense(256, activation="relu")
        self.drop2 = layers.Dropout(0.5)
        self.out_layer = layers.Dense(num_classes, activation="softmax")
   
    def call(self, x):
        x = self.pool1(self.conv1_2(self.conv1_1(x)))
        x = self.pool2(self.conv2_2(self.conv2_1(x)))
        x = self.pool3(self.conv3_3(self.conv3_2(self.conv3_1(x))))
        x = self.pool4(self.conv4_3(self.conv4_2(self.conv4_1(x))))
        x = self.pool5(self.conv5_3(self.conv5_2(self.conv5_1(x))))
        x = self.flatten(x)
        x = self.drop1(self.fc1(x))
        x = self.drop2(self.fc2(x))
        return self.out_layer(x)

def build_model(num_classes, input_shape):
    model = VGG16(num_classes=num_classes, input_shape=input_shape)
    model(tf.zeros((1,) + input_shape))  # 初始化
    return model


