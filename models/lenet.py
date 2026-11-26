'''
LeNet5 模型（1998），經典 CNN 架構
適合較小解析度影像（MNIST / CIFAR 等）。
'''

import tensorflow as tf
from tensorflow.keras import layers


# Subclassing 寫法
class LeNet5(tf.keras.Model):
    """
    初始化（建構子）
    ----------
    num_classes : int
    --> 最後要分類成幾類（你的 AFHQ 是 6 類）
    input_shape : tuple
    --> 輸入圖片大小，例如 (224, 224, 3)
    """
    
    def __init__(self, num_classes, input_shape):
        super().__init__()
      
        # 卷積池化層
        self.conv1 = layers.Conv2D(6, kernel_size=5, activation ='tanh')
        self.pool1 = layers.AveragePooling2D()
        self.conv2 = layers.Conv2D(16, kernel_size=5, activation ='tanh')
        self.pool2 = layers.AveragePooling2D()
        

        # 展平層+全連接層
        self.flatten = layers.Flatten()
        self.fc1 = layers.Dense(120, activation = 'tanh')
        self.fc2 = layers.Dense(84, activation='tanh')

        # 輸出層
        self.out_layer = layers.Dense(num_classes, activation='softmax')

    """
    建構方法 : 前向傳遞（forward pass）
    TensorFlow 會在 model(x) 時自動呼叫本函式
    """
    def call(self, x):
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.fc2(x)

        return self.out_layer(x)

def build_model(num_classes, input_shape):
    
    model = LeNet5(num_classes=num_classes, input_shape=input_shape)
    model.build((None, *input_shape))
    
    return model

# Sequential 寫法
# def build_lenet5_sequential(input_shape, num_classes):
#     model = models.Sequential([
#         layers.Conv2D(6, kernel_size=5, activation='tanh', input_shape=input_shape),
#         layers.AveragePooling2D(),
#         layers.Conv2D(16, kernel_size=5, activation='tanh'),
#         layers.AveragePooling2D(),
#         layers.Flatten(),
#         layers.Dense(120, activation='tanh'),
#         layers.Dense(84, activation='tanh'),
#         layers.Dense(num_classes, activation='softmax')
#     ])
#     return model