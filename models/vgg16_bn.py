import tensorflow as tf
from tensorflow.keras import layers

class VGG16_BN(tf.keras.Model):
    def __init__(self, num_classes, input_shape):
        super().__init__()

        # Block 1
        self.conv1_1 = layers.Conv2D(64, 3, padding="same")
        self.bn1_1   = layers.BatchNormalization()
        self.conv1_2 = layers.Conv2D(64, 3, padding="same")
        self.bn1_2   = layers.BatchNormalization()
        self.pool1   = layers.MaxPooling2D(2)

        # Block 2
        self.conv2_1 = layers.Conv2D(128, 3, padding="same")
        self.bn2_1   = layers.BatchNormalization()
        self.conv2_2 = layers.Conv2D(128, 3, padding="same")
        self.bn2_2   = layers.BatchNormalization()
        self.pool2   = layers.MaxPooling2D(2)

        # Block 3
        self.conv3_1 = layers.Conv2D(256, 3, padding="same")
        self.bn3_1   = layers.BatchNormalization()
        self.conv3_2 = layers.Conv2D(256, 3, padding="same")
        self.bn3_2   = layers.BatchNormalization()
        self.conv3_3 = layers.Conv2D(256, 3, padding="same")
        self.bn3_3   = layers.BatchNormalization()
        self.pool3   = layers.MaxPooling2D(2)

        # Block 4
        self.conv4_1 = layers.Conv2D(512, 3, padding="same")
        self.bn4_1   = layers.BatchNormalization()
        self.conv4_2 = layers.Conv2D(512, 3, padding="same")
        self.bn4_2   = layers.BatchNormalization()
        self.conv4_3 = layers.Conv2D(512, 3, padding="same")
        self.bn4_3   = layers.BatchNormalization()
        self.pool4   = layers.MaxPooling2D(2)

        # Block 5
        self.conv5_1 = layers.Conv2D(512, 3, padding="same")
        self.bn5_1   = layers.BatchNormalization()
        self.conv5_2 = layers.Conv2D(512, 3, padding="same")
        self.bn5_2   = layers.BatchNormalization()
        self.conv5_3 = layers.Conv2D(512, 3, padding="same")
        self.bn5_3   = layers.BatchNormalization()
        self.pool5   = layers.MaxPooling2D(2)

        # FC
        self.flatten = layers.Flatten()
        self.fc1 = layers.Dense(512, activation="relu")
        self.drop1 = layers.Dropout(0.5)
        self.fc2 = layers.Dense(512, activation="relu")
        self.drop2 = layers.Dropout(0.5)
        self.out_layer = layers.Dense(num_classes, activation="softmax")

    def call(self, x):
        # Block 1
        x = self.pool1(tf.nn.relu(self.bn1_2(self.conv1_2(tf.nn.relu(self.bn1_1(self.conv1_1(x)))))))
        # Block 2
        x = self.pool2(tf.nn.relu(self.bn2_2(self.conv2_2(tf.nn.relu(self.bn2_1(self.conv2_1(x)))))))
        # Block 3
        x = self.pool3(tf.nn.relu(self.bn3_3(self.conv3_3(tf.nn.relu(self.bn3_2(self.conv3_2(tf.nn.relu(self.bn3_1(self.conv3_1(x))))))))))
        # Block 4
        x = self.pool4(tf.nn.relu(self.bn4_3(self.conv4_3(tf.nn.relu(self.bn4_2(self.conv4_2(tf.nn.relu(self.bn4_1(self.conv4_1(x))))))))))
        # Block 5
        x = self.pool5(tf.nn.relu(self.bn5_3(self.conv5_3(tf.nn.relu(self.bn5_2(self.conv5_2(tf.nn.relu(self.bn5_1(self.conv5_1(x))))))))))
        
        x = self.flatten(x)
        x = self.drop1(self.fc1(x))
        x = self.drop2(self.fc2(x))
        return self.out_layer(x)

def build_model(input_shape, num_classes):
    model = VGG16_BN(num_classes=num_classes, input_shape=input_shape)
    model(tf.zeros((1,) + input_shape))  
    return model
