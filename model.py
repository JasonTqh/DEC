import tensorflow as tf

class AlexNet(tf.keras.Model):
    def __init__(self,layers_end=None):
        super(AlexNet, self).__init__()
        all_layers = [
            tf.keras.layers.Conv2D(filters=96, kernel_size=(11, 11), strides=(4, 4), activation='relu'),
            tf.keras.layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2)),
            tf.keras.layers.Conv2D(filters=256, kernel_size=(5, 5), activation='relu', padding='same'),
            tf.keras.layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2)),
            tf.keras.layers.Conv2D(filters=384, kernel_size=(3, 3), activation='relu', padding='same'),
            tf.keras.layers.Conv2D(filters=384, kernel_size=(3, 3), activation='relu', padding='same'),
            tf.keras.layers.Conv2D(filters=256, kernel_size=(3, 3), activation='relu', padding='same'),
            tf.keras.layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2)),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(4096, activation='relu'),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(4096, activation='relu'),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(1000, activation='softmax')
        ]
        layers_end = layers_end if layers_end is not None else len(all_layers)
        self.model = tf.keras.Sequential(all_layers[:layers_end + 1])

def call(self, inputs):
    return self.model(inputs)