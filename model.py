import tensorflow as tf
import time as tm
import numpy as np

class AlexNet(tf.keras.Model):
    def __init__(self, layers_end=None):
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
            tf.keras.layers.Dense(10, activation='softmax')
        ]
        layers_end = layers_end if layers_end is not None else len(all_layers)
        self.model = tf.keras.Sequential(all_layers[:layers_end + 1])
        self.optimizer = tf.keras.optimizers.Adam()

    def call(self, inputs):
        return self.model(inputs)

    # Get three training time by one iteration
    def get_time(self, data_set):
        iterator = iter(data_set)
        image, label = next(iterator)
        label = tf.expand_dims(label, axis=0)  # 添加批次维度
        # Add Batch Dimension，because the model expects four-dimensional inputs (batch_size, height, width, channels)
        image = tf.expand_dims(image, axis=0)
        training_times = {
            "forward": [],
            "backward": [],
            "update": []
        }
        with tf.GradientTape(persistent=True) as tape:
            x = image
            intermediates = []
            for layer in self.model.layers:
                start_time = tm.time()
                x = layer(x)
                intermediates.append(x)
                training_times['forward'].append(tm.time() - start_time)
            loss = tf.reduce_mean(tf.losses.categorical_crossentropy(label, x, from_logits=False))

        for idx, layer in enumerate(self.model.layers):
            if layer.trainable_variables:  # Only compute for layers with parameters
                start_time = tm.time()
                grads = tape.gradient(loss, layer.trainable_variables)
                training_times["backward"].append(tm.time() - start_time)
                # 更新参数
                start_time = tm.time()
                tf.keras.optimizers.Adam().apply_gradients(zip(grads, layer.trainable_variables))
                training_times["update"].append(tm.time() - start_time)
            else:
                training_times["backward"].append(0)
                training_times["update"].append(0)
        return training_times

    def get_model_parameters_size(self):
        layer_parameter_bits = []
        for layer in self.model.layers:
            if len(layer.weights) > 0:  # 如果层有参数
                total_bits = 0
                for w in layer.weights:
                    # 参数的总数乘以每个参数的比特数（假设是32位浮点数）
                    total_bits += tf.size(w).numpy() * 32
                layer_parameter_bits.append(total_bits)
            else:
                layer_parameter_bits.append(0)
        return layer_parameter_bits

    def get_layer_output_size(self, data_set):
        iterator = iter(data_set)
        image, label = next(iterator)
        # Add Batch Dimension，because the model expects four-dimensional inputs (batch_size, height, width, channels)
        image = tf.expand_dims(image, axis=0)
        dtype_size = np.dtype(tf.float32.as_numpy_dtype).itemsize * 8
        layer_output_sizes = []
        x = image
        for layer in self.model.layers:
            x = layer(x)
            # 获取当前层输出张量的形状
            output_shape = x.shape
            # 计算输出张量的元素个数
            num_elements = np.prod(output_shape)
            # 计算输出张量的总大小（以比特为单位）
            output_size_bits = num_elements * dtype_size
            layer_output_sizes.append(output_size_bits)
        return layer_output_sizes