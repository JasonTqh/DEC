import pickle
import time

import tensorflow as tf
import pynvml
import psutil
import socket
from model import AlexNet

def preprocess_image(image, label):
    image = tf.image.resize(image, [227, 227])  # Resize to the input size of AlexNet
    image = tf.cast(image, tf.float32)
    image /= 255.0  # Normalize to [0, 1]
    return image, label
def make_dataset():
    # Load the CIFAR-10 dataset
    (train_images, train_labels), _ = tf.keras.datasets.cifar10.load_data()
    train_dataset = tf.data.Dataset.from_tensor_slices((train_images, train_labels))
    train_dataset = train_dataset.map(preprocess_image).batch(128).shuffle(1000)
    return train_dataset


def get_time(model, data_set):
    iterator = iter(data_set)
    first_batch = next(iterator)
    image, label = first_batch[0][0], first_batch[1][0]
    training_times = {
        "forward": [],
        "backward": [],
        "update": []
    }
    with tf.GradientTape(persistent=True) as tape:
        x = image
        intermediates = []
        for layer in model.layers:
            start_time = time.time()
            x = layer(x)
            intermediates.append(x)
            training_times['forward'].append(time.time() - start_time)
        predictions = model(image, training=True)
        loss = tf.reduce_mean(tf.losses.categorical_crossentropy(label, x, from_logits=True))
    for idx, layer in enumerate(model.layers):
        start_time = time.time()
        grads = tape.gradient(loss, layer.trainable_variables)
        training_times["backward"].append(time.time() - start)
        # 更新参数
        start = time.time()
        model.optimizer.apply_gradients(zip(grads, model.trainable_variables))
        training_times["update"].append(time.time() - start)
    return training_times
def get_device_performance():
    # Get the number of gpu of the current device
    pynvml.nvmlInit()
    gpu_count = pynvml.nvmlDeviceGetCount()
    print("GPU count:", gpu_count)
    pynvml.nvmlShutdown()
    # Get the RAM size of the current device
    ram_info = psutil.virtual_memory()
    ram = ram_info.total / (1024 ** 3)
    print("RAM: ", ram, "GB")

    score = 0.7 * gpu_count + 0.3 * ram
    return score


def send_edge_server_info(data, score, host, port):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.connect((host, port))
        s.sendall(data, score)
def receive_cloud_info(host, port):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((host, port))
        s.listen()
        conn, addr = s.accept()
        with conn:
            data = b''
            while True:
                packet = conn.recv(4096)
                if not packet:
                    break
                data += packet
            params = pickle.loads(data)
    return params
def receive_device_info(host, port):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((host, port))
        s.listen()
        conn, addr = s.accept()
        with conn:
            data = b''
            while True:
                packet = conn.recv(4096)
                if not packet:
                    break
                data += packet
            params = pickle.loads(data)
    return params
def receive_number_sample(host, port):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((host, port))
        s.listen()
        conn, addr = s.accept()
        with conn:
            data = b''
            while True:
                packet = conn.recv(4096)
                if not packet:
                    break
                data += packet
            params = pickle.loads(data)
    return params
def send_edge_device_parameters(parameters, host, port):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.connect((host, port))
        s.sendall(parameters, score)

# training process
# data-parallel
def train_model_stop(model, data_set, stop_layer):
    for x, y in data_set:
        with tf.GradientTape() as tape:
            predictions = model(x, training=True)
            loss = tf.keras.losses.sparse_categorical_crossentropy(y, predictions)

        # Get weights that need to be updated (train only to the specified layer)
        trainable_vars = model.trainable_variables[:stop_layer]
        gradients = tape.gradient(loss, trainable_vars)
        model.optimizer.apply_gradients(zip(gradients, trainable_vars))
def train_and_evaluate(model, data_set, epochs=1):
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.fit(data_set, epochs=epochs)

if __name__ == "__main__":
    host = ['172.18.0.10', '172.18.0.11', '172.18.0.12']
    port = ['9091', '9092', '9093']
    cloud_info = receive_cloud_info('172.18.0.2', '8080' )
    layers = cloud_info
    model = AlexNet(layers)
    data_set = make_dataset()
    train_times = get_time(model, data_set)
    score = get_device_performance()
    send_edge_server_info(train_times, score, '172.18.0.2', '8080')
    number_sample_start, number_sample_end = receive_number_sample('172.18.0.2', '8080')
    md = receive_cloud_info('172.18.0.2', '8080')
    data_set_edge = data_set.skip(number_sample_start).take(number_sample_start, number_sample_end)
    train_model_stop(model, data_set_edge, md)
    weights = model.get_weights()
    weights_d=[]
    weights_d[0] = receive_device_info(host[0], port[0])
    weights_d[1] = receive_device_info(host[1], port[1])
    weights_d[2] = receive_device_info(host[2], port[2])
    for i in range(md):
        weights[i] = (weights_d[0][i] + weights_d[1][i]+weights_d[2][i]) / 3  # 简单平均聚合
    model.set_weights(weights)
    train_model_stop(model, data_set_edge, layers)
    weights = model.get_weights()
    send_edge_device_parameters(weights, '172.18.0.2', '8080')
