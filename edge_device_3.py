import pickle
import numpy as np
import tensorflow as tf
import socket
from model import AlexNet
import os

# loading dataset
def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict
def load_cifar10_data(data_dir):
    train_data = None
    train_labels = []

    for i in range(1, 6):
        data_path = os.path.join(data_dir, f"data_batch_{i}")
        batch_data = unpickle(data_path)
        if train_data is None:
            train_data = batch_data[b'data']
        else:
            train_data = np.vstack((train_data, batch_data[b'data']))
        train_labels += batch_data[b'labels']

    train_data = train_data.reshape((len(train_data), 3, 32, 32))
    train_data = np.transpose(train_data, (0, 2, 3, 1))

    return train_data, np.array(train_labels)
def preprocess_image(image, label):
    image = tf.image.resize(image, [227, 227])  # Resize to the input size of AlexNet
    image = tf.cast(image, tf.float32)
    image /= 255.0  # Normalize to [0, 1]
    label = tf.one_hot(label, depth=10)
    return image, label
def make_dataset(single_sample=False, data_dir="cifar-10-batches-py/"):
    train_images, train_labels = load_cifar10_data(data_dir)
    train_dataset = tf.data.Dataset.from_tensor_slices((train_images, train_labels))
    train_dataset = train_dataset.map(preprocess_image)
    if not single_sample:
        train_dataset = train_dataset.batch(128).shuffle(1000)
    return train_dataset


def calculate_averages(data):
    # 初始化存储平均值的字典
    averages = {'forward': [], 'backward': [], 'update': []}

    # 获取列表长度
    num_entries = len(data)

    # 对每个键进行操作
    for key in averages.keys():
        # 获取每个位置的长度
        length_of_lists = len(data[0][key])

        # 遍历每个位置
        for index in range(length_of_lists):
            # 计算每个位置的平均值
            sum_at_index = sum(d[key][index] for d in data if index < len(d[key]))
            count_at_index = sum(1 for d in data if index < len(d[key]) and d[key][index] is not None)
            average_at_index = sum_at_index / count_at_index if count_at_index > 0 else 0
            averages[key].append(average_at_index)

    return averages
# Get the performance of the current device
def get_device_performance():
    # Get the number of gpu of the current device
    # pynvml.nvmlInit()
    # gpu_count = pynvml.nvmlDeviceGetCount()
    # print("GPU count:", gpu_count)
    # pynvml.nvmlShutdown()

    # Get the number of cpu of the current device
    # cpu_count = psutil.cpu_count(logical=True)
    cpu_count = 2
    print("logical CPUs count:", cpu_count)
    # Get the RAM size of the current device
    # ram_info = psutil.virtual_memory()
    # ram = ram_info.total / (1024 ** 3)
    ram = 2
    print("RAM: ", ram, "GB")
    score = 0.7 * cpu_count + 0.3 * ram
    return score

def send_edge_device_info(data, score, device,host, port):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.connect((host, port))
        data = pickle.dumps((data, score, device))
        print(data)
        s.sendall(data)
        s.close()
def receive_cloud_info(host, port):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((host, port))
        s.listen()
        s.settimeout(180)
        try:
            conn, addr = s.accept()
            with conn:
                conn.settimeout(60)
                data = b''
                while True:
                    try:
                        packet = conn.recv(4096)
                        if not packet:
                            break
                        data += packet
                    except socket.timeout:
                        print("Connection timed out while receiving data")
                        break
                if data:
                    params = pickle.loads(data)
                    return params
        except socket.timeout:
            print("No connection was made within the timeout period.")
        except Exception as e:
            print(f"An error occurred: {e}")
    return None
def receive_number_sample(host, port):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((host, port))
        s.listen()
        s.settimeout(180)
        try:
            conn, addr = s.accept()
            with conn:
                conn.settimeout(60)
                data = b''
                while True:
                    try:
                        packet = conn.recv(4096)
                        if not packet:
                            break
                        data += packet
                    except socket.timeout:
                        print("Connection timed out while receiving data")
                        break
                if data:
                    params = pickle.loads(data)
                    return params
        except socket.timeout:
            print("No connection was made within the timeout period.")
        except Exception as e:
            print(f"An error occurred: {e}")
    return None
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
    max_layers = 14
    # cloud_info = receive_cloud_info('172.18.0.2', 8080 )
    # layers = cloud_info
    model = AlexNet(max_layers)
    print("Model loaded.")
    single_sample_data_set = make_dataset(single_sample=True)
    data_set = make_dataset()
    print("Dataset loaded.")
    train_times_list = []
    for i in range(50):
        train_times_list.append(model.get_time(single_sample_data_set))
    train_times = calculate_averages(train_times_list)
    score = get_device_performance()
    print(train_times, score)
    # send_edge_device_info(train_times, score, 2, '172.18.0.2', 8080)
    # print('训练时间和性能得分发送成功')
    # number_sample_start, number_sample_end = receive_number_sample('172.18.0.12', 9092)
    # data_set_device = data_set.skip(number_sample_start).take(number_sample_start, number_sample_end)
    # train_and_evaluate(model, data_set_device)
    # weights = model.get_weights()
    # send_edge_device_parameters(weights, '172.18.0.2', 8080)

