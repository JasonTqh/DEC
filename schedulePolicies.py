import time
import pickle
import tensorflow as tf
import pynvml
import psutil
import socket
from pulp import *
from model import AlexNet


# loading dataset
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


# send and receive data
def receive_info(host, port):
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
def send_model_layers(layers, host, port):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.connect((host, port))
        s.sendall(layers)
def send_number_sample(number_sample_start, number_sample_end, host, port):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.connect((host, port))
        s.sendall(number_sample_start, number_sample_end)


# Get three training time by one iteration
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
# Get the performance of the current device
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
def get_model_parameters_size(model):
    layer_parameter_bits=[]
    for layer in model.layers:
        if len(layer.weights) > 0:  # 如果层有参数
            total_bits = 0
            for w in layer.weights:
                # 参数的总数乘以每个参数的比特数（假设是32位浮点数）
                total_bits += tf.size(w).numpy() * 32
            layer_parameter_bits.append(total_bits)
    return layer_parameter_bits
def get_layer_output_size(model, data_set):
    iterator = iter(data_set)
    first_batch = next(iterator)
    image= first_batch[0][0]
    layer_bits = []
    intermediate_layer_model = tf.keras.models.Model(inputs=model.input,
                                                     outputs=[layer.output for layer in model.layers])
    intermediate_outputs = intermediate_layer_model(image)
    # 计算并打印每层输出的大小（以位为单位）
    for i, output in enumerate(intermediate_outputs):
        # 获取元素的总数
        total_elements = tf.size(output).numpy()
        # 获取每个元素的位数，假设使用的是float32，则每个元素32位
        element_size_bits = output.dtype.size * 8
        # 计算总位数
        total_bits = total_elements * element_size_bits
        layer_bits.append(total_bits)
    return layer_bits
def data_parallel(total_number_device, score, m):
    total_score_device = 0
    number_sample_device = []
    for i in range(m):
        total_score_device += score[i]
    for i in range(m):
        number_sample_device[i] = (score[i] / total_score_device) * total_number_device
    return number_sample_device

def total_training_time_one_iteration(md, me, max_layers, train_times, train_times_e, train_times_c, score, layer_bits, bandwith_device, layer_parameters_bits, total_number_samples ):
    number_samples_end=[]
    End = ['device', 'edge', 'cloud']
    prob = LpProblem("The total training time Problem", LpMinimize)
    end_vars = LpVariable.dicts('End', End, 0)

    number_sample_device = data_parallel(end_vars[0], score,3)

    total_forward_time_device_d = []
    total_forward_time_edge_d = 0
    total_forward_time_cloud_d = 0
    for j in range(3):
        for i in range(md):
            total_forward_time_device_d[j] += number_sample_device[j]*train_times[j]['forward'][i]
        total_forward_time_device_d[j] += (layer_bits[md-1]*number_sample_device[j])/bandwith_device[j]
    for i in range(md):
        total_forward_time_edge_d += end_vars[1] * train_times_e['forward'][i]
        total_forward_time_cloud_d += end_vars[2] * train_times_c['forward'][i]

    total_backward_time_device_d = []
    total_backward_time_edge_d = 0
    total_backward_time_cloud_d = 0
    for j in range(3):
        for i in range(md):
            total_backward_time_device_d[j] += number_sample_device[j]*train_times[j]['backward'][i]
        total_backward_time_device_d[j] += (layer_bits[md-1]*number_sample_device[j])/bandwith_device[j]
    for i in range(md):
        total_backward_time_edge_d += end_vars[1] * train_times_e['backward'][i]
        total_backward_time_cloud_d += end_vars[2] * train_times_c['backward'][i]
    total_forward_time_device_module = max(max(total_forward_time_device_d), total_forward_time_edge_d, total_forward_time_cloud_d)
    total_backward_time_device_module = max(max(total_backward_time_device_d), total_backward_time_edge_d, total_backward_time_cloud_d)

    total_forward_time_edge_e = 0
    total_forward_time_cloud_e = 0
    total_backward_time_edge_e = 0
    total_backward_time_cloud_e = 0
    for i in range(md, me):
        total_forward_time_edge_e += end_vars[1] * train_times_e['forward'][i]
        total_forward_time_cloud_e += end_vars[2] * train_times_c['forward'][i]
        total_backward_time_edge_e += end_vars[1] * train_times_e['backward'][i]
        total_backward_time_cloud_e += end_vars[2] * train_times_c['backward'][i]
    total_forward_time_edge_e += (layer_bits[me-1] * end_vars[1])/(1024*1024)
    total_backward_time_edge_e += (layer_bits[me - 1] * end_vars[1]) / (1024 * 1024)
    total_forward_time_edge_module = max(max(total_forward_time_edge_e), total_forward_time_cloud_e)
    total_backward_time_edge_module = max(max(total_backward_time_edge_e), total_backward_time_cloud_e)

    total_forward_time_cloud_module=0
    total_backward_time_cloud_module=0
    for i in range(me, max_layers):
        total_forward_time_cloud_module += end_vars[2] * train_times_c['forward'][i]
        total_backward_time_cloud_module += end_vars[2] * train_times_c['backward'][i]

    total_update_time_cloud = 0
    total_update_time_edge = 0
    total_update_time_device = []
    parameters_to_edge = 0
    parameters_to_device = 0
    for i in range(max_layers):
        total_update_time_cloud += train_times_c['update'][i]
    for i in range(me):
        total_update_time_edge += train_times_e['update'][i]
        parameters_to_edge += layer_parameters_bits[i]
    for j in range(3):
        for i in range(md):
            total_update_time_device[j]= train_times[j]['update'][i]
    for i in range(md):
        parameters_to_device += layer_parameters_bits[i]

    total_update_time = max(total_update_time_cloud, total_update_time_edge, max(total_update_time_device)) + max(parameters_to_edge/(1024*1024), parameters_to_device/(512*1024))

    # target equation
    prob += lpSum(total_forward_time_device_module, total_backward_time_device_module, total_forward_time_edge_module, total_backward_time_edge_module, total_forward_time_cloud_module, total_backward_time_cloud_module, total_update_time)
    # restrictive condition
    prob += lpSum([end_vars[i] for i in End]) == total_number_samples

    prob.solve()

    print("Status:", LpStatus[prob.status])

    for v in prob.variables():
        print(v.name, "=", v.varValue)
        number_samples_end.append(v.varValue)
    return number_samples_end, value(prob.objective)


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


if __name__ == '__main__':
    max_layers = 14  # AlexNet has 14 layers in total
    data_set = make_dataset()  # Load dataset
    model = AlexNet(max_layers)
    host = ['172.18.0.10', '172.18.0.11', '172.18.0.12']
    port = ['9091', '9092', '9093']
    bandwith_device = [512*1024, 512*1024, 1024*1024]
    train_times = []
    score = []
    number_samples_end = []
    number_sample_device = []
    train_time_c = get_time(model, data_set)
    layer_bits = get_layer_output_size(model, data_set)
    layer_parameters_bits = get_model_parameters_size(model)
    total_number_samples = len(data_set)
    total_training_time = 10000
    number_samples_end=[]
    for d in range(max_layers + 1):
        for e in range(max_layers + 1):
            if d <= e:  # To avoid duplicate models and ensure m <= e
                send_model_layers(d, host[0], port[0])
                send_model_layers(d, host[1], port[1])
                send_model_layers(d, host[2], port[2])
                send_model_layers(e, '172.18.0.3','8081')
                train_times[0], score[0] = receive_info(host[0], port[0])
                train_times[1], score[1] = receive_info(host[1], port[1])
                train_times[2], score[2] = receive_info(host[2], port[2])
                train_times_e, score_e = receive_info('172.18.0.3','8081')
                number_samples_end_current, total_training_time_current = total_training_time_one_iteration(d, e, max_layers, train_times, train_times_e, train_time_c, score, layer_bits, bandwith_device, layer_parameters_bits, total_number_samples)
                if total_training_time_current < total_training_time:
                    number_samples_end = number_samples_end_current
                    md = d
                    me = e
    number_sample_device = data_parallel(number_samples_end[0], score, 3)
    send_number_sample(0, number_sample_device[0], host[0], port[0])
    send_number_sample(number_sample_device[0], number_sample_device[0] + number_sample_device[1], host[1], port[1])
    send_number_sample(number_sample_device[0] + number_sample_device[1], number_samples_end[0], host[2], port[2])
    send_model_layers(d, '172.18.0.3', '8081')
    send_number_sample(number_samples_end[0], number_samples_end[0] + number_samples_end[1], '172.18.0.3', '8081')
    data_set_cloud = data_set.skip(number_samples_end[0] + number_samples_end[1])
    train_model_stop(model, data_set_cloud, md)
    weights = model.get_weights()
    weights_d=[]
    weights_d[0] = receive_info(host[0], port[0])
    weights_d[1] = receive_info(host[1], port[1])
    weights_d[2] = receive_info(host[2], port[2])
    for i in range(md):
        weights[i] = (weights_d[0][i] + weights_d[1][i] + weights_d[2][i]) / 3  # 简单平均聚合
    model.set_weights(weights)
    train_model_stop(model, data_set_cloud, me)
    weights = model.get_weights()
    weights_e = receive_info('172.18.0.3','8081')
    for i in range(me):
        weights[i] = (weights_e[i] + weights[i]) / 2  # 简单平均聚合
    model.set_weights(weights)
    train_model_stop(model,data_set_cloud,max_layers)

                # model_d = AlexNet(d)
                # model_e = AlexNet(e)
                # model_full = AlexNet(max_layers)  # Full model

                # print(f"Training model up to layer {d}")
                # train_and_evaluate(model_d, dataset)

                # transfer_and_aggregate_parameters(model_d, model_e)

                # print(f"Training model up to layer {e}")

                # train_and_evaluate(model_e, dataset)

                # print("Training full model")
                # train_and_evaluate(model_full, dataset)
