import time as tm
import pickle
import tensorflow as tf
import pynvml
import psutil
import socket
from pulp import *
from model import AlexNet
import numpy as np
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
# def make_dataset(single_sample=False):
#     # Load the CIFAR-10 dataset
#     (train_images, train_labels), _ = tf.keras.datasets.cifar10.load_data()
#     train_dataset = tf.data.Dataset.from_tensor_slices((train_images, train_labels))
#     train_dataset = train_dataset.map(preprocess_image)
#     if not single_sample:
#         train_dataset = train_dataset.batch(128).shuffle(1000)
#     return train_dataset
# /app/data/cifar-10-batches-py/
def make_dataset(single_sample=False, data_dir="cifar-10-batches-py/"):
    train_images, train_labels = load_cifar10_data(data_dir)
    train_dataset = tf.data.Dataset.from_tensor_slices((train_images, train_labels))
    train_dataset = train_dataset.map(preprocess_image)
    if not single_sample:
        train_dataset = train_dataset.batch(128).shuffle(1000)
    return train_dataset



# send and receive data
def receive_info(host, port):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        print(host, port)
        s.bind((host, port))
        s.listen()
        while True:
            conn, addr = s.accept()
            with conn:
                print(f"Connected by {addr}")
                while True:
                    try:
                        packet = conn.recv(4096)
                        if not packet:
                            break
                        data = packet
                    except socket.timeout:
                        print("Connection timed out while receiving data")
                        break
                if data:
                    data, score, device = pickle.loads(data)
                    print(data, score, device)
                    return data, score, device
    return None

def receive_edge_server_info(host, port):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        print(host, port)
        s.bind((host, port))
        s.listen()
        while True:
            conn, addr = s.accept()
            with conn:
                print(f"Connected by {addr}")
                while True:
                    try:
                        packet = conn.recv(4096)
                        if not packet:
                            break
                        data = packet
                    except socket.timeout:
                        print("Connection timed out while receiving data")
                        break
                if data:
                    data = pickle.loads(data)
                    print(data)
                    return data
    return None
def send_model_layers(layers, host, port):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.connect((host, port))
        s.sendall(layers)
def send_number_sample(number_sample_start, number_sample_end, host, port):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.connect((host, port))
        s.sendall(number_sample_start, number_sample_end)


# Get three training time by one iteration
def get_time(alexNet, data_set):
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
        for layer in alexNet.model.layers:
            start_time = tm.time()
            x = layer(x)
            intermediates.append(x)
            training_times['forward'].append(tm.time() - start_time)
        loss = tf.reduce_mean(tf.losses.categorical_crossentropy(label, x, from_logits=False))

    for idx, layer in enumerate(alexNet.model.layers):
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
    for layer in model.model.layers:
        if len(layer.weights) > 0:  # 如果层有参数
            total_bits = 0
            for w in layer.weights:
                # 参数的总数乘以每个参数的比特数（假设是32位浮点数）
                total_bits += tf.size(w).numpy() * 32
            layer_parameter_bits.append(total_bits)
        else:
            layer_parameter_bits.append(0)
    return layer_parameter_bits
def get_layer_output_size(model, data_set):
    iterator = iter(data_set)
    image, label = next(iterator)
    # Add Batch Dimension，because the model expects four-dimensional inputs (batch_size, height, width, channels)
    image = tf.expand_dims(image, axis=0)
    dtype_size = np.dtype(tf.float32.as_numpy_dtype).itemsize * 8
    layer_output_sizes = []
    x = image
    for layer in model.model.layers:
        x = layer(x)
        # 获取当前层输出张量的形状
        output_shape = x.shape
        # 计算输出张量的元素个数
        num_elements = np.prod(output_shape)
        # 计算输出张量的总大小（以比特为单位）
        output_size_bits = num_elements * dtype_size
        layer_output_sizes.append(output_size_bits)

    return layer_output_sizes
def data_parallel(end_vars,score, m):
    total_score_device = 0
    number_sample_device = []
    print(score)
    for i in range(m):
        total_score_device += score[i]
    for i in range(m):
        number_sample_device.append((score[i] / total_score_device) * end_vars['device'])
    return number_sample_device
def total_training_time_one_iteration( md, me, max_layers, train_times, train_times_e, train_times_c, score, layer_bits, bandwith_device, layer_parameters_bits,total_number_samples ):
    number_samples_end=[]
    End = ['device', 'edge', 'cloud']
    prob = LpProblem("The total training time Problem", LpMinimize)
    end_vars = LpVariable.dicts('End', End ,0)
    print(end_vars)
    print("score:",score)
    number_sample_device = data_parallel(end_vars, score, 3)
    total_forward_time_device_d = [0, 0, 0]
    total_forward_time_edge_d = 0
    total_forward_time_cloud_d = 0
    for j in range(3):
        for i in range(md):
            total_forward_time_device_d[j] += number_sample_device[j] * train_times[j]['forward'][i]
        total_forward_time_device_d[j] += (layer_bits[md] * number_sample_device[j]) / bandwith_device[j]
        print(layer_bits[md])
        print("device",j," to edge:", (layer_bits[md] * number_sample_device[j]) / bandwith_device[j])
    for i in range(md):
        total_forward_time_edge_d += end_vars['edge'] * train_times_e['forward'][i]
        total_forward_time_cloud_d += end_vars['cloud'] * train_times_c['forward'][i]

    total_backward_time_device_d = [0, 0, 0]
    total_backward_time_edge_d = 0
    total_backward_time_cloud_d = 0
    for j in range(3):
        for i in range(md):
            total_backward_time_device_d[j] += number_sample_device[j] * train_times[j]['backward'][i]
        total_backward_time_device_d[j] += ((layer_bits[md ] * number_sample_device[j]) / bandwith_device[j])
    for i in range(md):
        total_backward_time_edge_d += end_vars['edge'] * train_times_e['backward'][i]
        total_backward_time_cloud_d += end_vars['cloud'] * train_times_c['backward'][i]

    max_total_forward_time_device_d = LpVariable("MaxTotalForwardTimeDeviceD", 0)
    max_total_backward_time_device_d = LpVariable("MaxTotalBackwardTimeDeviceD", 0)
    for var in total_forward_time_device_d:
        prob += max_total_forward_time_device_d >= var
    for var in total_backward_time_device_d:
        prob += max_total_backward_time_device_d >= var
    max_total_forward_time_device_module = LpVariable("MaxTotalForwardTimeDeviceModule", 0)
    max_total_backward_time_device_module = LpVariable("MaxTotalBackwardTimeDeviceModule", 0)
    prob += max_total_forward_time_device_module >= max_total_forward_time_device_d
    prob += max_total_forward_time_device_module >= total_forward_time_edge_d
    prob += max_total_forward_time_device_module >= total_forward_time_cloud_d
    prob += max_total_backward_time_device_module >= max_total_backward_time_device_d
    prob += max_total_backward_time_device_module >= total_backward_time_edge_d
    prob += max_total_backward_time_device_module >= total_backward_time_cloud_d


    total_forward_time_edge_e = 0
    total_forward_time_cloud_e = 0
    total_backward_time_edge_e = 0
    total_backward_time_cloud_e = 0
    for i in range(md, me):
        total_forward_time_edge_e += end_vars['edge'] * train_times_e['forward'][i]
        total_forward_time_cloud_e += end_vars['cloud'] * train_times_c['forward'][i]
        total_backward_time_edge_e += end_vars['edge'] * train_times_e['backward'][i]
        total_backward_time_cloud_e += end_vars['cloud'] * train_times_c['backward'][i]
    total_forward_time_edge_e += (layer_bits[me] * end_vars['edge']) / (1024 * 1024)
    total_backward_time_edge_e += (layer_bits[me] * end_vars['edge']) / (1024 * 1024)
    max_total_forward_time_edge_module = LpVariable("MaxTotalForwardTimeEdgeModule", 0)
    max_total_backward_time_edge_module = LpVariable("MaxTotalBackwardTimeEdgeModule", 0)
    prob += max_total_forward_time_edge_module >= total_forward_time_edge_e
    prob += max_total_forward_time_edge_module >= total_forward_time_cloud_e
    prob += max_total_backward_time_edge_module >= total_backward_time_edge_e
    prob += max_total_backward_time_edge_module >= total_backward_time_cloud_e


    total_forward_time_cloud_module = 0
    total_backward_time_cloud_module = 0
    for i in range(me, max_layers):
        total_forward_time_cloud_module += end_vars['cloud'] * train_times_c['forward'][i]
        total_backward_time_cloud_module += end_vars['cloud'] * train_times_c['backward'][i]

    total_update_time_cloud = 0
    total_update_time_edge = 0
    total_update_time_device = [0, 0, 0]
    parameters_to_edge = 0
    parameters_to_device = 0
    for i in range(max_layers):
        total_update_time_cloud += train_times_c['update'][i]
    for i in range(me):
        total_update_time_edge += train_times_e['update'][i]
        parameters_to_edge += layer_parameters_bits[i]
    for j in range(3):
        for i in range(md):
            total_update_time_device[j] = train_times[j]['update'][i]
    for i in range(md):
        parameters_to_device += layer_parameters_bits[i]
    max_total_update_time_device = LpVariable("MaxTotalUpdateTimeDevice", 0)
    for var in total_update_time_device:
        prob += max_total_update_time_device>= var
    max_parameter_to_time = LpVariable("MaxParameterToTime", 0)
    prob += max_parameter_to_time >= parameters_to_edge / (500*1024 * 1024)
    prob += max_parameter_to_time >= parameters_to_device / (100*1024 * 1024)

    total_update_time = LpVariable("TotalUpdateTime", 0)
    prob += total_update_time >= total_update_time_cloud
    prob += total_update_time >= total_update_time_edge
    prob += total_update_time >= max_total_update_time_device+max_parameter_to_time

    # target equation
    prob += lpSum([max_total_forward_time_device_module, max_total_backward_time_device_module, max_total_forward_time_edge_module, max_total_backward_time_edge_module, total_forward_time_cloud_module, total_backward_time_cloud_module, total_update_time])
    # restrictive condition
    prob += lpSum([end_vars[i] for i in End]) == total_number_samples
    prob += end_vars['device'] >= 0
    prob += end_vars['device'] <= md*total_number_samples
    prob += end_vars['edge'] >= 0
    prob += end_vars['edge'] <= me*total_number_samples

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
    single_sample_data_set = make_dataset(single_sample=True)
    print("Dataset loaded.")
    model = AlexNet(max_layers)
    print("Model loaded.")

    host = ['172.18.0.10', '172.18.0.11', '172.18.0.12']
    port = [9090, 9091, 9092]
    bandwith_device = [100*1024*1024, 100*1024*1024, 200*1024*1024]
    train_times = []
    score = []
    number_samples_end = []
    number_sample_device = []
    #train_time_c = get_time(model, single_sample_data_set)
    print("Single sample training time data acquisition completed for cloud servers.")
    layer_output_bits = get_layer_output_size(model, single_sample_data_set)
    layer_output_bits.insert(0,0)
    print("layer_output_bits:", layer_output_bits)
    print("Model layers output size acquisition completed. ")
    layer_parameters_bits = get_model_parameters_size(model)
    layer_parameters_bits.insert(0, 0)
    print("layer_parameters_bits:", layer_parameters_bits)
    print("Model parameters size acquisition completed.")

    total_number_samples = 128

    total_time = 10000
    print("Start enumerating split points. ")
    for d in range(max_layers+1 ):
        for e in range(max_layers +1):
            if d <= e:  # To avoid duplicate models and ensure m <= e
                print("--------------------------------------------------------------------------------")
                print("d:",d," e:",e)
                # send_model_layers(d, host[0], port[0])
                # send_model_layers(d, host[1], port[1])
                # send_model_layers(d, host[2], port[2])
                # send_model_layers(e, '172.18.0.3','8081')
                # print("Sending Segmentation Points to Edge Devices and Edge Servers.")

                # data = receive_info('172.18.0.2', 8080)
                # train_times.insert(data[2], data[0])
                # score.insert(data[2], data[1])
                # print('接收成功1个')
                # data = receive_info('172.18.0.2', 8080)
                # train_times.insert(data[2], data[0])
                # score.insert(data[2], data[1])
                # print('接收成功2个')
                # data = receive_info('172.18.0.2', 8080)
                # train_times.insert(data[2], data[0])
                # score.insert(data[2], data[1])
                # print('接收成功3个')
                # print(train_times,score)
                # train_times_e = receive_edge_server_info('172.18.0.2', 8080)
                # print(train_times_e)
                # print("Single sample training time and performance scores per DNN layer for edge devices and edge servers received.")
                train_times= [{'forward': [0.07367324829101562, 0.013790369033813477, 0.027213096618652344, 0.0018939971923828125, 0.015223026275634766, 0.016123294830322266, 0.015929222106933594, 0.001386404037475586, 0.004564523696899414, 0.9917354583740234, 0.005135774612426758, 0.17138338088989258, 0.0005443096160888672, 0.012929439544677734],
                                     'backward': [0.5779709815979004, 0, 0.1416771411895752, 0, 0.11222529411315918, 0.14815998077392578, 0.1245572566986084, 0, 0, 0.12415933609008789, 0, 0.1022031307220459, 0, 0.14038681983947754],
                                     'update': [0.05483675003051758, 0, 0.023286104202270508, 0, 0.0197906494140625, 0.02234649658203125, 0.0201570987701416, 0, 0, 1.3814826011657715, 0, 0.3044900894165039, 0, 0.0075855255126953125]},
                                    {'forward': [0.02651190757751465, 0.003038167953491211, 0.012700319290161133, 0.0009279251098632812, 0.010939359664916992, 0.011154651641845703, 0.01052713394165039, 0.0012743473052978516, 0.004322052001953125, 0.5074286460876465, 0.0044345855712890625, 0.27574682235717773, 0.0011761188507080078, 0.022207975387573242],
                                     'backward': [0.38134098052978516, 0, 0.17919611930847168, 0, 0.20291781425476074, 0.24898266792297363, 0.2745399475097656, 0, 0, 0.2650926113128662, 0, 0.3170645236968994, 0, 0.29589056968688965],
                                     'update': [0.0325162410736084, 0, 0.08191585540771484, 0, 0.031008005142211914, 0.03806257247924805, 0.03981184959411621, 0, 0, 2.284118175506592, 0, 0.49225544929504395, 0, 0.01584792137145996]},
                                    {'forward': [0.038259029388427734, 0.0037932395935058594, 0.017202377319335938, 0.0014278888702392578, 0.015506982803344727, 0.01452779769897461, 0.04888153076171875, 0.001316070556640625, 0.0045719146728515625, 0.7887036800384521, 0.006239652633666992, 0.10798430442810059, 0.0005450248718261719, 0.009839296340942383],
                                     'backward': [0.8009474277496338, 0, 0.4717066287994385, 0, 0.2610781192779541, 0.2759387493133545, 0.19504976272583008, 0, 0, 0.23992228507995605, 0, 0.3194422721862793, 0, 0.2980329990386963],
                                     'update': [0.023236989974975586, 0, 0.02462315559387207, 0, 0.0216524600982666, 0.030149459838867188, 0.05676627159118652, 0, 0, 2.764164447784424, 0, 0.6943049430847168, 0, 0.011168479919433594]}]
                score = [3.6999999999999997, (2*0.7)+(3*0.3),  (2*0.7)+(3*0.3)]

                train_time_c = {'forward': [0.022725820541381836, 0.002012491226196289, 0.010140419006347656, 0.0009963512420654297,
                             0.008079051971435547, 0.006978034973144531, 0.005948781967163086, 0.001049041748046875,
                             0.0, 0.03751373291015625, 0.0, 0.016866683959960938, 0.0, 0.0070264339447021484],
                 'backward': [0.06008648872375488, 0, 0.040029287338256836, 0, 0.03848981857299805, 0.03903555870056152,
                              0.03855752944946289, 0, 0, 0.040686845779418945, 0, 0.03847002983093262, 0,
                              0.03724050521850586],
                 'update': [0.0161590576171875, 0, 0.014053106307983398, 0, 0.014043092727661133, 0.013985872268676758,
                            0.013251781463623047, 0, 0, 0.13432598114013672, 0, 0.06319355964660645, 0,
                            0.010152101516723633]}
                train_times_e ={'forward': [0.0251920223236084, 0.0025517940521240234, 0.015084505081176758, 0.00151824951171875, 0.015735149383544922, 0.010651826858520508, 0.010903358459472656, 0.0021288394927978516, 0.00421452522277832, 0.11026287078857422, 0.0033779144287109375, 0.08013916015625, 0.0010640621185302734, 0.010649442672729492],
                                'backward': [0.17377114295959473, 0, 0.07596111297607422, 0, 0.07927060127258301, 0.07131075859069824, 0.06863188743591309, 0, 0, 0.07320070266723633, 0, 0.09154963493347168, 0, 0.07548689842224121],
                                'update': [0.0183868408203125, 0, 0.01685190200805664, 0, 0.025641918182373047, 0.019095659255981445, 0.017769575119018555, 0, 0, 0.9297165870666504, 0, 0.19216704368591309, 0, 0.007329463958740234]}
                number_samples_end_current, total_training_time_current = total_training_time_one_iteration(d, e, max_layers, train_times, train_times_e, train_time_c, score, layer_output_bits, bandwith_device, layer_parameters_bits, total_number_samples)
                print("Current data sample split points and total time spent are calculated.")
                if total_training_time_current < total_time:
                    total_time = total_training_time_current
                    number_samples_end = number_samples_end_current
                    md = d
                    me = e
                    print("Less time spent updating split points and total time spent")
    print("----------------------------------")
    print("最少时间：", total_time )
    print("数据分割：", number_samples_end)
    print("md:", md)
    print("me", me)
    # number_sample_device = data_parallel(number_samples_end[0], score, 3)
    # send_number_sample(0, number_sample_device[0], host[0], port[0])
    # send_number_sample(number_sample_device[0], number_sample_device[0] + number_sample_device[1], host[1], port[1])
    # send_number_sample(number_sample_device[0] + number_sample_device[1], number_samples_end[0], host[2], port[2])
    # send_model_layers(d, '172.18.0.3', '8081')
    # send_number_sample(number_samples_end[0], number_samples_end[0] + number_samples_end[1], '172.18.0.3', '8081')
    # data_set_cloud = data_set.skip(number_samples_end[0] + number_samples_end[1])
    # train_model_stop(model, data_set_cloud, md)
    # weights = model.get_weights()
    # weights_d=[]
    # weights_d[0] = receive_info(host[0], port[0])
    # weights_d[1] = receive_info(host[1], port[1])
    # weights_d[2] = receive_info(host[2], port[2])
    # for i in range(md):
    #     weights[i] = (weights_d[0][i] + weights_d[1][i] + weights_d[2][i]) / 3  # 简单平均聚合
    # model.set_weights(weights)
    # train_model_stop(model, data_set_cloud, me)
    # weights = model.get_weights()
    # weights_e = receive_info('172.18.0.3','8081')
    # for i in range(me):
    #     weights[i] = (weights_e[i] + weights[i]) / 2  # 简单平均聚合
    # model.set_weights(weights)
    # train_model_stop(model,data_set_cloud,max_layers)

