import socket
import numpy as np
import pickle
import tensorflow as tf
from pulp import *
from model import AlexNet
import schedulePolicies


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
        train_dataset = train_dataset.batch(128).shuffle(100)
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

    train_times = []
    train_time_c_list = []
    score = []
    number_samples_end = []
    number_sample_device = []

    for i in range(50):
        train_time_c_list.append(model.get_time(single_sample_data_set))
    train_time_c = calculate_averages(train_time_c_list)
    print(train_time_c)


    print("Single sample training time data acquisition completed for cloud servers.")
    layer_output_bits = model.get_layer_output_size(single_sample_data_set)
    layer_output_bits.insert(0, 0)
    print("layer_output_bits:", layer_output_bits)
    print("Model layers output size acquisition completed. ")

    layer_parameters_bits = model.get_model_parameters_size()
    layer_parameters_bits.insert(0, 0)
    print("layer_parameters_bits:", layer_parameters_bits)
    print("Model parameters size acquisition completed.")

    data = receive_info('172.18.0.2', 8080)
    train_times.insert(data[2], data[0])
    score.insert(data[2], data[1])
    print('接收成功1个')
    data = receive_info('172.18.0.2', 8080)
    train_times.insert(data[2], data[0])
    score.insert(data[2], data[1])
    print('接收成功2个')
    data = receive_info('172.18.0.2', 8080)
    train_times.insert(data[2], data[0])
    score.insert(data[2], data[1])
    print('接收成功3个')
    print(train_times, score)
    train_times_e = receive_edge_server_info('172.18.0.2', 8080)
    print("train_time_c:", train_time_c)
    print("train_times_e", train_times_e)
    print(
        "Single sample training time and performance scores per DNN layer for edge devices and edge servers received.")


    # train_times = [{'forward': [0.005578203201293945, 0.0007617998123168945, 0.0032428550720214845, 0.0015528154373168944, 0.002669386863708496, 0.0020356512069702147, 0.0018107175827026367, 0.0013807296752929687, 0.00026689529418945314, 0.008571276664733887, 0.0003960752487182617, 0.00879286766052246, 0.000276484489440918, 0.001729569435119629], 'backward': [0.0696021032333374, 0.0, 0.07464311122894288, 0.0, 0.08056345939636231, 0.0848447036743164, 0.08316206455230712, 0.0, 0.0, 0.08360941410064697, 0.0, 0.09703088283538819, 0.0, 0.09438573360443116], 'update': [0.005001368522644043, 0.0, 0.00806269645690918, 0.0, 0.010129294395446777, 0.010294151306152344, 0.053738794326782226, 0.0, 0.0, 0.63262122631073, 0.0, 0.2655773639678955, 0.0, 0.004683270454406739]},
    #                {'forward': [0.0051999521255493165, 0.0008099174499511719, 0.004597649574279785, 0.0006011533737182617, 0.0026715946197509765, 0.003653550148010254, 0.001643695831298828, 0.0005362033843994141, 0.0002503681182861328, 0.010943403244018554, 0.00144744873046875, 0.008214254379272461, 0.00028135299682617186, 0.000478358268737793], 'backward': [0.09518705368041992, 0.0, 0.10260601997375489, 0.0, 0.09941388607025146, 0.10588201045989991, 0.10696715831756592, 0.0, 0.0, 0.09730989933013916, 0.0, 0.1244185733795166, 0.0, 0.12279264450073242], 'update': [0.005068697929382324, 0.0, 0.011739540100097656, 0.0, 0.009766111373901367, 0.017915377616882323, 0.057077579498291016, 0.0, 0.0, 0.8234802198410034, 0.0, 0.3266575765609741, 0.0, 0.0032807064056396486]},
    #                {'forward': [0.007358236312866211, 0.0007529735565185546, 0.005754404067993164, 0.0005962419509887696, 0.0029455852508544923, 0.004291768074035644, 0.005867347717285156, 0.0005568790435791015, 0.0002601003646850586, 0.017388038635253907, 0.00040820121765136716, 0.02470158100128174, 0.00030015945434570313, 0.0004939794540405274], 'backward': [0.14070546627044678, 0.0, 0.16953875064849855, 0.0, 0.1396746301651001, 0.1444025754928589, 0.15117568492889405, 0.0, 0.0, 0.1517722225189209, 0.0, 0.15397511005401612, 0.0, 0.1791773843765259], 'update': [0.003875265121459961, 0.0, 0.018239083290100096, 0.0, 0.024341130256652833, 0.025681052207946777, 0.06559869766235352, 0.0, 0.0, 1.2530751037597656, 0.0, 0.5120708847045898, 0.0, 0.0050767135620117185]}]
    # score = [3.4, 2.6999999999999997, 2]
    # train_time_c = {'forward': [0.0022550439834594727, 0.00046624183654785155, 0.0028285884857177732, 0.00043401241302490237, 0.0018346691131591797, 0.001876382827758789, 0.0016598796844482422, 0.00039066314697265624, 0.00032468795776367186, 0.004253525733947754, 4.031181335449219e-05, 0.0021938514709472657, 8.273601531982421e-05, 0.0006422853469848633],
    #                 'backward': [0.03926238059997558, 0.0, 0.03796694278717041, 0.0, 0.03905566692352295, 0.0390348482131958, 0.039145498275756835, 0.0, 0.0, 0.03891606330871582, 0.0, 0.038100900650024416, 0.0, 0.03813835144042969],
    #                 'update': [0.005943922996520996, 0.0, 0.009044961929321289, 0.0, 0.009240655899047852, 0.009940242767333985, 0.009335799217224121, 0.0, 0.0, 0.11747444629669189, 0.0, 0.05629349708557129, 0.0, 0.005795726776123047]}
    # #train_time_c_gpu = {'forward': [0.009280509948730468, 0.00039058685302734375, 0.0012915182113647462, 0.00012657642364501953, 0.0005895328521728515, 0.0006817293167114258, 0.00046671390533447265, 0.00015988826751708984, 0.00020503520965576172, 0.0005442380905151367, 0.00010100841522216797, 0.00027742862701416013, 0.00014592647552490233, 0.0003923797607421875], 'backward': [0.007177262306213379, 0.0, 0.003665165901184082, 0.0, 0.0036067962646484375, 0.003592386245727539, 0.0034822511672973635, 0.0, 0.0, 0.0038079833984375, 0.0, 0.0037491416931152346, 0.0, 0.0035330009460449218], 'update': [0.003529825210571289, 0.0, 0.003308258056640625, 0.0, 0.0030957841873168945, 0.0031998586654663087, 0.003224196434020996, 0.0, 0.0, 0.003302035331726074, 0.0, 0.0031706762313842775, 0.0, 0.0032815456390380858]}
    #
    # train_times_e = {'forward': [0.002325572967529297, 0.0011096429824829102, 0.0023488998413085938, 0.0006024408340454102, 0.0027244424819946288, 0.001961636543273926, 0.0017951202392578124, 0.0005308294296264649, 0.00025977611541748045, 0.006167521476745605, 0.000351872444152832, 0.002673211097717285, 0.00027735710144042967, 0.00043271064758300783],
    #                  'backward': [0.04019423484802246, 0.0, 0.03780625343322754, 0.0, 0.03774367809295654, 0.03827947616577149, 0.0381923770904541, 0.0, 0.0, 0.03881290912628174, 0.0, 0.07105258464813233, 0.0, 0.060614280700683594],
    #                  'update': [0.003953075408935547, 0.0, 0.008085980415344238, 0.0, 0.008262710571289062, 0.00893993377685547, 0.009440712928771973, 0.0, 0.0, 0.2975821113586426, 0.0, 0.12006674766540527, 0.0, 0.004255151748657227]}
    # train_times_e_2 = {'forward': [0.00607752799987793, 0.0007113361358642578, 0.007185091972351074, 0.0006056451797485352, 0.011363263130187989, 0.005613551139831543, 0.004076752662658691, 0.0005869388580322266, 0.0002862834930419922, 0.013560690879821778, 0.00033905029296875, 0.01927560329437256, 0.0003166818618774414, 0.0005123138427734375], 'backward': [0.1514293384552002, 0.0, 0.14870049476623534, 0.0, 0.15068767547607423, 0.13885716438293458, 0.14417964458465576, 0.0, 0.0, 0.15811503410339356, 0.0, 0.1527001142501831, 0.0, 0.16206525325775145], 'update': [0.005087413787841797, 0.0, 0.01458202362060547, 0.0, 0.021660313606262208, 0.03075451374053955, 0.014729857444763184, 0.0, 0.0, 1.2992603778839111, 0.0, 0.5213971662521363, 0.0, 0.00467437744140625]}
    # train_times_e_4 = {'forward': [0.003454599380493164, 0.0007144403457641601, 0.004257502555847168, 0.0006154727935791016, 0.0025711727142333986, 0.0036083650588989258, 0.0027010965347290038, 0.0006059360504150391, 0.00026060104370117187, 0.00999380588531494, 0.00037233829498291017, 0.008473434448242188, 0.0010816478729248047, 0.0007520103454589843], 'backward': [0.06851108551025391, 0.0, 0.08089652061462402, 0.0, 0.08085945129394531, 0.08231288433074951, 0.08048355102539062, 0.0, 0.0, 0.0806512975692749, 0.0, 0.09603053569793701, 0.0, 0.09939339637756348], 'update': [0.003733921051025391, 0.0, 0.008748226165771485, 0.0, 0.008335285186767578, 0.01196746826171875, 0.013296799659729004, 0.0, 0.0, 0.6307250881195068, 0.0, 0.25768168926239016, 0.0, 0.0034594964981079102]}
    # train_times_e_6 = {'forward': [0.0025041007995605467, 0.0007089662551879883, 0.002982921600341797, 0.0005990171432495117, 0.0029000234603881836, 0.002555270195007324, 0.0017994356155395508, 0.0005682706832885742, 0.00026629924774169923, 0.007017927169799805, 0.00035500049591064454, 0.004693803787231446, 0.0002900075912475586, 0.0007764339447021484], 'backward': [0.04788794994354248, 0.0, 0.04769852638244629, 0.0, 0.04902707576751709, 0.05019299507141113, 0.049658255577087404, 0.0, 0.0, 0.04968822002410889, 0.0, 0.0666124963760376, 0.0, 0.07236931324005128], 'update': [0.0040090084075927734, 0.0, 0.009233694076538086, 0.0, 0.009720392227172851, 0.009493436813354492, 0.011129212379455567, 0.0, 0.0, 0.4134153938293457, 0.0, 0.16866907119750976, 0.0, 0.0035370969772338867]}
    # train_times_e_list = [train_times_e_2, train_times_e_4, train_times_e_6, train_times_e]
    schedulePolicies.dec_process_md_me(max_layers,  train_times, train_times_e, train_time_c, score, layer_output_bits, layer_parameters_bits, 128)
    schedulePolicies.dec_process_bandwidth(max_layers,  train_times, train_times_e, train_time_c, score, layer_output_bits, layer_parameters_bits, 128)
    #schedulePolicies.dec_process_edge_cpus(max_layers,  train_times, train_times_e_list, train_time_c, score, layer_output_bits, layer_parameters_bits, 128 )







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
