import tensorflow as tf
from figure import *
from pulp import *

def data_parallel(end_vars, score, m):
    total_score_device = 0
    number_sample_device = []
    for i in range(m):
        total_score_device += score[i]
    for i in range(m):
        number_sample_device.append((score[i] / total_score_device) * end_vars['device'])
    return number_sample_device



def dec_process_bandwidth(max_layers, train_times, train_times_e, train_time_c, score, layer_output_bits, layer_parameters_bits, total_number_samples):
    different_bandwidth = [[100 * 1000 * 1024, 100 * 1000 * 1024, 100 * 1000 * 1024, 20 * 1000 * 1024],
                           [100 * 1000 * 1024, 100 * 1000 * 1024, 100 * 1000 * 1024, 30 * 1000 * 1024],
                           [100 * 1000 * 1024, 100 * 1000 * 1024, 100 * 1000 * 1024, 40 * 1000 * 1024],
                           [100 * 1000 * 1024, 100 * 1000 * 1024, 100 * 1000 * 1024, 50 * 1000 * 1024],
                           [100 * 1000 * 1024, 100 * 1000 * 1024, 100 * 1000 * 1024, 60 * 1000 * 1024],
                           [100 * 1000 * 1024, 100 * 1000 * 1024, 100 * 1000 * 1024, 70 * 1000 * 1024],
                           [100 * 1000 * 1024, 100 * 1000 * 1024, 100 * 1000 * 1024, 80 * 1000 * 1024],
                           [100 * 1000 * 1024, 100 * 1000 * 1024, 100 * 1000 * 1024, 90 * 1000 * 1024],
                           [100 * 1000 * 1024, 100 * 1000 * 1024, 100 * 1000 * 1024, 100 * 1000 * 1024]]
    total_time = 10000
    min_total_time_bandwidth = []
    md_bandwidth = []
    me_bandwidth = []
    point_bandwidth = []

    total_time_hiertrain = 10000
    min_total_time_bandwidth_hiertrain = []
    md_bandwidth_hiertrain = []
    me_bandwidth_hiertrain = []
    point_bandwidth_hiertrain = []

    min_total_time_all_cloud = []
    min_total_time_all_edge = []
    min_total_time_Feel = []
    print("Start enumerating split points. ")
    for bandwidth in different_bandwidth:
        for d in range(max_layers + 1):
            min_total_time_d = 1000
            min_total_time_d_hiertrain = 1000
            for e in range(max_layers + 1):
                if d <= e:  # To avoid duplicate models and ensure m <= e
                    print("--------------------------------------------------------------------------------")
                    print("d:", d, " e:", e)


                    var_current, total_training_time_current = total_training_time_one_iteration(d, e, max_layers,
                                                                                                 train_times,
                                                                                                 train_times_e,
                                                                                                 train_time_c, score,
                                                                                                 layer_output_bits,
                                                                                                 bandwidth,
                                                                                                 layer_parameters_bits,
                                                                                                 total_number_samples)
                    hiertrain_var_current, hiertrain_total_training_time_current = hierTrain(d, e, max_layers,
                                                                                             train_times[2],
                                                                                             train_times_e,
                                                                                             train_time_c,
                                                                                             layer_output_bits,
                                                                                             bandwidth,
                                                                                             layer_parameters_bits,
                                                                                             total_number_samples)

                    print("Current data sample split points and total time spent are calculated.")
                    if total_training_time_current < total_time:
                        total_time = total_training_time_current
                        number_samples_end = var_current[:3]
                        md = d
                        me = e
                        print("Less time spent updating split points and total time spent")
                    if hiertrain_total_training_time_current < total_time_hiertrain:
                        total_time_hiertrain = hiertrain_total_training_time_current
                        number_samples_end_hiertrain = hiertrain_var_current[:3]
                        md_hiertrain = d
                        me_hiertrain = e
        min_total_time_bandwidth.append(total_time)
        md_bandwidth.append(md)
        me_bandwidth.append(me)
        point_bandwidth.append(number_samples_end)

        min_total_time_bandwidth_hiertrain.append(total_time_hiertrain)
        md_bandwidth_hiertrain.append(md_hiertrain)
        me_bandwidth_hiertrain.append(me_hiertrain)
        point_bandwidth_hiertrain.append(number_samples_end_hiertrain)

        min_total_time_all_cloud.append(total_training_time_one_iteration_all_cloud(train_time_c,128, layer_parameters_bits, bandwidth))
        min_total_time_all_edge.append(total_training_time_one_iteration_all_edge(train_times_e, 128, layer_parameters_bits, bandwidth))
        min_total_time_Feel.append(train_FEEL(14, train_times, layer_output_bits, bandwidth, 128, score, layer_parameters_bits))
    print(min_total_time_bandwidth)
    print(md_bandwidth)
    print(me_bandwidth)
    print(point_bandwidth)

    print(min_total_time_bandwidth_hiertrain)
    print(md_bandwidth_hiertrain)
    print(me_bandwidth_hiertrain)
    print(point_bandwidth_hiertrain)
    figure_dec_all_cloud_all_edge_bandwidth(min_total_time_bandwidth, min_total_time_all_cloud,min_total_time_all_edge,[ 20, 30, 40, 50, 60, 70, 80, 90, 100])
    figure_hiertrain_dec_FEEL(min_total_time_bandwidth_hiertrain, min_total_time_bandwidth, min_total_time_Feel, [ 20, 30, 40, 50, 60, 70, 80, 90, 100])
def dec_process_md_me(max_layers, train_times, train_times_e, train_time_c, score, layer_output_bits, layer_parameters_bits, total_number_samples):
    bandwidth = [100 * 1000 * 1024, 100 * 1000 * 1024, 100 * 1000 * 1024, 50 * 1000 * 1024]
    total_time = 10000
    print("Start enumerating split points. ")
    time_for_figure = []
    point_figure = []
    min_total_time_md = []
    min_total_time_me = [1000]*15
    for d in range(max_layers + 1):
        min_total_time_d = 1000
        for e in range(max_layers + 1):
            if d <= e:  # To avoid duplicate models and ensure m <= e
                print("--------------------------------------------------------------------------------")
                print("d:", d, " e:", e)

                var_current, total_training_time_current = total_training_time_one_iteration(d, e, max_layers,
                                                                                             train_times,
                                                                                             train_times_e,
                                                                                             train_time_c, score,
                                                                                             layer_output_bits,
                                                                                             bandwidth,
                                                                                             layer_parameters_bits,
                                                                                             total_number_samples)

                time_for_figure.append(total_training_time_current)
                point_figure.append(var_current)

                print("Current data sample split points and total time spent are calculated.")
                if total_training_time_current < total_time:
                    total_time = total_training_time_current
                    number_samples_end = var_current[:3]
                    md = d
                    me = e
                    print("Less time spent updating split points and total time spent")

                if total_training_time_current < min_total_time_d:
                    min_total_time_d = total_training_time_current

                if total_training_time_current < min_total_time_me[e]:
                    min_total_time_me[e] = total_training_time_current
        min_total_time_md.append(min_total_time_d)


    print("----------------------------------")
    print("最少时间：", total_time)
    print("数据分割：", number_samples_end)
    print("md:", md)
    print("me", me)
    print(min_total_time_md)
    figure_Md(min_total_time_md)
    figure_Me(min_total_time_me)
def dec_process_edge_cpus(max_layers, train_times, train_times_e_list, train_time_c, score, layer_output_bits, layer_parameters_bits, total_number_samples):
    different_bandwidth = [[100 * 1000 * 1024, 100 * 1000 * 1024, 100 * 1000 * 1024, 20 * 1000 * 1024],
                           [100 * 1000 * 1024, 100 * 1000 * 1024, 100 * 1000 * 1024, 30 * 1000 * 1024],
                           [100 * 1000 * 1024, 100 * 1000 * 1024, 100 * 1000 * 1024, 40 * 1000 * 1024],
                           [100 * 1000 * 1024, 100 * 1000 * 1024, 100 * 1000 * 1024, 50 * 1000 * 1024],
                           [100 * 1000 * 1024, 100 * 1000 * 1024, 100 * 1000 * 1024, 60 * 1000 * 1024],
                           [100 * 1000 * 1024, 100 * 1000 * 1024, 100 * 1000 * 1024, 70 * 1000 * 1024],
                           [100 * 1000 * 1024, 100 * 1000 * 1024, 100 * 1000 * 1024, 80 * 1000 * 1024],
                           [100 * 1000 * 1024, 100 * 1000 * 1024, 100 * 1000 * 1024, 90 * 1000 * 1024],
                           [100 * 1000 * 1024, 100 * 1000 * 1024, 100 * 1000 * 1024, 100 * 1000 * 1024]]
    total_time = 10000

    min_total_time_bandwidth_list = []
    md_bandwidth = []
    me_bandwidth = []
    point_bandwidth = []

    print("Start enumerating split points. ")
    for train_times_e in train_times_e_list:
        min_total_time_bandwidth = []
        for bandwidth in different_bandwidth:
            for d in range(max_layers + 1):
                for e in range(max_layers + 1):
                    if d <= e:  # To avoid duplicate models and ensure m <= e
                        print("--------------------------------------------------------------------------------")
                        print("d:", d, " e:", e)

                        var_current, total_training_time_current = total_training_time_one_iteration(d, e, max_layers,
                                                                                                     train_times,
                                                                                                     train_times_e,
                                                                                                     train_time_c,
                                                                                                     score,
                                                                                                     layer_output_bits,
                                                                                                     bandwidth,
                                                                                                     layer_parameters_bits,
                                                                                                     total_number_samples)

                        print("Current data sample split points and total time spent are calculated.")
                        if total_training_time_current < total_time:
                            total_time = total_training_time_current
                            number_samples_end = var_current[:3]
                            md = d
                            me = e
                            print("Less time spent updating split points and total time spent")
            min_total_time_bandwidth.append(total_time)
            md_bandwidth.append(md)
            me_bandwidth.append(me)
            point_bandwidth.append(number_samples_end)
        min_total_time_bandwidth_list.append(min_total_time_bandwidth)
        print(min_total_time_bandwidth_list)
        print(md_bandwidth)
        print(me_bandwidth)
        print(point_bandwidth)




    figure_dec_edge_cpus( min_total_time_bandwidth_list,[20, 30, 40, 50, 60, 70, 80, 90, 100])
def total_training_time_one_iteration(md, me, max_layers, train_times, train_times_e, train_times_c, score, layer_bits, bandwidth, layer_parameters_bits, total_number_samples):
    number_samples_end = []
    End = ['device', 'edge', 'cloud']
    prob = LpProblem("The total training time Problem", LpMinimize)
    end_vars = LpVariable.dicts('End', End, 0)
    print("score:", score)
    number_sample_device = data_parallel(end_vars, score, 3)
    print("number_sample_device:", number_sample_device)
    total_forward_time_device_d = [0, 0, 0]
    total_forward_time_edge_d = 0
    total_forward_time_cloud_d = 0
    for j in range(3):
        for i in range(md):
            total_forward_time_device_d[j] += number_sample_device[j] * train_times[j]['forward'][i]
        total_forward_time_device_d[j] += (layer_bits[md] * number_sample_device[j]) / bandwidth[j]
    print("total_forward_time_device_d:", total_forward_time_device_d)
    for i in range(md):
        total_forward_time_edge_d += end_vars['edge'] * train_times_e['forward'][i]
        total_forward_time_cloud_d += end_vars['cloud'] * train_times_c['forward'][i]
    print("total_forward_time_edge_d:", total_forward_time_edge_d)
    print("total_forward_time_cloud_d:", total_forward_time_cloud_d)

    total_backward_time_device_d = [0, 0, 0]
    total_backward_time_edge_d = 0
    total_backward_time_cloud_d = 0
    for j in range(3):
        for i in range(md):
            total_backward_time_device_d[j] += number_sample_device[j] * train_times[j]['backward'][i]
        total_backward_time_device_d[j] += ((layer_bits[md] * number_sample_device[j]) / bandwidth[j])
    for i in range(md):
        total_backward_time_edge_d += end_vars['edge'] * train_times_e['backward'][i]
        total_backward_time_cloud_d += end_vars['cloud'] * train_times_c['backward'][i]

    max_total_forward_time_device_module = LpVariable("MaxTotalForwardTimeDeviceModule", 0)
    max_total_backward_time_device_module = LpVariable("MaxTotalBackwardTimeDeviceModule", 0)
    for var in total_forward_time_device_d:
        print(var)
        prob += max_total_forward_time_device_module >= var
    prob += max_total_forward_time_device_module >= total_forward_time_edge_d
    prob += max_total_forward_time_device_module >= total_forward_time_cloud_d
    for var in total_backward_time_device_d:
        prob += max_total_backward_time_device_module >= var
    prob += max_total_backward_time_device_module >= total_backward_time_edge_d
    prob += max_total_backward_time_device_module >= total_backward_time_cloud_d

    # edge_module
    total_forward_time_edge_e = 0
    total_forward_time_cloud_e = 0
    total_backward_time_edge_e = 0
    total_backward_time_cloud_e = 0
    for i in range(md, me):
        total_forward_time_edge_e += (end_vars['edge'] + end_vars['device']) * train_times_e['forward'][i]
        total_forward_time_cloud_e += (end_vars['cloud']) * train_times_c['forward'][i]
        total_backward_time_edge_e += (end_vars['edge'] + end_vars['device']) * train_times_e['backward'][i]
        total_backward_time_cloud_e += (end_vars['cloud']) * train_times_c['backward'][i]
    total_forward_time_edge_e += (layer_bits[me] * (end_vars['edge'] + end_vars['device'])) / bandwidth[3]
    total_backward_time_edge_e += (layer_bits[me] * (end_vars['edge'] + end_vars['device'])) / bandwidth[3]
    max_total_forward_time_edge_module = LpVariable("MaxTotalForwardTimeEdgeModule", 0)
    max_total_backward_time_edge_module = LpVariable("MaxTotalBackwardTimeEdgeModule", 0)
    prob += max_total_forward_time_edge_module >= total_forward_time_edge_e
    prob += max_total_forward_time_edge_module >= total_forward_time_cloud_e
    prob += max_total_backward_time_edge_module >= total_backward_time_edge_e
    prob += max_total_backward_time_edge_module >= total_backward_time_cloud_e

    # cloud_module
    total_forward_time_cloud_module = 0
    total_backward_time_cloud_module = 0
    for i in range(me, max_layers):
        total_forward_time_cloud_module += total_number_samples * train_times_c['forward'][i]
        total_backward_time_cloud_module += total_number_samples * train_times_c['backward'][i]
    print("total_forward_time_cloud_module:", total_forward_time_cloud_module)
    print("total_backward_time_cloud_module:", total_backward_time_cloud_module)

    # update time
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
    max_total_update_time_device = LpVariable("MaxTotalUpdateTimeDevice", 0)
    for var in total_update_time_device:
        prob += max_total_update_time_device >= var
    for i in range(md):
        parameters_to_device += layer_parameters_bits[i]

    print("parameters_to_device:", parameters_to_device)
    print("parameters_to_edge:", parameters_to_edge)
    max_parameter_to_time = LpVariable("MaxParameterToTime", 0)
    prob += max_parameter_to_time >= parameters_to_edge / bandwidth[3]
    prob += max_parameter_to_time >= parameters_to_device / bandwidth[3]

    total_update_time = LpVariable("TotalUpdateTime", 0)
    prob += total_update_time >= total_update_time_cloud
    prob += total_update_time >= total_update_time_edge
    prob += total_update_time >= max_total_update_time_device

    # target equation
    prob += lpSum([max_total_forward_time_device_module, max_total_backward_time_device_module,
                   max_total_forward_time_edge_module, max_total_backward_time_edge_module,
                   total_forward_time_cloud_module, total_backward_time_cloud_module, total_update_time+max_parameter_to_time])
    # restrictive condition
    prob += lpSum([end_vars[i] for i in End]) == total_number_samples
    prob += end_vars['device'] >= 0
    prob += end_vars['device'] <= md * total_number_samples
    prob += end_vars['edge'] >= 0
    prob += end_vars['edge'] <= me * total_number_samples

    prob.solve()

    print("Status:", LpStatus[prob.status])

    for v in prob.variables():
        print(v.name, "=", v.varValue)
        number_samples_end.append(v.varValue)
    print(value(prob.objective))
    return number_samples_end, value(prob.objective)
def total_training_time_one_iteration_all_cloud(train_time, total_number_samples, layer_parameters_bits, bandwidth):
    train_forward_time = 0
    train_backward_time = 0
    train_update_time = 0
    parameters_to_device = 0
    for i in range(14):
        train_forward_time += train_time['forward'][i] * total_number_samples
        train_backward_time += train_time['backward'][i] * total_number_samples
        train_update_time += train_time['update'][i]
        parameters_to_device += layer_parameters_bits[i]
    parameters_download_time = parameters_to_device/bandwidth[3]
    return train_update_time+train_backward_time+train_forward_time+parameters_download_time
def total_training_time_one_iteration_all_edge(train_time, total_number_samples, layer_parameters_bits, bandwidth):
    train_forward_time = 0
    train_backward_time = 0
    train_update_time = 0
    parameters_to_device = 0
    for i in range(14):
        train_forward_time += train_time['forward'][i] * total_number_samples
        train_backward_time += train_time['backward'][i] * total_number_samples
        train_update_time += train_time['update'][i]
        parameters_to_device += layer_parameters_bits[i]
    parameters_download_time = parameters_to_device / bandwidth[0]
    return train_update_time+train_backward_time+train_forward_time+parameters_download_time
def train_FEEL(max_layers, train_times, layer_bits, bandwidth, total_number_samples, score, layer_parameters_bits):
    number_sample_device = []
    total_score_device = 0
    for i in range(3):
        total_score_device += score[i]
    for i in range(3):
        number_sample_device.append((score[i] / total_score_device) * total_number_samples)

    total_forward_time_device = [0, 0, 0]
    for j in range(3):
        for i in range(max_layers):
            total_forward_time_device[j] += number_sample_device[j] * train_times[j]['forward'][i]
        total_forward_time_device[j] += (layer_bits[max_layers] * number_sample_device[j]) / bandwidth[j]
    print("total_forward_time_device_d:", total_forward_time_device)
    total_backward_time_device = [0, 0, 0]
    for j in range(3):
        for i in range(max_layers):
            total_backward_time_device[j] += number_sample_device[j] * train_times[j]['backward'][i]
        total_backward_time_device[j] += ((layer_bits[max_layers] * number_sample_device[j]) / bandwidth[j])
    total_update_time = [0, 0, 0]
    for j in range(3):
        for i in range(max_layers):
            total_update_time[j] += train_times[j]['update'][i]

    parameters_to_device = 0
    for i in range(14):
        parameters_to_device += layer_parameters_bits[i]
    parameters_download_time = parameters_to_device / bandwidth[0]

    total_time = max(total_forward_time_device) + max(total_backward_time_device) + max(total_update_time) + parameters_download_time
    return total_time
def hierTrain(md, me, max_layers, train_times_d, train_times_e, train_times_c, layer_bits, bandwidth, layer_parameters_bits, total_number_samples):
    number_samples_end = []
    End = ['device', 'edge', 'cloud']
    prob = LpProblem("The total training time Problem", LpMinimize)
    end_vars = LpVariable.dicts('End', End, 0)

    # device module
    total_forward_time_device_d = 0
    total_forward_time_edge_d = 0
    total_forward_time_cloud_d = 0
    for i in range(md):
        total_forward_time_device_d += end_vars['device'] * train_times_d['forward'][i]
        total_forward_time_edge_d += end_vars['edge'] * train_times_e['forward'][i]
        total_forward_time_cloud_d += end_vars['cloud'] * train_times_c['forward'][i]
    total_forward_time_device_d += (layer_bits[md] * end_vars['device']) / bandwidth[0]

    total_backward_time_device_d = 0
    total_backward_time_edge_d = 0
    total_backward_time_cloud_d = 0
    for i in range(md):
        total_backward_time_device_d += end_vars['device'] * train_times_d['backward'][i]
        total_backward_time_edge_d += end_vars['edge'] * train_times_e['backward'][i]
        total_backward_time_cloud_d += end_vars['cloud'] * train_times_c['backward'][i]
    total_backward_time_device_d += (layer_bits[md] * end_vars['device']) / bandwidth[0]

    max_total_forward_time_device_module = LpVariable("MaxTotalForwardTimeDeviceModule", 0)
    max_total_backward_time_device_module = LpVariable("MaxTotalBackwardTimeDeviceModule", 0)
    prob += max_total_forward_time_device_module >= total_forward_time_device_d
    prob += max_total_forward_time_device_module >= total_forward_time_edge_d
    prob += max_total_forward_time_device_module >= total_forward_time_cloud_d
    prob += max_total_backward_time_device_module >= total_backward_time_device_d
    prob += max_total_backward_time_device_module >= total_backward_time_edge_d
    prob += max_total_backward_time_device_module >= total_backward_time_cloud_d

    # edge_module
    total_forward_time_edge_e = 0
    total_forward_time_cloud_e = 0
    total_backward_time_edge_e = 0
    total_backward_time_cloud_e = 0
    for i in range(md, me):
        total_forward_time_edge_e += (end_vars['edge'] ) * train_times_e['forward'][i]
        total_forward_time_cloud_e += (end_vars['cloud'] + end_vars['device']) * train_times_c['forward'][i]
        total_backward_time_edge_e += (end_vars['edge'] ) * train_times_e['backward'][i]
        total_backward_time_cloud_e += (end_vars['cloud'] + end_vars['device']) * train_times_c['backward'][i]

    total_forward_time_edge_e += (layer_bits[me] * (end_vars['edge'])) / bandwidth[3]
    total_backward_time_edge_e += (layer_bits[me] * (end_vars['edge'])) / bandwidth[3]

    max_total_forward_time_edge_module = LpVariable("MaxTotalForwardTimeEdgeModule", 0)
    max_total_backward_time_edge_module = LpVariable("MaxTotalBackwardTimeEdgeModule", 0)
    prob += max_total_forward_time_edge_module >= total_forward_time_edge_e
    prob += max_total_forward_time_edge_module >= total_forward_time_cloud_e
    prob += max_total_backward_time_edge_module >= total_backward_time_edge_e
    prob += max_total_backward_time_edge_module >= total_backward_time_cloud_e

    # cloud_module
    total_forward_time_cloud_module = 0
    total_backward_time_cloud_module = 0
    for i in range(me, max_layers):
        total_forward_time_cloud_module += total_number_samples * train_times_c['forward'][i]
        total_backward_time_cloud_module += total_number_samples * train_times_c['backward'][i]

    # update time
    total_update_time_cloud = 0
    total_update_time_edge = 0
    total_update_time_device = 0
    parameters_to_edge = 0
    parameters_to_device = 0

    for i in range(max_layers):
        total_update_time_cloud += train_times_c['update'][i]
    for i in range(me):
        total_update_time_edge += train_times_e['update'][i]
        parameters_to_edge += layer_parameters_bits[i]
    for i in range(md):
        total_update_time_device = train_times_d['update'][i]
        parameters_to_device += layer_parameters_bits[i]

    max_parameter_to_time = LpVariable("MaxParameterToTime", 0)
    prob += max_parameter_to_time >= parameters_to_edge / bandwidth[3]
    prob += max_parameter_to_time >= parameters_to_device / bandwidth[3]

    total_update_time = LpVariable("TotalUpdateTime", 0)
    prob += total_update_time >= total_update_time_cloud
    prob += total_update_time >= total_update_time_edge
    prob += total_update_time >= total_update_time_device

    # target equation
    prob += lpSum([max_total_forward_time_device_module, max_total_backward_time_device_module,
                   max_total_forward_time_edge_module, max_total_backward_time_edge_module,
                   total_forward_time_cloud_module, total_backward_time_cloud_module, total_update_time+max_parameter_to_time])
    # restrictive condition
    prob += lpSum([end_vars[i] for i in End]) == total_number_samples
    prob += end_vars['device'] >= 0
    prob += end_vars['device'] <= md * total_number_samples
    prob += end_vars['edge'] >= 0
    prob += end_vars['edge'] <= me * total_number_samples

    prob.solve()

    print("Status:", LpStatus[prob.status])

    for v in prob.variables():
        print(v.name, "=", v.varValue)
        number_samples_end.append(v.varValue)
    print(value(prob.objective))
    return number_samples_end, value(prob.objective)





