import copy
import os
import json
import pickle
import numpy as np
import yaml

# rl_dir = '/home/wangfuyu/Desktop/DNN_ACC/Scheduler/RL_Simba/report/simba_arch_4x'

rl_dir2 = '/home/wangfuyu/Desktop/DNN_ACC/Scheduler/RL_mind2/report/simba_arch/fitness_latency/sampled_episodes_32'
cosa_dir = '/home/wangfuyu/Desktop/DNN_ACC/mindmappings/mindmappings/report/1000steps'

# rl_dir2 = '/home/wangfuyu/Desktop/DNN_ACC/Scheduler/RL_Eyeriss2/report/eyeriss{}/sampled_episodes_32'.format(arch)
# cosa_dir = '/home/wangfuyu/Desktop/DNN_ACC/Mapper/cosa/output_dir/eyeriss_v3{}'.format(arch)

dnns = ['resnet50', 'vgg16', 'deepbench', 'resnext50_32x4d']
# dnns = ['transformer']
# dnns = ['vgg16']
# dnns=['deepbench']
# dnns = ['resnext50_32x4d']
total_rl = []
total_cosa = []
for dnn in dnns:
    # input_sizes = [1, 2, 4, 16, 64]
    input_sizes = [1, 16]
    layers = yaml.load(open('../in_config/{}_problems/layers.yaml'.format(dnn), 'r'), Loader=yaml.SafeLoader)
    if dnn == 'resnet50':
        layer_ids = [0, 1, 2, 3, 5, 11, 12, 13, 14, 15, 16, 24, 25, 26, 27, 28, 29, 43, 44, 45, 46, 47, 48, 53]
    elif dnn == 'vgg16':
        layer_ids = [0, 1, 2, 3, 4, 5, 7, 8, 10, 13, 14, 15] # vgg16
    elif dnn == 'deepbench':
        layer_ids = [0, 1, 2, 3, 4, 5, 6, 7, 8]  # deepbench
    elif dnn == 'resnext50_32x4d':
        layer_ids = [1, 2, 3, 11, 12, 13, 16, 24, 25, 26, 29, 43, 44, 45, 47, 48] # resnext50
    else:
        layer_ids = [0, 1, 2]   # transformer
    num_layers = len(layer_ids)
    total_rl_fitnesses2 = []
    total_cosa_fitnesses = []
    for i, layer_id in enumerate(layer_ids):
        # for i in [2, 24, 28, 44]:
        #     layer = layers[i]
        with open('../in_config/{}_problems/{}.yaml'.format(dnn, layers[layer_id]), 'r') as fd:
            layer_problem = yaml.load(fd, Loader=yaml.SafeLoader)
            if layer_problem['problem']['Wstride'] > 1:
                continue
        fd.close()

        rl_fitnesses2 = []
        cosa_fitnesses = []

        for input_size in input_sizes:
            cosa_chkpt = pickle.load(open(
                os.path.join(cosa_dir, '{}_input{}'.format(dnn, input_size), 'layer-{}'.format(layer_id),
                             'env_chkpt.plt'),
                'rb'))
            cosa_fitness = -1 * cosa_chkpt['latency']
            cosa_fitnesses.append(cosa_fitness)

            rl_chkpt2 = pickle.load(open(
                os.path.join(rl_dir2, '{}_input{}'.format(dnn, input_size), 'layer-{}'.format(layer_id), 'env_chkpt.plt'),
                'rb'))
            rl_fitness2 = -1 * rl_chkpt2['best_fitness_record'][4]
            # rl_fitness2 = -1 * rl_chkpt2['best_energy_record'][14]*1e6
            # rl_fitness2 = -1 * rl_chkpt2['best_latency_record'][14]
            # rl_fitness2 = cosa_fitness /1.6
            rl_fitnesses2.append(rl_fitness2)

        # print(i, layer_id)
        # print('cosa: ', cosa_fitnesses, np.around(np.array(cosa_fitnesses[-1]) / np.array(cosa_fitnesses[0]), 2))
        # print('rl2: ', rl_fitnesses2)

        total_rl_fitnesses2.append(rl_fitnesses2)
        total_cosa_fitnesses.append(cosa_fitnesses)

    total_cosa_fitnesses = np.array(total_cosa_fitnesses)
    total_rl_fitnesses2 = np.array(total_rl_fitnesses2)
    total_rl_fitnesses2_tmp = copy.deepcopy(total_rl_fitnesses2)
    total_cosa_fitnesses_tmp = copy.deepcopy(total_cosa_fitnesses)
    # total_rl_fitnesses2_tmp[total_cosa_fitnesses==float('inf')] = 0
    total_cosa_fitnesses_tmp[total_cosa_fitnesses==float('inf')] = 0
    # print(total_cosa_fitnesses_tmp.mean(axis=0) / total_rl_fitnesses2_tmp.mean(axis=0))
    print((total_cosa_fitnesses_tmp / total_rl_fitnesses2_tmp).sum(axis=0) / (total_cosa_fitnesses_tmp>0).sum(axis=0))
    # np.save('./npy/total_rl_fitnesses_{}_simba_arch{}_pe.npy'.format(dnn, arch), total_rl_fitnesses2)
    # np.save('./npy/total_cosa_fitnesses_{}_simba_arch{}_pe.npy'.format(dnn, arch), total_cosa_fitnesses)

    # np.save('./npy/total_rl_energy_{}_simba_arch{}_pe.npy'.format(dnn, arch), total_rl_fitnesses2)
    # np.save('./npy/total_cosa_energy_{}_simba_arch{}_pe.npy'.format(dnn, arch), total_cosa_fitnesses)

    # np.save('./npy/total_rl_fitnesses_{}_eyeriss_arch{}_pe.npy'.format(dnn, arch), total_rl_fitnesses2)
    # np.save('./npy/total_cosa_fitnesses_{}_eyeriss_arch{}_pe.npy'.format(dnn, arch), total_cosa_fitnesses)

    total_cosa.append(total_cosa_fitnesses_tmp)
    total_rl.append(total_rl_fitnesses2_tmp)

total_cosa = np.concatenate(total_cosa, axis=0)
total_rl = np.concatenate(total_rl, axis=0)

print((total_cosa / total_rl).sum(axis=0) / (total_cosa>0).sum(axis=0))
# print((total_cosa[:, 1] / total_cosa[:, 0]).sum(axis=0) / (total_cosa[:, 1]>0).sum(axis=0),
#       (total_cosa[:, 2] / total_cosa[:, 0]).sum(axis=0) / (total_cosa[:, 2]>0).sum(axis=0),
#       (total_cosa[:, -2] / total_cosa[:, 0]).sum(axis=0) / (total_cosa[:, -2]>0).sum(axis=0),
#       (total_cosa[:, -1] / total_cosa[:, 0]).sum(axis=0) / (total_cosa[:, -1]>0).sum(axis=0))
# print((total_rl[:, 1] / total_rl[:, 0]).sum(axis=0) / (total_rl[:, 1]>0).sum(axis=0),
#       (total_rl[:, 2] / total_rl[:, 0]).sum(axis=0) / (total_rl[:, 2]>0).sum(axis=0),
#       (total_rl[:, 3] / total_rl[:, 0]).sum(axis=0) / (total_rl[:, 3]>0).sum(axis=0),
#       (total_rl[:, 4] / total_rl[:, 0]).sum(axis=0) / (total_rl[:, 4]>0).sum(axis=0),
#       (total_rl[:, -2] / total_rl[:, 0]).sum(axis=0) / (total_rl[:, -2]>0).sum(axis=0),
#       (total_rl[:, -1] / total_rl[:, 0]).sum(axis=0) / (total_rl[:, -1]>0).sum(axis=0))
# print(total_rl.shape, total_rl.shape)