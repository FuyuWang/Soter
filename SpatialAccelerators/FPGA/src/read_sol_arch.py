import copy
import os
import json
import pickle
import numpy as np
import yaml

cosa_dir1 = '/home/wangfuyu/Desktop/DNN_ACC/Mapper/cosa/output_dir/simba_v3_pe'
cosa_dir2 = '/home/wangfuyu/Desktop/DNN_ACC/Mapper/cosa/output_dir/simba_v3_16x_pe'
cosa_dir4 = '/home/wangfuyu/Desktop/DNN_ACC/Mapper/cosa/output_dir/simba_v3_64x_pe'
rl_dir1 = '/home/wangfuyu/Desktop/DNN_ACC/Scheduler/RL_Simba/report/simba_arch_pe/sampled_episodes_32'
rl_dir2 = '/home/wangfuyu/Desktop/DNN_ACC/Scheduler/RL_Simba/report/simba_arch_16x_pe/sampled_episodes_32'
rl_dir4 = '/home/wangfuyu/Desktop/DNN_ACC/Scheduler/RL_Simba/report/simba_arch_64x_pe/sampled_episodes_32'

# cosa_dir1 = '/home/wangfuyu/Desktop/DNN_ACC/Mapper/cosa/output_dir/eyeriss_v3'
# cosa_dir2 = '/home/wangfuyu/Desktop/DNN_ACC/Mapper/cosa/output_dir/eyeriss_v3_4x'
# cosa_dir4 = '/home/wangfuyu/Desktop/DNN_ACC/Mapper/cosa/output_dir/eyeriss_v3_16x'
# rl_dir1 = '/home/wangfuyu/Desktop/DNN_ACC/Scheduler/RL_Eyeriss/report/eyeriss/sampled_episodes_32'
# rl_dir2 = '/home/wangfuyu/Desktop/DNN_ACC/Scheduler/RL_Eyeriss/report/eyeriss_4x/sampled_episodes_32'
# rl_dir4 = '/home/wangfuyu/Desktop/DNN_ACC/Scheduler/RL_Eyeriss/report/eyeriss_16x/sampled_episodes_32'

# dnns = ['resnet50', 'vgg16', 'deepbench', 'resnext50_32x4d']
dnns = ['transformer']
# dnn = 'vgg16'
# dnn='deepbench'
# dnn = 'resnext50_32x4d'
total_rl1 = []
total_rl2 = []
total_rl4 = []
total_cosa1 = []
total_cosa2 = []
total_cosa4 = []
for dnn in dnns:
    # input_sizes = [1, 2, 4, 8, 16]
    input_sizes = [1]
    layers = yaml.load(open('../in_config/{}_problems/layers.yaml'.format(dnn), 'r'), Loader=yaml.SafeLoader)
    if dnn == 'resnet50':
        layer_ids = [0, 1, 2, 3, 5, 11, 12, 13, 14, 15, 16, 24, 25, 26, 27, 28, 29, 43, 44, 45, 46, 47, 48, 53]
    elif dnn == 'vgg16':
        layer_ids = [0, 1, 2, 3, 4, 5, 7, 8, 10, 13, 14, 15] # vgg16
    elif dnn == 'deepbench':
        layer_ids = [0, 1, 2, 3, 4, 5, 6, 7, 8]  # deepbench
    else:
        layer_ids = [1, 2, 3, 11, 12, 13, 16, 24, 25, 26, 29, 43, 44, 45, 47, 48] # resnext50num_layers = len(layer_ids)
    num_layers = len(layer_ids)
    total_cosa_fitnesses1 = np.zeros((num_layers, len(input_sizes)))
    total_cosa_fitnesses2 = np.zeros((num_layers, len(input_sizes)))
    total_cosa_fitnesses4 = np.zeros((num_layers, len(input_sizes)))
    total_rl_fitnesses1 = np.zeros((num_layers, len(input_sizes)))
    total_rl_fitnesses2 = np.zeros((num_layers, len(input_sizes)))
    total_rl_fitnesses4 = np.zeros((num_layers, len(input_sizes)))

    for i, layer_id in enumerate(layer_ids):
        cosa_fitnesses1 = []
        cosa_fitnesses2 = []
        cosa_fitnesses4 = []
        rl_fitnesses1 = []
        rl_fitnesses2 = []
        rl_fitnesses4 = []

        for input_size in input_sizes:
            try:
                cosa_chkpt1 = json.load(open(os.path.join(cosa_dir1, '{}_input{}'.format(dnn, input_size), 'layer-{}'.format(layer_id), 'tc.dict.json'), 'r'))
                cosa_fitness1 =  list(cosa_chkpt1.values())[0]['cycle']
                # cosa_fitness =  list(cosa_chkpt.values())[0]['energy']
                # cosa_fitness =  list(cosa_chkpt.values())[0]['energy'] * list(cosa_chkpt.values())[0]['cycle']
            except:
                cosa_fitness1 = float('inf')
            cosa_fitnesses1.append(cosa_fitness1)

            try:
                cosa_chkpt2 = json.load(open(os.path.join(cosa_dir2, '{}_input{}'.format(dnn, input_size), 'layer-{}'.format(layer_id), 'tc.dict.json'), 'r'))
                cosa_fitness2 =  list(cosa_chkpt2.values())[0]['cycle']
                # cosa_fitness =  list(cosa_chkpt.values())[0]['energy']
                # cosa_fitness =  list(cosa_chkpt.values())[0]['energy'] * list(cosa_chkpt.values())[0]['cycle']
            except:
                cosa_fitness2 = float('inf')
            cosa_fitnesses2.append(cosa_fitness2)

            try:
                cosa_chkpt4 = json.load(open(os.path.join(cosa_dir4, '{}_input{}'.format(dnn, input_size), 'layer-{}'.format(layer_id), 'tc.dict.json'), 'r'))
                cosa_fitness4 =  list(cosa_chkpt4.values())[0]['cycle']
                # cosa_fitness =  list(cosa_chkpt.values())[0]['energy']
                # cosa_fitness =  list(cosa_chkpt.values())[0]['energy'] * list(cosa_chkpt.values())[0]['cycle']
            except:
                cosa_fitness4 = float('inf')
            cosa_fitnesses4.append(cosa_fitness4)

            rl_chkpt1 = pickle.load(open(
                os.path.join(rl_dir1, '{}_input{}'.format(dnn, input_size), 'layer-{}'.format(layer_id), 'env_chkpt.plt'),
                'rb'))
            rl_fitness1 = -1 * rl_chkpt1['best_fitness_record'][14]
            # rl_fitness2 = -1 * rl_chkpt2['best_latency_record'][19]*1e6*rl_chkpt2['best_fitness_record'][19]
            # rl_fitness2 = -1 * rl_chkpt2['best_latency_record'][14]*1e6
            rl_fitnesses1.append(rl_fitness1)

            rl_chkpt2 = pickle.load(open(
                os.path.join(rl_dir2, '{}_input{}'.format(dnn, input_size), 'layer-{}'.format(layer_id), 'env_chkpt.plt'),
                'rb'))
            rl_fitness2 = -1 * rl_chkpt2['best_fitness_record'][14]
            # rl_fitness2 = -1 * rl_chkpt2['best_latency_record'][19]*1e6*rl_chkpt2['best_fitness_record'][19]
            # rl_fitness2 = -1 * rl_chkpt2['best_latency_record'][14]*1e6
            rl_fitnesses2.append(rl_fitness2)

            rl_chkpt4 = pickle.load(open(
                os.path.join(rl_dir4, '{}_input{}'.format(dnn, input_size), 'layer-{}'.format(layer_id), 'env_chkpt.plt'),
                'rb'))
            rl_fitness4 = -1 * rl_chkpt4['best_fitness_record'][14]
            # rl_fitness2 = -1 * rl_chkpt2['best_latency_record'][19]*1e6*rl_chkpt2['best_fitness_record'][19]
            # rl_fitness2 = -1 * rl_chkpt2['best_latency_record'][14]*1e6
            rl_fitnesses4.append(rl_fitness4)
            # if rl_fitness1 / rl_fitness2 > cosa_fitness1 / cosa_fitness2 :
            #     print(i, layer_id, cosa_fitness1, cosa_fitness2, cosa_fitness4, cosa_fitness1 / cosa_fitness2, cosa_fitness1 / cosa_fitness4)
            #     print(i, layer_id, rl_fitness1, rl_fitness2, rl_fitness4, rl_fitness1 / rl_fitness2, rl_fitness1 / rl_fitness4)

        total_rl_fitnesses1[i] = rl_fitnesses1
        total_rl_fitnesses2[i] = rl_fitnesses2
        total_rl_fitnesses4[i] = rl_fitnesses4
        total_cosa_fitnesses1[i] = cosa_fitnesses1
        total_cosa_fitnesses2[i] = cosa_fitnesses2
        total_cosa_fitnesses4[i] = cosa_fitnesses4



    total_rl_fitnesses1_tmp = copy.deepcopy(total_rl_fitnesses1)
    total_cosa_fitnesses1_tmp = copy.deepcopy(total_cosa_fitnesses1)
    total_cosa_fitnesses1_tmp[total_cosa_fitnesses1 == float('inf')] = 0

    total_rl_fitnesses2_tmp = copy.deepcopy(total_rl_fitnesses2)
    total_cosa_fitnesses2_tmp = copy.deepcopy(total_cosa_fitnesses2)
    total_cosa_fitnesses2_tmp[total_cosa_fitnesses2 == float('inf')] = 0

    total_rl_fitnesses4_tmp = copy.deepcopy(total_rl_fitnesses4)
    total_cosa_fitnesses4_tmp = copy.deepcopy(total_cosa_fitnesses4)
    total_cosa_fitnesses4_tmp[total_cosa_fitnesses4 == float('inf')] = 0

    total_cosa1.append(total_cosa_fitnesses1_tmp)
    total_rl1.append(total_rl_fitnesses1_tmp)

    total_cosa2.append(total_cosa_fitnesses2_tmp)
    total_rl2.append(total_rl_fitnesses2_tmp)

    total_cosa4.append(total_cosa_fitnesses4_tmp)
    total_rl4.append(total_rl_fitnesses4_tmp)

total_cosa1 = np.concatenate(total_cosa1, axis=0)
total_rl1 = np.concatenate(total_rl1, axis=0)

total_cosa2 = np.concatenate(total_cosa2, axis=0)
total_rl2 = np.concatenate(total_rl2, axis=0)

total_cosa4 = np.concatenate(total_cosa4, axis=0)
total_rl4 = np.concatenate(total_rl4, axis=0)

print((total_cosa1 / total_rl1).sum(axis=0) / (total_cosa1>0).sum(axis=0))
print((total_cosa2 / total_rl2).sum(axis=0) / (total_cosa2>0).sum(axis=0))
print((total_cosa4 / total_rl4).sum(axis=0) / (total_cosa4>0).sum(axis=0))
# print(total_cosa4)

print(total_cosa1.shape, total_cosa2.shape)

# print((total_cosa1[:, 0] / total_cosa2[:, 0]).mean())
# print((total_cosa2[:, 0] / total_cosa4[:, 0]).mean())
# print((total_cosa1[:, 0] / total_cosa4[:, 0]).mean())
#
# print((total_rl1[:, 0] / total_rl2[:, 0]).mean())
# print((total_rl2[:, 0] / total_rl4[:, 0]).mean())
# print((total_rl1[:, 0] / total_rl4[:, 0]).mean())

# total_rl_fitnesses2_tmp = copy.deepcopy(total_rl_fitnesses2)
# total_cosa_fitnesses_tmp = copy.deepcopy(total_cosa_fitnesses)
# # total_rl_fitnesses2_tmp[total_cosa_fitnesses==float('inf')] = 0
# total_cosa_fitnesses_tmp[total_cosa_fitnesses==float('inf')] = 0
# print(total_cosa_fitnesses_tmp.mean(axis=0) / total_rl_fitnesses2_tmp.mean(axis=0))
# print((total_cosa_fitnesses_tmp / total_rl_fitnesses2_tmp).sum(axis=0) / (total_cosa_fitnesses_tmp>0).sum(axis=0))
# np.save('./npy/total_rl_fitnesses_{}_simba_arch{}_pe.npy'.format(dnn, arch), total_rl_fitnesses2)
# np.save('./npy/total_cosa_fitnesses_{}_simba_arch{}_pe.npy'.format(dnn, arch), total_cosa_fitnesses)

# np.save('./npy/total_rl_energy_{}_simba_arch{}_pe.npy'.format(dnn, arch), total_rl_fitnesses2)
# np.save('./npy/total_cosa_energy_{}_simba_arch{}_pe.npy'.format(dnn, arch), total_cosa_fitnesses)

# np.save('./npy/total_rl_fitnesses_{}_eyeriss_arch{}_pe.npy'.format(dnn, arch), total_rl_fitnesses2)
# np.save('./npy/total_cosa_fitnesses_{}_eyeriss_arch{}_pe.npy'.format(dnn, arch), total_cosa_fitnesses)
