import logging

import numpy as np
import yaml
import os, sys
import copy
import math
from functools import reduce
from collections import defaultdict, OrderedDict
from subprocess import Popen, PIPE, call
from parse_timeloop_output import parse_timeloop_stats
# from pytimeloop.app import Model
# from pytimeloop import ConfigDict
from utils import *
import re


class TimeloopEnv(object):
    def __init__(self, config_path='./out_config', in_config_dir= './in_config', arch_file='arch', map_file='mapspace',
                 debug=False, use_sparse=False, density=None):

        self.config_path = config_path
        self.use_sparse = use_sparse
        with open(os.path.join(in_config_dir, '{}.yaml'.format(arch_file)), 'r') as fd:
            self.arch = yaml.load(fd, Loader = yaml.SafeLoader)
        fd.close()
        with open(os.path.join(in_config_dir, 'problem.yaml'), 'r') as fd:
            self.problem = yaml.load(fd,Loader = yaml.SafeLoader)
        fd.close()
        with open(os.path.join(in_config_dir, '{}.yaml'.format(map_file)), 'r') as fd:
            self.mapspace = yaml.load(fd, Loader = yaml.SafeLoader)
        fd.close()
        if self.use_sparse:
            with open(os.path.join(in_config_dir, 'sparse.yaml'), 'r') as fd:
                self.sparse = yaml.load(fd,Loader = yaml.SafeLoader)
            fd.close()

        buffer_name_list, buffer_size_list, buffer_spmap_cstr, num_buffer_levels, num_pes = self.get_buffer_info()
        self.buffer_name_list = buffer_name_list
        self.buffer_size_list = buffer_size_list
        self.buffer_spmap_cstr = buffer_spmap_cstr
        self.buffers_with_spmap = set([key for key, value in self.buffer_spmap_cstr.items() if value > 1])
        self.num_buffer_level = num_buffer_levels
        self.num_pes = num_pes
        self._executable = 'timeloop-model'
        self.debug = debug
        self.buf_energy_cost = self.get_default_buffer_energy_cost()
        self.density = density

        self.dim_note = ['N', 'K', 'C', 'P', 'Q', 'R', 'S']
        self.dimension, self.dimension_dict = self.get_problem_info()
        self.dimension_prime = {key: self.get_prime_factors(self.dimension_dict[key]) for key in self.dim_note}

        self.prime2idx = {}
        primes = set()
        for i, key in enumerate('NKCPQRS'):
            tile_budget = self.dimension_prime[key]
            for k in tile_budget.keys():
                primes.add(k)
        primes = sorted(primes)
        self.prime2idx = {pf: i for i, pf in enumerate(primes)}
        self.num_primes = len(self.prime2idx.keys())

        # self.bypass_dict = {}
        # for i in range(self.num_buffer_level, 0, -1):
        #     if i == 6:  # DRAM
        #         self.bypass_dict[self.buffer_name_list[f'l{i}']] = {'Inputs':False, 'Weights': False, 'Outputs': False}
        #     elif i == 5:    # GlobalBuffer
        #         self.bypass_dict[self.buffer_name_list[f'l{i}']] = {'Inputs':False, 'Weights': True, 'Outputs': False}
        #     elif i == 4:    # PEInputBuffer
        #         self.bypass_dict[self.buffer_name_list[f'l{i}']] = {'Inputs':False, 'Weights': True, 'Outputs': True}
        #     elif i == 3:    # PEWeightBuffer
        #         self.bypass_dict[self.buffer_name_list[f'l{i}']] = {'Inputs':True, 'Weights': False, 'Outputs': True}
        #     elif i == 2:    # PEAccBuffer
        #         self.bypass_dict[self.buffer_name_list[f'l{i}']] = {'Inputs':True, 'Weights': True, 'Outputs': False}
        #     elif i == 1:     # PEWeightReg
        #         self.bypass_dict[self.buffer_name_list[f'l{i}']] = {'Inputs':True, 'Weights': False, 'Outputs': True}
        # print(self.bypass_dict)

    def get_default_buffer_energy_cost(self):
        buf_energy_cost = {'DRAM': 200,
                           'l2': 2.2,
                           'l1': 1.12,
                           'MAC': 1.0,
        }
        return buf_energy_cost

    def get_num_buffer_levels(self):
        return self.num_buffer_level

    def get_buffers_with_spmap(self):
        return self.buffers_with_spmap

    def get_buffer_spmap_cstr(self):
        return self.buffer_spmap_cstr

    def get_buffer_size_list(self):
        return self.buffer_size_list

    def get_prime_factors(self, n):
        primes = defaultdict(int)
        while n % 2 == 0:
            primes['2'] += 1
            n = n // 2
        for i in range(3, int(math.sqrt(n)) + 1, 2):
            while n % i == 0:
                primes[f'{i}'] += 1
                n = n // i
        if n > 2:
            primes[f'{n}'] += 1
        return primes

    def get_factors(self, n):
        return list(reduce(list.__add__,
                           ([i, n // i] for i in range(1, int(n ** 0.5) + 1) if n % i == 0)))

    def get_dimension_factors(self, dimension_dict):
        dimension_factors = dict()
        for key, value in dimension_dict.items():
            factors = self.get_factors(value)
            dimension_factors[key] = factors
        return dimension_factors

    def get_dimension_primes(self):
        return self.dimension, self.dimension_prime, self.prime2idx

    def get_problem_info(self):
        dim_note = 'NKCPQRS'
        problem = copy.deepcopy(self.problem)
        dimension = []
        dimension_dicts = {}
        for key in dim_note:
            value = problem['problem']['instance'][self.get_timeloop_notation(key)]
            dimension.append(value)
            dimension_dicts[key] = value
        return dimension, dimension_dicts

    def get_buffer_info(self):
        arch = copy.deepcopy(self.arch)
        buffer_name_list = []
        buffer_size_list = []
        num_instances = []
        num_buffer_levels = 0
        arch = arch['architecture']

        main_memory = arch['subtree'][0]
        buffer_name = main_memory['name']
        attributes = main_memory['local'][0]['attributes']
        depth = attributes['depth'] if 'depth' in attributes else float('Inf')
        word_bits = attributes['word-bits'] if 'word-bits' in attributes else 8
        width = attributes['width'] if 'width' in attributes else 8
        block_size = attributes['block-size'] if 'block-size' in attributes else 1
        buffer_size = depth * block_size
        instances = 1
        re_ret = re.search('.*\[', buffer_name)
        if re_ret:
            instances = int(buffer_name.split('..')[1].split(']')[0]) + 1
        buffer_name = main_memory['local'][0]['name']
        buffer_name_list.append(buffer_name)
        buffer_size_list.append(buffer_size)
        num_instances.append(instances)
        num_buffer_levels += 1

        global_buffer = arch['subtree'][0]['subtree'][0]
        buffer_name = global_buffer['name']
        attributes = global_buffer['local'][0]['attributes']
        depth = attributes['depth'] if 'depth' in attributes else float('Inf')
        word_bits = attributes['word-bits'] if 'word-bits' in attributes else 8
        width = attributes['width'] if 'width' in attributes else 8
        block_size = attributes['block-size'] if 'block-size' in attributes else 1
        buffer_size = depth * block_size
        re_ret = re.search('.*\[', buffer_name)
        if re_ret:
            instances *= int(buffer_name.split('..')[1].split(']')[0]) + 1
        buffer_name = global_buffer['local'][0]['name']
        buffer_name_list.append(buffer_name)
        buffer_size_list.append(buffer_size)
        num_instances.append(instances)
        num_buffer_levels += 1

        local_buffer = arch['subtree'][0]['subtree'][0]['subtree'][0]
        buffer_name = local_buffer['name']
        attributes = local_buffer['local'][0]['attributes']
        depth = attributes['depth'] if 'depth' in attributes else float('Inf')
        word_bits = attributes['word-bits'] if 'word-bits' in attributes else 8
        width = attributes['width'] if 'width' in attributes else 8
        block_size = attributes['block-size'] if 'block-size' in attributes else 1
        buffer_size = depth * block_size
        re_ret = re.search('.*\[', buffer_name)
        if re_ret:
            instances *= int(buffer_name.split('..')[1].split(']')[0]) + 1
        buffer_name = local_buffer['local'][0]['name']
        buffer_name_list.append(buffer_name)
        buffer_size_list.append(buffer_size)
        num_instances.append(instances)
        num_buffer_levels += 1

        pe_buffer = arch['subtree'][0]['subtree'][0]['subtree'][0]['subtree'][0]
        buffer_name = pe_buffer['name']
        attributes = pe_buffer['local'][0]['attributes']
        depth = attributes['depth'] if 'depth' in attributes else float('Inf')
        word_bits = attributes['word-bits'] if 'word-bits' in attributes else 8
        width = attributes['width'] if 'width' in attributes else 8
        block_size = attributes['block-size'] if 'block-size' in attributes else 1
        buffer_size = depth * block_size
        re_ret = re.search('.*\[', buffer_name)
        if re_ret:
            instances *= int(buffer_name.split('..')[1].split(']')[0]) + 1
        buffer_name = pe_buffer['local'][0]['name']
        buffer_name_list.append(buffer_name)
        buffer_size_list.append(buffer_size)
        num_instances.append(instances)
        num_buffer_levels += 1
        num_pes = instances

        macc = arch['subtree'][0]['subtree'][0]['subtree'][0]['subtree'][0]['local'][1]['name']
        re_ret = re.search('.*\[', macc)
        if re_ret:
            instances *= (int(macc.split('..')[1].split(']')[0]) + 1)
        num_instances.append(instances)


        sp_cstr = []
        for i in range(len(num_instances) - 1):
            allowed_sp_size = num_instances[i + 1] // num_instances[i]
            sp_cstr.append(allowed_sp_size)
            if num_instances[i + 1] % num_instances[i] != 0:
                raise ValueError('Invalid Architecture File. '
                                 'Buffer hierarchy not perfectly divisible.')

        print(buffer_name_list, num_instances, buffer_size_list, num_pes, sp_cstr)

        return {f'l{level}': name for level, name in zip(np.arange(num_buffer_levels, 0, -1), buffer_name_list)}, \
               {f'l{level}': name for level, name in zip(np.arange(num_buffer_levels, 0, -1), buffer_size_list)}, \
               {f'l{level}': name for level, name in zip(np.arange(num_buffer_levels, 0, -1), sp_cstr)}, \
               num_buffer_levels, num_pes

    def get_timeloop_notation(self, g):
        # timeloop_dict = {'N': 'N', 'K': 'M', 'C': 'C', 'Y': 'P', 'X': 'Q', 'R': 'R', 'S': 'S'}
        timeloop_dict = {'N': 'N', 'K': 'K', 'C': 'C', 'P': 'P', 'Q': 'Q', 'R': 'R', 'S': 'S'}
        return timeloop_dict[g]

    def get_dimension_dict(self, dim_value):
        dim_note = 'NKCPQRS'
        return {note: value for note, value in zip(dim_note, dim_value)}

    def init_tp_tile_size(self):
        series =  [f'{self.get_timeloop_notation(note)}={1}' for note in 'NKCPQRS']
        return ' '.join(series)

    def get_tp_tile_size(self, dim_value):
        series =  [f'{self.get_timeloop_notation(note)}={value}' for note, value in dim_value.items()]
        return ' '.join(series)

    def get_tp_sp_tile_size(self, dim_value, sp_dim, sp_dim_value, timeloop_notation=True):
        if timeloop_notation:
            temporal_series = []
            spatial_series = []
            for note, value in dim_value.items():
                if note not in sp_dim:
                    temporal_series.append(f'{self.get_timeloop_notation(note)}={value}')
                    spatial_series.append(f'{self.get_timeloop_notation(note)}=1')
                else:
                    sp_value = sp_dim_value[note]
                    tp_value = value // sp_value
                    temporal_series.append(f'{self.get_timeloop_notation(note)}={tp_value}')
                    spatial_series.append(f'{self.get_timeloop_notation(note)}={sp_value}')
            return ' '.join(temporal_series), ' '.join(spatial_series)
        else:
            temporal_series = []
            spatial_series = []
            for note in 'NKCPQRS':
                if note not in sp_dim:
                    temporal_series.append(dim_value[note])
                    spatial_series.append(1)
                else:
                    sp_value = sp_dim_value[note]
                    tp_value = dim_value[note] // sp_value
                    temporal_series.append(tp_value)
                    spatial_series.append(sp_value)
            return np.array(temporal_series), np.array(spatial_series)

    def get_loop_order(self, loop_order):
        series = [self.get_timeloop_notation(g) for g in loop_order]
        return ''.join(series)

    def create_pool_env(self, num_pools, dimension, sol, use_IO=False):
        os.makedirs(self.config_path, exist_ok=True)
        if use_IO:
            arch_paths, problem_paths, map_paths, sparse_paths, pool_paths = [], [], [], [], []
            for i in range(num_pools):
                pool_dir = os.path.join(self.config_path, f'pool-{i}')
                os.makedirs(pool_dir, exist_ok=True)
                pool_paths.append(pool_dir)
                arch_paths.append(os.path.abspath(os.path.join(pool_dir, 'arch.yaml')))
                problem_paths.append(os.path.abspath(os.path.join(pool_dir, 'problem.yaml')))
                map_paths.append(os.path.abspath(os.path.join(pool_dir, 'map.yaml')))
                sparse_paths.append(os.path.abspath(os.path.join(pool_dir, 'sparse.yaml')))
            self.arch_path, self.problem_path, self.map_path, self.sparse_path, self.pool_path =  arch_paths, problem_paths, map_paths, sparse_paths, pool_paths
        else:
            arch, problem, map = self.get_configs(dimension, sol)
            cfg = {}
            cfg.update(arch)
            cfg.update(map)
            cfg.update(problem)
            if self.use_sparse:
                cfg.update(self.sparse)
                # cfg.update({'sparse_optimizations': self.sparse})
            config = ConfigDict(cfg)
            with stdout_redirected():
                timeloop_app = Model(config, self.config_path)
            with open(os.path.join(self.config_path, 'timeloop-model.ART.yaml'), 'r') as fd:
                art = yaml.load(fd, Loader = yaml.SafeLoader)
            with open(os.path.join(self.config_path, 'timeloop-model.ERT.yaml'), 'r') as fd:
                ert = yaml.load(fd, Loader = yaml.SafeLoader)
            cfg.update(art)
            cfg.update(ert)
            self.art = art
            self.ert = ert
            self.shared_cfg = cfg

    def get_arch_configs(self, l2_size, l1_size, num_pes):
        arch = copy.deepcopy(self.arch)
        arch['architecture']['subtree'][0]['subtree'][0]['local'][0]['attributes']['depth'] = l2_size
        arch['architecture']['subtree'][0]['subtree'][0]['subtree'][0]['local'][0]['name']=f'RegisterFile[0..{num_pes}]'
        arch['architecture']['subtree'][0]['subtree'][0]['subtree'][0]['local'][0]['attributes']['depth'] = l1_size
        arch['architecture']['subtree'][0]['subtree'][0]['subtree'][0]['local'][1]['name']=f'MACC[0..{num_pes}]'
        return arch

    def get_problem_configs(self, dimension):
        problem =  copy.deepcopy(self.problem)
        dimension_dict = self.get_dimension_dict(dimension)
        for key, value in dimension_dict.items():
            problem['problem']['instance'][self.get_timeloop_notation(key)] = value
        if self.use_sparse:
            problem['problem']['instance']['densities'] = {}
            for key in ['Inputs', 'Weights', 'Outputs']:
                cur_density = self.density[key]
                if cur_density < 1:
                    problem['problem']['instance']['densities'][key] = {}
                    problem['problem']['instance']['densities'][key]['distribution'] = 'fixed-structured'
                    # problem['problem']['instance']['densities'][key]['distribution'] = 'hypergeometric'
                    problem['problem']['instance']['densities'][key]['density'] = cur_density
        return problem

    def get_prod(self, dicts):
        ret_value = 1
        for k, v in dicts.items():
            ret_value *= ((int(k))**v)
        return ret_value

    def get_bypass(self, bypass):
        to_pass = [k  for k, v in bypass.items() if v]
        to_keep = [k  for k, v in bypass.items() if not v]
        return to_pass, to_keep

    def get_input_weight_output_tile(self, tiles):
        N, K, C, Y, X, R, S = tiles
        input_tile, weight_tile, output_tile = N*(Y+R-1)*(X+S-1)*C, K*R*S*C, Y*X*K*N
        return input_tile, weight_tile, output_tile

    def get_ideal_perf(self, dimension):
        N, K, C, Y, X, R, S = dimension
        input_size, weight_size, output_size = [N*Y*X*C, R*S*C*K, N*Y*X*K] # Input, weight, output
        num_flops = N*R*S*C*Y*X*K
        energys = {}
        for level in range(1, self.num_buffer_level+1):
            if level == 1:
                buf_energy_cost = self.buf_energy_cost['l1']
            elif level == self.num_buffer_level:
                buf_energy_cost = self.buf_energy_cost['DRAM']
            else:
                buf_energy_cost = self.buf_energy_cost['l2']
            energys[f'l{level}-Inputs'] = input_size * buf_energy_cost
            energys[f'l{level}-Weights'] = weight_size * buf_energy_cost
            energys[f'l{level}-Outputs'] = output_size * buf_energy_cost
        energys['compute'] = num_flops * self.buf_energy_cost['MAC']
        energy = sum(e for e in energys.values()) * 1e-6  # energy_uJ
        # cycles = num_flops/self.num_pes
        cycles = num_flops/(self.num_pes-1)
        edp = cycles * energy
        return edp, cycles, energy

    def get_map_config(self, sol):
        steps_per_level = 7
        # sol [level*steps_per_level, 5]
        dim2note = {0: 'N', 1: 'K', 2: 'C', 3: 'P', 4: 'Q', 5: 'R', 6: 'S'}
        mapping = []
        # self.check_tile_fit_buffer(sol)
        num_primes = len(self.prime2idx.keys())
        for level in range(1, self.num_buffer_level+1):
            target = self.buffer_name_list[f'l{level}']
            level_sol = sol[(level-1)*steps_per_level:level*steps_per_level,:]
            par_dims = set()
            permutation = ''
            tile_sizes_dict = {}
            sp_tile_sizes_dict = {}
            for i in range(steps_per_level):
                note = dim2note[level_sol[i, 0]]
                permutation += note
                if level_sol[i, num_primes + 1] == 1:
                    par_dims.add(note)
                tile_sizes_dict[note] = 1
                for k, v in self.prime2idx.items():
                    tile_sizes_dict[note] *= pow(int(k), level_sol[i, int(v) + 1])
                sp_tile_sizes_dict[note] = pow(2, level_sol[i, num_primes + 2])

            # bypass = self.bypass_dict[self.buffer_name_list[f'l{level}']]
            # to_pass, to_keep = self.get_bypass(bypass)
            # bypass_map = {'target': target,
            #               'type': 'bypass',
            #               'keep': to_keep,
            #               'bypass': to_pass
            #             }
            bypass_map = self.mapspace['mapspace']['constraints'][level - 1]

            tp_tile_sizes, sp_tile_sizes = self.get_tp_sp_tile_size(tile_sizes_dict, par_dims, sp_tile_sizes_dict)
            sp_tile_sizes_tmp = np.array(self.get_tp_sp_tile_size(tile_sizes_dict, par_dims, sp_tile_sizes_dict,
                                                                  timeloop_notation=False)[1])
            if np.prod(sp_tile_sizes_tmp) > self.buffer_spmap_cstr[f'l{level}']:
                print("level-tile_sizes: ", level, level_sol, sp_tile_sizes, sp_tile_sizes_tmp, self.buffer_spmap_cstr[f'l{level}'], par_dims)

            cur_map = {'target': target,
                       'type': 'temporal',
                        'factors': tp_tile_sizes,
                       'permutation': permutation,
                       }
            mapping.append(cur_map)
            if f'l{level}' in self.buffers_with_spmap:
                # print(f'l{level}', self.buffers_with_spmap, target)
                cur_map = {'target': target,
                           'type': 'spatial',
                           'factors': sp_tile_sizes,
                           'permutation': permutation,
                           }
                mapping.append(cur_map)
            mapping.append(bypass_map)
            # else:
            #     tp_tile_sizes = self.get_tp_tile_size(tile_sizes)
            #     cur_map = {'target': target,
            #                'type': 'temporal',
            #                'factors': tp_tile_sizes,
            #                'permutation': permutation,
            #                }
            #     mapping.append(cur_map)
        return {'mapping': mapping}

    def get_configs(self, dimension, sol):
        arch = self.arch
        problem = self.get_problem_configs(dimension)
        map = self.get_map_config(sol)
        return arch, problem, map

    def write_config(self, arch, problem, map, arch_path, problem_path, map_path, sparse_path=None):
        with open(arch_path, 'w') as fd:
            yaml.dump(arch, fd)
        fd.close()
        with open(problem_path, 'w') as fd:
            yaml.dump(problem, fd)
        fd.close()
        with open(map_path, 'w') as fd:
            yaml.dump(map, fd)
        fd.close()
        if self.use_sparse:
            with open(sparse_path, 'w') as fd:
                yaml.dump(self.sparse, fd)
            fd.close()

    def dump_timeloop_config_files(self, dimension, sol, out_dir):
        arch, problem, map = self.get_configs(dimension, sol)
        self.write_config(arch, problem, map, arch_path=os.path.join(out_dir, 'arch.yaml'),
                          problem_path=os.path.join(out_dir, 'problem.yaml'), map_path=os.path.join(out_dir, 'map.yaml'),
                          sparse_path=os.path.join(out_dir, 'sparse.yaml'),)

    def run_timeloop(self, dimension,  sol,
                               pool_idx=0, use_IO=False, fitness_obj=['latency']):
        arch, problem, map = self.get_configs(dimension, sol)
        # print("pool_idx  ", pool_idx, sol, map)
        if use_IO:
            self.write_config(arch, problem, map, arch_path=self.arch_path[pool_idx],
                              problem_path=self.problem_path[pool_idx], map_path=self.map_path[pool_idx], sparse_path=self.sparse_path[pool_idx])
            command = [self._executable, self.arch_path[pool_idx], self.problem_path[pool_idx], self.map_path[pool_idx]]
            if self.use_sparse:
                command += [self.sparse_path[pool_idx]]
            process = Popen(command, stdout=PIPE, stderr=PIPE, cwd=self.pool_path[pool_idx])
            stdout, stderr = process.communicate()
            process.wait()
            if stderr:
                print("stderrstderr: ", stderr, sol)
                return [-float('Inf')] * len(fitness_obj)
            else:
                try:
                    stats = parse_timeloop_stats(self.pool_path[pool_idx])
                    # stats = extract_timeloop_perf(self.pool_path[pool_idx])
                    fitness = self.judge_IO(stats, fitness_obj)
                except Exception as e:
                    print("Exception: ", e)
                    fitness = [-float('Inf')] * len(fitness_obj)
                return fitness
        else:
            # cfg = {}
            # cfg.update(arch)
            # cfg.update(map)
            # cfg.update(problem)
            # cfg.update(self.art)
            # cfg.update(self.ert)
            cfg = copy.deepcopy(self.shared_cfg)
            cfg.update(map)
            config = ConfigDict(cfg)
            if not self.debug:
                with stdout_redirected():
                    try:
                        timeloop_app = Model(config,'.')
                        eval_stats = timeloop_app.run()
                        fitness = self.judge(eval_stats, fitness_obj)
                    except:
                        fitness = [-float('Inf')] * len(fitness_obj)
            else:
                # print(sol)
                self.dump_timeloop_config_files(dimension, sol, './report/')
                timeloop_app = Model(config,'.')
                eval_stats = timeloop_app.run()
                fitness = self.judge(eval_stats, fitness_obj)
                # print(fitness)
            return fitness

    def judge_IO(self, stats, fitness_obj='all'):
        if fitness_obj == 'all':
            fitness_obj = ['edp', 'latency', 'energy', 'utilization']
        ret = []

        for f in fitness_obj:
            if f == 'edp':
                ret.append(-stats['cycles'] * stats['energy_pJ'] * 1E-6) # energy_uJ
            if f == 'latency':
                ret.append(-stats['cycles'])
            if f == 'utilization':
                ret.append(stats['utilization'])
            if f == 'energy':
                ret.append(-stats['energy_pJ'] * 1E-6) # energy_uJ
        return ret

    def judge(self, stats, fitness_obj='all'):
        if fitness_obj == 'all':
            return self.get_stats(stats)
        ret = []
        for f in fitness_obj:
            if f == 'edp':
                ret.append(-stats.cycles * stats.energy * 1E-6) # energy_uJ
            if f == 'latency':
                ret.append(-stats.cycles)
            if f == 'area':
                ret.append(-stats.area)
            if f == 'utilization':
                ret.append(-stats.utilization)
            if f == 'energy':
                ret.append(-stats.energy * 1E-6) # energy_uJ
        return ret

    def get_stats(self, stats):
        return [stats.cycles * stats.energy * 1E-6, stats.cycles, stats.energy * 1E-6, stats.utilization, stats.energy/stats.algorithmic_compute, stats.energy/stats.actual_compute, stats.area * 1E-6]
