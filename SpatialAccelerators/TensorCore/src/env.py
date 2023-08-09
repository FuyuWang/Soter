import numpy as np
import torch
import yaml
import os, sys
import copy
import random
from timeloop_env import TimeloopEnv
from multiprocessing.pool import Pool
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import cpu_count
import shutil
from functools import cmp_to_key, partial
from collections import defaultdict, OrderedDict
from utils import timing, is_pareto
import math
import re
import glob
import pickle
from datetime import datetime
import pandas as pd


class Environment(object):
    def __init__(self, in_config_dir='./in_config', arch_file='arch', fitness_obj=['latency'], report_dir='./report',
                 use_pool=True, use_IO=True, log_level=0, debug=False, to_par_RS=False,
                 save_chkpt=False, use_sparse=True, density=None, explore_bypass=False, emulate_random=False,
                 batch_size=4):
        self.debug = bool(debug)
        self.fitness_obj = fitness_obj
        self.dim_note = ['N', 'K', 'C', 'P', 'Q', 'R', 'S']
        self.parallizable_dim_note = ['N', 'K', 'C', 'P', 'Q'] if bool(to_par_RS ) is False else self.dim_note
        self.parallizable_dim_note_set = set(self.parallizable_dim_note)
        self.len_dimension = len(self.dim_note)
        self.timeloop_configfile_path = f'./tmp/out_config_{datetime.now().strftime("%H:%M:%S")}'
        # self.timeloop_configfile_path = f'./out_config'
        self.report_dir = report_dir
        self.use_sparse = use_sparse
        self.explore_bypass = explore_bypass
        self.density = self.get_default_density() if density is None else density
        self.timeloop_env = TimeloopEnv(config_path=self.timeloop_configfile_path, in_config_dir=in_config_dir,
                                        arch_file=arch_file, debug=self.debug,
                                        use_sparse=self.use_sparse, density=self.density)
        self.num_buf_levels = self.timeloop_env.get_num_buffer_levels()
        # print(f'Number of buffer levels: {self.num_buf_levels}')
        self.buffer_size_list = self.timeloop_env.get_buffer_size_list()
        self.buf_spmap_cstr = self.timeloop_env.get_buffer_spmap_cstr()
        self.buffers_with_spmap = list(self.timeloop_env.get_buffers_with_spmap())
        self.dimension, self.dimension_prime, self.prime2idx = self.timeloop_env.get_dimension_primes()
        self.num_primes = len(self.prime2idx.keys())
        # print(self.buf_spmap_cstr, self.buffers_with_spmap, self.buffer_size_list, self.prime2idx)
        self.use_pool = bool(use_pool)
        self.use_IO = bool(use_IO)
        self.log_level = log_level
        self.idealperf = {}
        self.save_chkpt = save_chkpt

        self.best_fitness_record = []
        self.best_latency_record = []
        self.best_energy_record = []
        self.best_sol_record = []
        self.best_fitness = float("-Inf")
        self.best_latency = float("-Inf")
        self.best_energy = float("-Inf")
        self.best_sol = None

        self.emulate_random = emulate_random

        self.set_dimension()
        # self.level_order = [5, 4, 1, 2, 3, 6, 1]
        self.level_order = [2, 3, 1, 4, 2]
        self.start_level_order = 0
        self.cur_buffer_level = self.level_order[self.start_level_order]
        self.steps_per_level = 7
        self.total_steps = self.num_buf_levels * self.steps_per_level
        self.mode = (self.cur_buffer_level - 1) * self.steps_per_level
        self.time_steps = 0
        self.batch_size = batch_size
        self.max_tile = 30

        self.initial_parallel_mask = np.zeros((self.batch_size, self.len_dimension, 2+1), dtype=np.float)
        self.initial_parallel_mask[:, 5:, 1:] = float('-inf')
        self.initial_parallel_mask[:, :, 2] = float('-inf')
        self.initial_order_mask = np.zeros((self.batch_size, self.len_dimension + 1), dtype=np.float)
        self.initial_order_mask[:, -1] = float('inf')

        self.tile_budgets = np.zeros((self.batch_size, self.len_dimension, self.num_primes), dtype=np.int32)
        self.initial_tile_masks = np.zeros((self.batch_size, self.len_dimension, self.num_primes, (self.max_tile+1)*2), dtype=np.float)

        # self.initial_tile2_mask_dict = {'K':None, 'C':None, 'P':None, 'Q':None, 'R':None, 'S':None}
        # self.initial_tile3_mask_dict = {'K':None, 'C':None, 'P':None, 'Q':None, 'R':None, 'S':None}
        # self.initial_tile7_mask_dict = {'K':None, 'C':None, 'P':None, 'Q':None, 'R':None, 'S':None}
        # print("buffers_with_spmap", self.buffers_with_spmap)
        for i, key in enumerate('NKCPQRS'):
            tile_budget = self.dimension_prime[key]
            # print(tile_budget, tile_budget.keys())
            # print(tile_budget['2'], tile_budget['3'], tile_budget['7'])
            # if '2' in tile_budget:
            #     self.tile2_budget[:, i] = tile_budget['2']
            #     self.initial_tile2_mask[:, i, tile_budget['2'] + 1:] = float('-inf')
            # else:
            #     self.tile2_budget[:, i] = 0
            #     self.initial_tile2_mask[:, i, 1:] = float('-inf')
            # print(self.initial_tile2_mask[0, i])
            # self.tile3_budget[:, i] = tile_budget['3']
            # self.initial_tile3_mask[:, i, tile_budget['3'] + 1:] = float('-inf')
            # print(self.initial_tile3_mask[0, i])
            # self.tile5_budget[:, i] = tile_budget['5']
            # self.initial_tile5_mask[:, i, tile_budget['5'] + 1:] = float('-inf')
            # print(self.initial_tile5_mask[0, i])
            # self.tile7_budget[:, i] = tile_budget['7']
            # self.initial_tile7_mask[:, i, tile_budget['7'] + 1:] = float('-inf')
            # print(self.initial_tile7_mask[0, i])
            for k, v in self.prime2idx.items():
                self.tile_budgets[:, i, v] = tile_budget[k]
                self.initial_tile_masks[:, i, v, tile_budget[k] + 1:] = float('-inf')

        self.parallel_mask = copy.deepcopy(self.initial_parallel_mask)
        self.order_mask = copy.deepcopy(self.initial_order_mask)

        # self.tile2_remain_budget = copy.deepcopy(self.tile2_budget)
        # self.tile3_remain_budget = copy.deepcopy(self.tile3_budget)
        # self.tile5_remain_budget = copy.deepcopy(self.tile5_budget)
        # self.tile7_remain_budget = copy.deepcopy(self.tile7_budget)
        # self.tile2_mask = copy.deepcopy(self.initial_tile2_mask)
        # self.tile3_mask = copy.deepcopy(self.initial_tile3_mask)
        # self.tile5_mask = copy.deepcopy(self.initial_tile5_mask)
        # self.tile7_mask = copy.deepcopy(self.initial_tile7_mask)
        self.tile_remain_budgets = copy.deepcopy(self.tile_budgets)
        self.tile_masks = copy.deepcopy(self.initial_tile_masks)

        length = self.num_primes * 2 + 2
        self.initial_trg_seq = np.zeros((self.batch_size, self.total_steps + 1, length), dtype=np.int32)
        self.initial_trg_seq[:, 0, 0] = self.steps_per_level
        self.initial_trg_seq[:, 0, 1: self.num_primes + 1] = self.max_tile
        self.initial_trg_seq[:, 0, self.num_primes + 1] = 2
        self.initial_trg_seq[:, 0, self.num_primes + 2: self.num_primes + 2 + self.num_primes] = self.max_tile

        for i in range(self.num_buf_levels):
            for j in range(self.steps_per_level):
                self.initial_trg_seq[:, i * self.steps_per_level + j + 1, 0] = j
                self.initial_trg_seq[:, i * self.steps_per_level + j + 1, self.num_primes + 1] = 0
                self.initial_trg_seq[:, i * self.steps_per_level + j + 1, self.num_primes + 2:] = 0
                if i == self.num_buf_levels - 1:
                    self.initial_trg_seq[:, i * self.steps_per_level + j + 1, 1: self.num_primes + 1] = self.tile_budgets[:, j]
                    # self.initial_trg_seq[:, i * self.steps_per_level + j + 1, 2] = self.tile3_budget[:, j]
                    # self.initial_trg_seq[:, i * self.steps_per_level + j + 1, 3] = self.tile5_budget[:, j]
                    # self.initial_trg_seq[:, i * self.steps_per_level + j + 1, 4] = self.tile7_budget[:, j]
                else:
                    self.initial_trg_seq[:, i * self.steps_per_level + j + 1, 1: self.num_primes + 1] = 0

        self.min_reward = None

        self.trg_seq = copy.deepcopy(self.initial_trg_seq[:, :1, :])
        self.trg_seq_disorder = copy.deepcopy(self.initial_trg_seq[:, 1:, :])
        self.final_trg_seq = copy.deepcopy(self.initial_trg_seq[:, 1:, :])

    def get_default_density(self):
        density = {'Weights': 1,
                   'Inputs': 1,
                   'Outputs': 1}
        return density

    def set_dimension(self):
        self.idealperf['edp'], self.idealperf['latency'], self.idealperf['energy'] = self.timeloop_env.get_ideal_perf(self.dimension)
        self.idealperf['utilization'] = 1

        self.best_fitness_record = []
        self.best_latency_record = []
        self.best_energy_record = []
        self.best_sol_record = []
        self.best_fitness = float("-Inf")
        self.best_latency = float("-Inf")
        self.best_energy = float("-Inf")
        self.best_sol = None

    def reset(self):
        self.start_level_order = 0
        self.cur_buffer_level = self.level_order[self.start_level_order]
        self.mode = (self.cur_buffer_level - 1) * self.steps_per_level
        self.time_steps = 0
        self.order_mask = copy.deepcopy(self.initial_order_mask)
        self.tile_remain_budgets = copy.deepcopy(self.tile_budgets)
        self.tile_masks = copy.deepcopy(self.initial_tile_masks)

        self.parallel_mask = copy.deepcopy(self.initial_parallel_mask)
        if f'l{self.cur_buffer_level}' not in self.buffers_with_spmap:
            self.parallel_mask[:, :, 1:] = float('-inf')
        else:
            self.parallel_mask = copy.deepcopy(self.initial_parallel_mask)

        order_mask = copy.deepcopy(self.order_mask)
        tile_remain_budgets = copy.deepcopy(self.tile_remain_budgets)
        tile_masks = copy.deepcopy(self.tile_masks[:, :, :, 0:self.max_tile + 1])
        parallel_mask = copy.deepcopy(self.parallel_mask)

        self.trg_seq = copy.deepcopy(self.initial_trg_seq[:, :1, :])
        self.trg_seq_disorder = copy.deepcopy(self.initial_trg_seq[:, 1:, :])
        self.final_trg_seq = copy.deepcopy(self.initial_trg_seq[:, 1:, :])

        trg_seq = copy.deepcopy(self.trg_seq)
        trg_seq_disorder = copy.deepcopy(self.trg_seq_disorder)

        length = self.num_primes * 2 + 2
        trg_mask = np.zeros((self.batch_size, self.total_steps + 1, length), dtype=np.float)
        trg_mask[:, 0, :] = 1.

        return trg_seq, trg_mask, order_mask, tile_remain_budgets, tile_masks, parallel_mask, self.mode, \
            self.cur_buffer_level, trg_seq_disorder

    def get_reward(self, final_trg_seq):
        if self.use_pool:
            pool = ProcessPoolExecutor(self.batch_size)
            self.timeloop_env.create_pool_env(num_pools=self.batch_size, dimension=self.dimension,
                                              sol=final_trg_seq[0, :, :], use_IO=self.use_IO)
        else:
            pool = None
            self.timeloop_env.create_pool_env(num_pools=1, dimension=self.dimension, sol=final_trg_seq[0, :, :],
                                              use_IO=self.use_IO)

        reward = self.evaluate(final_trg_seq[:, :, :], pool)

        return reward

    def step(self, actions):
        if self.cur_buffer_level < self.num_buf_levels:
            order_action, tile_actions, parallel_action, sp_tile_actions = actions
            order_action = order_action.cpu().numpy()
            tile_actions = tile_actions.cpu().numpy()
            parallel_action = parallel_action.cpu().numpy()
            sp_tile_actions = sp_tile_actions.cpu().numpy()
            length = self.num_primes * 2 + 2
            cur_seq = np.zeros((self.batch_size, length), dtype=np.int32)
            cur_seq[:, 0] = order_action
            cur_seq[:, 1: self.num_primes + 1] = tile_actions
            cur_seq[:, self.num_primes + 1] = parallel_action
            cur_seq[:, self.num_primes + 2:] = sp_tile_actions

            self.trg_seq = np.concatenate((self.trg_seq, np.expand_dims(cur_seq, 1)), axis=1)

            seq_disorder_ind = (self.cur_buffer_level - 1) * self.steps_per_level + order_action
            self.trg_seq_disorder[np.arange(self.batch_size), seq_disorder_ind, 1: self.num_primes + 1] = tile_actions
            self.trg_seq_disorder[np.arange(self.batch_size), seq_disorder_ind, self.num_primes + 1] = parallel_action
            self.trg_seq_disorder[np.arange(self.batch_size), seq_disorder_ind, self.num_primes + 2:] = sp_tile_actions

            self.final_trg_seq[np.arange(self.batch_size), self.mode, 0] = order_action
            self.final_trg_seq[np.arange(self.batch_size), self.mode, 1: self.num_primes + 1] = tile_actions
            self.final_trg_seq[np.arange(self.batch_size), self.mode, self.num_primes + 1] = parallel_action
            self.final_trg_seq[np.arange(self.batch_size), self.mode, self.num_primes + 2:] = sp_tile_actions
            # print(seq_disorder_ind, self.mode)

            self.order_mask[np.arange(self.batch_size), order_action] = float('inf')
            # if f'l{self.cur_buffer_level}' not in self.buffers_with_spmap:
            #     self.parallel_mask[:, :, 1:] = float('-inf')
            self.tile_remain_budgets[np.arange(self.batch_size), order_action] -= tile_actions
            tile_remain_budgets = self.tile_remain_budgets[np.arange(self.batch_size), order_action]
            for i in range(1, self.max_tile+1):
                for j in range(0, self.num_primes):
                    tile_remain_budget = tile_remain_budgets[:, j]
                    self.tile_masks[np.arange(self.batch_size), order_action, j, tile_remain_budget + i] = float('-inf')
        else:
            order_action, _, _, _, _ = actions
            order_action = order_action.cpu().numpy()
            tile_actions = self.tile_remain_budgets[np.arange(self.batch_size), order_action]
            parallel_action = np.zeros(self.batch_size, dtype=np.int32)

            length = self.num_primes * 2 + 2
            cur_seq = np.zeros((self.batch_size, length), dtype=np.int32)
            cur_seq[:, 0] = order_action
            cur_seq[:, 1: self.num_primes + 1] = tile_actions
            cur_seq[:, self.num_primes + 1] = parallel_action
            cur_seq[:, self.num_primes + 2:] = 0
            self.trg_seq = np.concatenate((self.trg_seq, np.expand_dims(cur_seq, 1)), axis=1)

            seq_disorder_ind = (self.cur_buffer_level - 1) * self.steps_per_level + order_action
            self.trg_seq_disorder[np.arange(self.batch_size), seq_disorder_ind, 1: self.num_primes + 1] = tile_actions
            self.trg_seq_disorder[np.arange(self.batch_size), seq_disorder_ind, self.num_primes + 1] = parallel_action
            self.trg_seq_disorder[np.arange(self.batch_size), seq_disorder_ind, self.num_primes + 2:] = 0
            self.final_trg_seq[np.arange(self.batch_size), self.mode, 0] = order_action
            self.final_trg_seq[np.arange(self.batch_size), self.mode, 1: self.num_primes + 1] = tile_actions
            self.final_trg_seq[np.arange(self.batch_size), self.mode, self.num_primes + 1] = parallel_action
            self.final_trg_seq[np.arange(self.batch_size), self.mode, self.num_primes + 2:] = 0

            self.order_mask[np.arange(self.batch_size), order_action] = float('inf')

        self.time_steps += 1
        self.mode += 1
        if self.mode % self.steps_per_level == 0:
            self.start_level_order += 1
            self.cur_buffer_level = self.level_order[self.start_level_order]
            self.mode = (self.cur_buffer_level - 1) * self.steps_per_level
            self.order_mask = copy.deepcopy(self.initial_order_mask)

        if f'l{self.cur_buffer_level}' not in self.buffers_with_spmap:
            self.parallel_mask[:, :, 1:] = float('-inf')
        else:
            self.parallel_mask = copy.deepcopy(self.initial_parallel_mask)

        order_mask = copy.deepcopy(self.order_mask)
        tile_remain_budgets = copy.deepcopy(self.tile_remain_budgets)
        tile_masks = copy.deepcopy(self.tile_masks[:, :, :, 0:self.max_tile + 1])
        parallel_mask = copy.deepcopy(self.parallel_mask)

        trg_seq = copy.deepcopy(self.trg_seq)
        trg_seq_disorder = copy.deepcopy(self.trg_seq_disorder)
        length = self.num_primes * 2 + 2
        trg_mask = np.zeros((self.batch_size, self.total_steps + 1, length), dtype=np.float)
        trg_mask[:, 0, :] = 1.

        if self.time_steps < self.total_steps:
            done = 0
            info = None
            reward = np.zeros(self.batch_size)
        else:
            done = 1
            info = None
            reward_saved = copy.deepcopy(self.get_reward(self.final_trg_seq))
            # reward_saved[reward_saved==float('-inf')] = self.min_reward
            # sort_idx = np.argsort(reward_saved)
            # top_k_idx = sort_idx[int(self.batch_size / 4) - 1]
            # reward = (reward_saved - reward_saved[top_k_idx])
            reward_saved[reward_saved==float('-inf')] = float('inf')
            if self.min_reward is None:
                self.min_reward = reward_saved.min()
            else:
                self.min_reward = min(self.min_reward, reward_saved.min())
            reward_saved[reward_saved == float('inf')] = self.min_reward
            reward = reward_saved - self.min_reward
            # reward_saved[reward_saved==float('inf')] = reward_saved.min()
            # reward = (reward_saved - reward_saved.min()) / (reward_saved.std() + 1e-12)
            # self.last_reward = reward_saved

        return (trg_seq, trg_mask, order_mask, tile_remain_budgets, tile_masks, parallel_mask,
                self.mode, self.cur_buffer_level, trg_seq_disorder), reward, done, info

    def thread_fun(self, args, fitness_obj=None):
        sol, pool_idx = args
        fit = self.timeloop_env.run_timeloop(self.dimension, sol, pool_idx=pool_idx, use_IO=self.use_IO,
                                             fitness_obj=fitness_obj if fitness_obj is not None else self.fitness_obj)
        return fit

    def evaluate(self, sols, pool):
        fitness = np.ones((self.batch_size, len(self.fitness_obj))) * np.NINF
        if not pool:
            for i, sol in enumerate(sols):
                fit = self.thread_fun((sol, 0))
                fitness[i] = fit
        else:
            while(1):
                try:
                    fits = list(pool.map(self.thread_fun, zip(sols, np.arange(len(sols)))))
                    for i, fit in enumerate(fits):
                        fitness[i] = fit
                    break
                except Exception as e:
                    if self.log_level>2:
                        print(type(e).__name__, e)
                    pool.shutdown(wait=False)
                    pool = ProcessPoolExecutor(self.batch_size)

        latency_fitness = fitness[:, 1]
        energy_fitness = fitness[:, 2]
        fitness = fitness[:, 0]
        best_idx = np.argmax(fitness)
        if fitness[best_idx] > self.best_fitness:
            self.best_fitness = fitness[best_idx]
            self.best_latency = latency_fitness[best_idx]
            self.best_energy = energy_fitness[best_idx]
            self.best_sol = sols[best_idx]
            self.create_timeloop_report(self.best_sol, self.report_dir)
        print("Achieved Fitness: ", self.best_fitness, self.mode, fitness[best_idx], (fitness > float('-inf')).sum())
        return fitness

    def create_timeloop_report(self, sol, dir_path):
        fitness = self.thread_fun((sol, 0))
        stats = self.thread_fun((sol, 0), fitness_obj='all')
        os.makedirs(dir_path, exist_ok=True)
        columns = ['EDP (uJ cycles)', 'Cycles', 'Energy (pJ)', 'Utilization', 'pJ/Algorithm-Compute', 'pJ/Actual-Compute', 'Area (mm2)'][:len(stats)]
        if self.use_IO is False:
            self.timeloop_env.dump_timeloop_config_files(self.dimension, sol, dir_path)
        else:
            os.system(f'cp -d -r {os.path.join(self.timeloop_configfile_path, "pool-0")}/* {dir_path}')
        with open(os.path.join(dir_path,'RL-Timeloop.txt'), 'w') as fd:
            value = [f'{v:.5e}' for v in fitness]
            fd.write(f'Achieved Fitness: {value}\n')
            fd.write(f'Statistics\n')
            fd.write(f'{columns}\n')
            fd.write(f'{stats}')
        stats = np.array(stats).reshape(1, -1)
        df = pd.DataFrame(stats, columns=columns)
        df.to_csv(os.path.join(dir_path,'Gamma-Timeloop.csv'))

    def record_chkpt(self, write=False):
        self.best_fitness_record.append(self.best_fitness)
        self.best_latency_record.append(self.best_latency)
        self.best_energy_record.append(self.best_energy)
        self.best_sol_record.append(self.best_sol)

        chkpt = None
        if write:
            with open(os.path.join(self.report_dir, 'env_chkpt.plt'), 'wb') as fd:
                chkpt = {
                    'best_fitness_record': self.best_fitness_record,
                    'best_latency_record': self.best_latency_record,
                    'best_energy_record': self.best_energy_record,
                    'best_sol_record': self.best_sol_record,
                    'best_fitness': self.best_fitness,
                    'best_latency': self.best_latency,
                    'best_energy': self.best_energy,
                    'best_sol': self.best_sol
                }
                pickle.dump(chkpt, fd)
        return chkpt

    def clean_timeloop_output_files(self):
        shutil.rmtree(self.timeloop_configfile_path)
        out_prefix = "./timeloop-model."
        output_file_names = []
        output_file_names.append( "tmp-accelergy.yaml")
        output_file_names.append(out_prefix + "accelergy.log")
        output_file_names.extend(glob.glob("*accelergy.log"))
        output_file_names.extend(glob.glob("*tmp-accelergy.yaml"))
        output_file_names.append(out_prefix + ".log")
        output_file_names.append(out_prefix + "ART.yaml")
        output_file_names.append(out_prefix + "ART_summary.yaml")
        output_file_names.append(out_prefix + "ERT.yaml")
        output_file_names.append(out_prefix + "ERT_summary.yaml")
        output_file_names.append(out_prefix + "flattened_architecture.yaml")
        output_file_names.append(out_prefix + "map+stats.xml")
        output_file_names.append(out_prefix + "map.txt")
        output_file_names.append(out_prefix + "stats.txt")
        for f in output_file_names:
            if os.path.exists(f):
                os.remove(f)








