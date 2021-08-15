# -*- coding: utf-8 -*-
"""
Created on Wed Jul  1 14:24:31 2020

@author: Administrator
"""
from Dueling_DDQN import Dueling_DDQN
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from data_structures.Replay_Buffer import Replay_Buffer
from exploration_strategies.OU_Noise_Exploration import OU_Noise_Exploration
from config import Config
from WMMSE import WMMSE
from Net_module import Net_dueling_DDQN
from Beamspace_channel import beamspace_channel
from data_structures.Prioritised_Replay_Buffer import Prioritised_Replay_Buffer
from data_structures.Beam_Buffer import Beam_Buffer
import scipy.io  as scio 
from unfolder import Unfolder
from complex_matrix import *

class my_dueling_DDQN(Dueling_DDQN):

     def __init__(self, config):       
        self.config = config         
        self.H = None
        self.Hnp = None
        self.Pmax = np.power(10,config.snr/10)
        
        self.selected_beam = None
        self.N = config.beam_num
        self.M = config.M
        self.K = config.K
        self.M_bar = config.M_bar
        self.L = config.path_num
        self.hyperparameters = config.hyperparameters
        self.H_number = config.H_number
        self.Hnp_set = []
        self.rate_list = []
        self.seperate_rate_list = []
        self.joint_rate_list = []
        self.benchmark_rate_list = []
        self.beam_energy = None
        self.H_set = self.create_Hset()
        self.max_sinr = 0
        self.turn_off_exploration = False
        Dueling_DDQN.__init__(self, config)   
        self.memory = Prioritised_Replay_Buffer(self.hyperparameters, config.seed,config.device)
        self.seperate_unfolder = Unfolder(config)        
        self.joint_unfolder = Unfolder(config)
        self.beam_memory = Beam_Buffer(800)
        self.violation_counter = 0
        
        self.store_loss = []
        self.store_rate = []
        self.store_value = []
        self.store_seperate = []
        self.store_joint = []
        self.action_value_temp = torch.zeros(config.action_size)
        self.seperate_unfolder.self_train_network(epoch=1)
        
     def create_NN(self, input_dim, output_dim):   
        net = Net_dueling_DDQN(input_dim, output_dim)
        
        return net
    
     def create_Hset(self):
        """Create channel set"""
        H_set = []
        for i in range(self.H_number):
            Hnp = beamspace_channel(self.M, self.K, self.L)
            self.Hnp_set.append(Hnp)
            tempH = np.zeros([1, 3, self.M, self.K])
            tempH[0, 0, ...] = Hnp.real
            tempH[0, 1, ...] = Hnp.imag
            tempH[0, 2, ...] = np.ones([self.M, self.K])
            """select the biggest M_bar beam"""
            torchH = torch.from_numpy(tempH)
            
            beam_energy = np.sum(np.square(np.abs(Hnp)), 1)
            index = np.argsort(beam_energy)
            
            cut_H = torchH[:,:,index[-self.M_bar:],:]
            
            H = cut_H.float()
            H_set.append(H)
        return H_set
    
     def run_n_episodes(self, num_episodes=None, save_and_print_results=True):
            """Runs game to completion n times and then summarises results and saves model (if asked to)"""
            if num_episodes is None: num_episodes = self.config.num_episodes_to_run            
            while self.episode_number < num_episodes:
              
                self.reset_game()
                self.step()
                self.beam_memory.add_experience(self.selected_beam)
                H = self.convert_beam_format(self.selected_beam)
                self.seperate_rate_list.append(self.seperate_unfolder.forward(H))
                self.joint_rate_list.append(self.joint_unfolder.forward(H))
                
                if self.episode_number % 50 == 0:  
                    print("Epoch:================")
                    print(self.episode_number)
                    temp = np.array(self.loss_array)
                    print(temp.mean())
                    
                    self.store_loss.append(np.array(temp.mean()))
                    
                    self.loss_array = []
                    if save_and_print_results: self.save_and_print_result()
                    
                    print('DRL + unfolding (seperate):')
                    seperate_rate = np.array(self.seperate_rate_list).mean()
                    print(seperate_rate)
                    self.seperate_rate_list= []
                    self.store_seperate.append(np.array(seperate_rate))
                    
                    print('DRL + unfolding (joint):')
                    joint_rate = np.array(self.joint_rate_list).mean()
                    print(joint_rate)
                    self.joint_rate_list= []
                    self.store_joint.append(np.array(joint_rate))
                    
                    
                    
                    self.store_rate.append(np.array(self.rate_list).mean())
                    self.rate_list = [] 
                    
                    
                if self.episode_number % 200 == 0:                    
                    """start training the unfloding network using selected beams"""
                    H_set = self.beam_memory.sample()
                    converted_H_set = []
                    for H in H_set:
                        H = self.convert_beam_format(H)
                        converted_H_set.append(H)
                    self.joint_unfolder.train_network(converted_H_set,epochs=1)
                    
                    
                
                    
                if self.episode_number == 500:
                    self.turn_off_exploration = True
                    print('start joint training =======')
                    
                    
            return
        
     def save_and_print_result(self):
            torch.save(self.q_network_local.state_dict(), 'net_params_256_30dB_DDQN.pkl')
            
     def learn(self):
        """Runs a learning iteration for the Q network after sampling from the replay buffer in a prioritised way"""
        sampled_experiences, importance_sampling_weights = self.memory.sample()
        states, actions, rewards, next_states, dones = sampled_experiences
        loss, td_errors = self.compute_loss_and_td_errors(states, next_states, rewards, actions, dones, importance_sampling_weights)
        self.loss_array.append(loss.item())
        self.take_optimisation_step(self.q_network_optimizer, self.q_network_local, loss, self.hyperparameters["gradient_clipping_norm"])
        self.soft_update_of_target_network(self.q_network_local, self.q_network_target, self.hyperparameters["tau"])
        self.memory.update_td_errors(td_errors.squeeze(1))

     def save_experience(self):
        """Saves the latest experience including the td_error"""
        max_td_error_in_experiences = self.memory.give_max_td_error() + 1e-9
        self.memory.add_experience(max_td_error_in_experiences, self.state, self.action, self.reward, self.next_state, self.done)

     def compute_loss_and_td_errors(self, states, next_states, rewards, actions, dones, importance_sampling_weights):
        """Calculates the loss for the local Q network. It weighs each observations loss according to the importance
        sampling weights which come from the prioritised replay buffer"""
        Q_targets = self.compute_q_targets(next_states, rewards, dones)
        Q_expected = self.compute_expected_q_values(states, actions)
        loss = F.mse_loss(Q_expected, Q_targets)
        loss = loss * importance_sampling_weights
        loss = torch.mean(loss)
        td_errors = Q_targets.data.cpu().numpy() - Q_expected.data.cpu().numpy()
        return loss, td_errors
    
     def reset_game(self):
         """Resets the game information so we are ready to play a new episode"""
         self.state = self.reset_environment()
         self.next_state = None
         self.action = None
         self.reward = None
         self.done = False
         self.episode_rewards = []
         self.episode_log_probabilities = []
         self.episode_step_number = 0
         self.selected_beam = torch.zeros(2,self.N,self.K)

    
     def conduct_action(self):
        """picks the beam selected"""
        self.next_state = self.state.clone()
        self.next_state [0,2,self.action,:] = torch.zeros(1,self.K) 
        self.selected_beam[:,self.episode_step_number-1,:] = self.state[0,0:2,self.action,:]
        self.reward = self.compute_reward()
    
     def compute_reward(self):
        """computes the reward brought by the action"""
        if self.episode_step_number < self.N:
            rate = 0
            reward = self.selected_beam_sinr(self.selected_beam[:,self.episode_step_number-1,:])*0
            reward += self.beam_energy[self.action]/self.beam_energy[-1]*20
            if self.episode_step_number > self.N*0.63:
                reward += self.average_user_energy_reward()

            #reward = self.selected_beam[:,self.episode_step_number-1,:].norm()-self.H.norm()/self.M
            #reward = 0
        else:
            rate = WMMSE(self.selected_beam,self.Pmax,1)
            self.rate_list.append(rate) #save the rate
            if self.episode_number >500:
                H = self.convert_beam_format(self.selected_beam)
                rate = self.joint_unfolder.forward(H)
            reward = rate + self.selected_beam_sinr(self.selected_beam[:,self.episode_step_number-1,:])*0 #- self.average_sinr
            reward += self.beam_energy[self.action] / self.beam_energy[-1] * 20
            reward += self.average_user_energy_reward()
            self.done = True
        if self.state[0,2,self.action,0] == 0:
            reward = rate - 50
            self.violation_counter += 1
        return reward
     
     def convert_beam_format(self, beam):
         beamT = conjT(beam)
         H = beamT.unsqueeze(2)
         return H
         
     def average_user_energy_reward(self):
         abs_total_selected_beam = torch.pow(self.selected_beam[0, 0:self.episode_step_number,:], 2) + torch.pow(self.selected_beam[1, 0:self.episode_step_number, :], 2)
         current_total_user_energy = abs_total_selected_beam.sum(0)
         selected_beam_energy = torch.pow(self.selected_beam[0, self.episode_step_number-1,:], 2) + torch.pow(self.selected_beam[1, self.episode_step_number-1, :], 2)
         division = selected_beam_energy/(current_total_user_energy+0.1)
         rectified_reward = 0
         for i in range(self.K):
             if (current_total_user_energy[i]- selected_beam_energy[i])< 0.2 and division[i] >0.5:
                 rectified_reward += division[i]*6

         return rectified_reward

     def reset_environment(self):
       self.H = self.H_set[self.episode_number % self.H_number]
       self.Hnp = self.Hnp_set[self.episode_number % self.H_number]
       self.beam_energy = self.compute_beam_energy(self.H)
       if self.episode_number<self.H_number:
          self.compute_bench_rate()
       #self.max_sinr = self.compute_max_sinr(self.H)

       return self.H.clone()

     def compute_beam_energy(self,H):
         absH = torch.pow(H[0, 0, :, :], 2) + torch.pow(H[0, 1, :, :], 2)
         beam_energy = absH.sum(1)
         return beam_energy

     def compute_bench_rate(self):
        bench_H = self.H[0,0:2,-self.N:,:]
        bench_rate = WMMSE(bench_H, self.Pmax, 1)
        self.benchmark_rate_list.append(bench_rate)

     def selected_beam_sinr(self,beam):
         sinr = 0
         sum_beam_power = torch.pow(beam.norm(),2)
         for i in range(self.K):
             power = torch.pow(beam[:,i].norm(),2)
             sinr += power/(sum_beam_power - power + 0.1)
         return torch.log2(1+sinr)

     def compute_max_sinr(self,H):
         sinr_list = []
         for i in range(self.M_bar):
             beam_sinr = self.selected_beam_sinr(H[0, 0:2, i, :])
             sinr_list.append(beam_sinr)
         return max(sinr_list)


if __name__ == "__main__":
    config = Config()
    agent = my_dueling_DDQN(config)
    agent.run_n_episodes(num_episodes=config.num_episodes_to_run)