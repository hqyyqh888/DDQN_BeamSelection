# -*- coding: utf-8 -*-
"""
Created on Wed Mar  3 21:19:25 2021

@author: yanzhenliu
"""

from collections import namedtuple, deque
import random
import torch
import numpy as np

class Beam_Buffer(object):
    def __init__(self, buffer_size):
        self.memory = deque(maxlen=buffer_size)

    def add_experience(self, beam):
        """Adds experience(s) into the replay buffer"""
        self.memory.append(beam)
   
    def sample(self, num_experiences=100 ):
        """Draws a random sample of experience from the replay buffer"""
        experiences = self.pick_experiences(num_experiences)

        return experiences

    def pick_experiences(self, num_experiences=None):
        if num_experiences is not None: batch_size = num_experiences
        else: batch_size = self.batch_size
        return random.sample(self.memory, k=batch_size)

    def __len__(self):
        return len(self.memory)