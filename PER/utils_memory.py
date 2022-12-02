import random
from typing import (
    Tuple,
)

import torch
import numpy as np
from utils_types import (
    BatchAction,
    BatchDone,
    BatchNext,
    BatchReward,
    BatchState,
    TensorStack5,
    TorchDevice,
)
from utils_sumtree import  SumTree

class ReplayMemory(object):

    def __init__(
            self,
            channels: int,
            capacity: int,
            device: TorchDevice,
    ) -> None:

        self.__device = device
        self.__capacity = capacity


        self.tree = torch.zeros(2 * self.__capacity - 1).to(device)
        self.abs_err_upper = 1.
        self.epsilon = 0.01  # small amount to avoid zero priority
        self.alpha = 0.6  # [0~1] convert the importance of TD error to priority
        self.beta = 0.4  # importance-sampling, from initial value increasing to 1

        self.__size = 0
        self.__pos = 0

        self.__m_states = torch.zeros(
            (capacity, channels, 84, 84), dtype=torch.uint8).to(device)
        self.__m_actions = torch.zeros((capacity, 1), dtype=torch.long).to(device)
        self.__m_rewards = torch.zeros((capacity, 1), dtype=torch.int8).to(device)
        self.__m_dones = torch.zeros((capacity, 1), dtype=torch.bool).to(device)

    def push(
            self,
            folded_state: TensorStack5,
            action: int,
            reward: int,
            done: bool,
    ) -> None:


        self.__m_states[self.__pos] = folded_state
        self.__m_actions[self.__pos, 0] = action
        self.__m_rewards[self.__pos, 0] = reward
        self.__m_dones[self.__pos, 0] = done

        # max_p = torch.max(self.tree[-self.__capacity:])
        max_p = self.abs_err_upper
        # if max_p == 0:
        #     max_p = self.abs_err_upper
        tree_idx = self.__pos + self.__capacity - 1
        self.tree_update(tree_idx, max_p)

        self.__pos += 1
        self.__size = max(self.__size, self.__pos)
        self.__pos %= self.__capacity

        

    def sample(self, batch_size: int) :
        #actions, rewards , dones =  [], [], []
        idxs = torch.zeros((batch_size), dtype=torch.long).to(self.__device)
        indices = torch.zeros((batch_size), dtype=torch.long).to(self.__device)
        segment = self.total_p() / batch_size

        for i in range(batch_size):
            a = segment*i
            b = segment*(i+1)
            s = random.uniform(a,b)
            idx , dataidx =self.get_leaf(s)

            idxs[i] = idx
            indices[i] =dataidx
        
        b_state = self.__m_states[indices, :4].to(self.__device).float()
        b_next = self.__m_states[indices, 1:].to(self.__device).float()
        b_action = self.__m_actions[indices].to(self.__device)
        b_reward = self.__m_rewards[indices].to(self.__device).float()
        b_done = self.__m_dones[indices].to(self.__device).float()

        return idxs, b_state, b_action, b_reward, b_next, b_done

    def update(self, tree_idx, abs_errors):
        abs_errors += self.epsilon  # convert to abs and avoid 0
        clipped_errors = torch.minimum(abs_errors, torch.tensor(self.abs_err_upper))
        ps = torch.pow(clipped_errors, self.alpha)
        ps = ps.reshape(-1)
        for i in range(ps.size(0)):
            self.tree_update(tree_idx[i], ps[i])

    def total_p(self):
        return self.tree[0]  # the root

    def tree_update(self, tree_idx, p):
        change = p - self.tree[tree_idx]
        self.tree[tree_idx] = p
        # then propagate the change through tree
        while tree_idx != 0:    # this method is faster than the recursive loop in the reference code
            tree_idx = (tree_idx - 1) // 2
            self.tree[tree_idx] += change
            
    def __len__(self) -> int:
        return self.__size

    def get_leaf(self, v):
        parent_idx = 0
        while True:     # the while loop is faster than the method in the reference code
            cl_idx = 2 * parent_idx + 1         # this leaf's left and right kids
            cr_idx = cl_idx + 1
            if cl_idx >= len(self.tree):        # reach bottom, end search
                leaf_idx = parent_idx
                break
            else:       # downward search, always search for a higher priority node
                if v <= self.tree[cl_idx]:
                    parent_idx = cl_idx
                else:
                    v -= self.tree[cl_idx]
                    parent_idx = cr_idx

        data_idx = leaf_idx - self.__capacity + 1
        return leaf_idx, data_idx