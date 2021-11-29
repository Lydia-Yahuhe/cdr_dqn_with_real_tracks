"""
Data structure of the input .npz:
the data is save in python dictionary format with keys: 'acs', 'ep_rets', 'rews', 'obs'
the values of each item is a list storing the expert trajectory sequentially
a transition can be: (data['obs'][t], data['acs'][t], data['obs'][t+1]) and get reward data['rews'][t]
"""
import numpy as np


class Dset(object):
    def __init__(self, inputs, randomize):
        self.inputs = inputs

        self.randomize = randomize
        self.num_pairs = len(inputs)
        self.pointer = None

        self.__init_pointer()

    def __init_pointer(self):
        self.pointer = 0
        if self.randomize:
            idx = np.arange(self.num_pairs)
            np.random.shuffle(idx)
            self.inputs = self.inputs[idx]

    def get_next_batch(self, batch_size):
        # if batch_size is negative -> return all
        if batch_size < 0:
            return self.inputs

        if self.pointer + batch_size >= self.num_pairs:
            self.__init_pointer()
        end = self.pointer + batch_size
        inputs = self.inputs[self.pointer:end]
        self.pointer = end
        return inputs
