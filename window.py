'''Bla bla

'''
import math

class WindowMaker1D(object):
    '''Bla bla

    '''
    def __init__(self, window_width, stride):
        self.window_width = window_width
        self.stride = stride

    def __call__(self, x):
        '''Bla bla

        '''
        if x.shape[1] < self.window_width:
            return []

        x = x.unfold(1, self.window_width, self.stride)
        x = x.reshape(-1, self.window_width)

        return x

    def ensemble_size(self, length):
        return max(0, math.floor(1 + (length - self.window_width) / self.stride))

    def non_ensemble_size(self, length):
        return (length - self.window_width) % self.stride