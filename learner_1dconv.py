'''Bla bla

'''
import sys
import torch
from torch import nn
from torch.utils.data import DataLoader

from model_1dconv import AudioModel1DAbdoli
from dataset import ChirpyDataset, AudioToTensorTransform
from rawdata import label_maker_factory

class AudioLearner1DConv(object):
    '''Bla bla

    '''
    def __init__(self,
                 db_rootdir, subfolder,
                 loader_batch_size=16, num_workers=1,
                 f_out=sys.stdout):

        self.inp_f_out = f_out
        self.inp_loader_batch_size = loader_batch_size
        self.inp_num_workers = num_workers

        transform = [AudioToTensorTransform()]
        self.dataset = ChirpyDataset(db_rootdir=db_rootdir, subfolder=subfolder,
                                     label_maker=label_maker_factory.create('english name'),
                                     transform=transform)
        self.dataloader = DataLoader(self.dataset, batch_size=self.inp_loader_batch_size,
                                     shuffle=True,
                                     num_workers=self.inp_num_workers)
        self.model = AudioModel1DAbdoli()
        self.criterion = nn.MSELoss()

        self.optimizer = None
        self.lr_scheduler = None

    def train(self, n_epochs):
        '''Bla bla

        '''
        self.model.train()
        for epoch in range(n_epochs):
            print ('Epoch {}/{}...'.format(epoch, n_epochs - 1), file=self.inp_f_out)

            for inputs in self.dataloader:
                print (inputs)
                raise RuntimeError


def test1():
    learner = AudioLearner1DConv(db_rootdir='./db_April26', subfolder='audio')
    learner.train(1)