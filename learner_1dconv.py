'''Bla bla

'''
import sys
import torch
from torch import nn
from torch.utils.data import DataLoader

from model_1dconv import AudioModel1DAbdoli
from dataset import ChirpyDataset
from transforms import AudioToTensorTransform, AudioDownSampleTransform, AudioRandomChunkTransform, Compose
from ensemble_criterion import MajorityVoter
from rawdata import label_maker_factory

class AudioLearner1DConv(object):
    '''Bla bla

    '''
    def __init__(self,
                 db_rootdir, subfolder,
                 loader_batch_size=16, num_workers=0,
                 f_out=sys.stdout):

        self.inp_f_out = f_out
        self.inp_loader_batch_size = loader_batch_size
        self.inp_num_workers = num_workers

        transform = Compose([AudioDownSampleTransform(16000),
                             AudioRandomChunkTransform(run_time=5000, append_method='cycle'),
                             AudioToTensorTransform()])
        self.dataset = ChirpyDataset(db_rootdir=db_rootdir, subfolder=subfolder,
                                     label_maker=label_maker_factory.create('english name'),
                                     transform=transform)
        self.dataloader = DataLoader(self.dataset, batch_size=self.inp_loader_batch_size,
                                     shuffle=True,
                                     num_workers=self.inp_num_workers)
        self.model = AudioModel1DAbdoli(n_classes=10)
        self.criterion = MajorityVoter(ensemble_size=9)

        self.optimizer = None
        self.lr_scheduler = None

    def train(self, n_epochs):
        '''Bla bla

        '''
        self.model.train()
        for epoch in range(n_epochs):
            print ('Epoch {}/{}...'.format(epoch + 1, n_epochs), file=self.inp_f_out)

            for inputs in self.dataloader:
                print (inputs['audio'].shape)
                mini_classes = self.model(inputs['audio'].float())
                print (mini_classes.shape)
                loss = self.criterion(mini_classes, inputs['label'])
                raise RuntimeError


def test1():
    learner = AudioLearner1DConv(db_rootdir='./db_April26', subfolder='audio', loader_batch_size=2)
    learner.train(1)

if __name__ == '__main__':
    test1()