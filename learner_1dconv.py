'''Bla bla

'''
import sys
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch import optim

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
                 f_out=sys.stdout, save_tmp_name='model_in_training',
                 optimizer='SGD', lr=0.001, momentum=0.9, weight_decay=0.0, betas=(0.9,0.999),
                 scheduler_step_size=10, scheduler_gamma=0.1):

        self.inp_f_out = f_out
        self.inp_save_tmp_name = save_tmp_name
        self.inp_loader_batch_size = loader_batch_size
        self.inp_num_workers = num_workers

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

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

        if optimizer == 'SGD':
            self.optimizer = optim.SGD(self.model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
        elif optimizer == 'Adam':
            self.optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay, betas=betas)
        elif optimizer == 'AdamW':
            self.optimizer = optim.AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay, betas=betas)
        else:
            raise ValueError('Unknown optimizer type: {}'.format(optimizer))

        self.lr_scheduler = optim.lr_scheduler.StepLR(self.optimizer,
                                                      step_size=scheduler_step_size,
                                                      gamma=scheduler_gamma)

    def save_model(self, model_path):
        '''Save encoder state dictionary
        Args:
            model_path (str): Path and name to file to save state dictionary to. The filename on disk is this argument
                appended with suffix `.tar`
        '''
        torch.save({'model_state': self.model.state_dict()}, '{}.tar'.format(model_path))

    def train(self, n_epochs):
        '''Bla bla

        '''
        for epoch in range(n_epochs):
            print ('Epoch {}/{}...'.format(epoch + 1, n_epochs), file=self.inp_f_out)

            self.model.train()
            running_loss = 0.0
            for inputs in self.dataloader:
                size_batch = inputs['audio'].size(0)
                audio = inputs['audio'].float().to(self.device)

                self.optimizer.zero_grad()
                mini_classes = self.model(audio)

                loss = self.criterion(mini_classes, inputs['label'])
                loss.backward()
                self.optimizer.step()
                self.lr_scheduler.step()

                running_loss += loss.item() * size_batch

            running_loss = running_loss / float(len(self.dataset))
            self.save_model(self.inp_save_tmp_name)


def test1():
    learner = AudioLearner1DConv(db_rootdir='./db_April26', subfolder='audio', loader_batch_size=2)
    learner.train(1)

if __name__ == '__main__':
    test1()