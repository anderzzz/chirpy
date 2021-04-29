'''Bla bla

'''
import sys

import torch
from torch import nn
from torch.utils.data import DataLoader
from torch import optim

class Learner(object):
    '''Bla bla

    '''
    def __init__(self, data_train, data_test, model, criterion, scheduler,
                 data_key, label_key,
                 loader_batch_size=16, num_workers=0,
                 f_out=sys.stdout, save_tmp_name='model_in_training',
                 optimizer='SGD', lr=0.001, momentum=0.9, weight_decay=0.0, betas=(0.9,0.999),
                 scheduler_step_size=10, scheduler_gamma=0.1):

        self.inp_data_key = data_key
        self.inp_label_key = label_key
        self.inp_f_out = f_out
        self.inp_save_tmp_name = save_tmp_name
        self.inp_loader_batch_size = loader_batch_size
        self.inp_num_workers = num_workers

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.n_data_train = len(data_train)
        self.dataloader_train = DataLoader(dataset=data_train,
                                           batch_size=self.inp_loader_batch_size,
                                           shuffle=True,
                                           num_workers=self.inp_num_workers)
        self.n_data_test = len(data_test)
        self.dataloader_test = DataLoader(dataset=data_test,
                                          batch_size=self.inp_loader_batch_size,
                                          shuffle=False,
                                          num_workers=self.inp_num_workers)
        self.model = model

        if isinstance(criterion, nn.Module):
            self.criterion = criterion
        elif criterion == 'CrossEntropyLoss':
            self.criterion = nn.CrossEntropyLoss()
        elif criterion == 'MSELoss':
            self.criterion = nn.MSELoss()
        else:
            raise ValueError('Unknown criterion type: {}'.format(criterion))

        if isinstance(optimizer, optim.Optimizer):
            self.optimizer = optimizer
        elif optimizer == 'SGD':
            self.optimizer = optim.SGD(self.model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
        elif optimizer == 'Adam':
            self.optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay, betas=betas)
        elif optimizer == 'AdamW':
            self.optimizer = optim.AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay, betas=betas)
        else:
            raise ValueError('Unknown optimizer type: {}'.format(optimizer))

        if isinstance(scheduler, optim.lr_scheduler._LRScheduler):
            self.lr_scheduler = scheduler
        elif scheduler == 'StepLR':
            self.lr_scheduler = optim.lr_scheduler.StepLR(self.optimizer,
                                                          step_size=scheduler_step_size,
                                                          gamma=scheduler_gamma)
        else:
            raise ValueError('Unknown learning-rate scheduler type: {}'.format(scheduler))

    def save_model(self, model_path):
        '''Save encoder state dictionary
        Args:
            model_path (str): Path and name to file to save state dictionary to. The filename on disk is this argument
                appended with suffix `.tar`
        '''
        torch.save({'model_state': self.model.state_dict()}, '{}.tar'.format(model_path))

    def load_model(self, model_path):
        '''Load image classification model from saved state dictionary
        Args:
            model_path (str): Path to the saved model to load
        '''
        saved_dict = torch.load('{}.tar'.format(model_path))
        self.model.load_state_dict(saved_dict['model_state'])

    def train(self, n_epochs):
        '''Bla bla

        '''
        for epoch in range(n_epochs):
            print ('Epoch {}/{}...'.format(epoch + 1, n_epochs), file=self.inp_f_out)

            self.model.train()
            running_loss = 0.0
            for inputs in self.dataloader_train:
                size_batch = inputs[self.inp_data_key].size(0)
                data_in = inputs[self.inp_data_key].float().to(self.device)

                self.optimizer.zero_grad()
                mini_classes = self.model(data_in)

                loss = self.criterion(mini_classes, inputs[self.inp_label_key])
                loss.backward()
                self.optimizer.step()
                self.lr_scheduler.step()

                running_loss += loss.item() * size_batch
                print (loss.item(), size_batch)

            running_loss = running_loss / float(self.n_data_train)
            print ('Training Loss: {}'.format(running_loss))
            self.save_model(self.inp_save_tmp_name)

            self.model.eval()
            running_loss = 0.0
            for inputs in self.dataloader_test:
                size_batch = inputs[self.inp_data_key].size(0)
                audio = inputs[self.inp_data_key].float().to(self.device)

                mini_classes = self.model(audio)
                loss = self.criterion(mini_classes, inputs[self.inp_label_key])

                running_loss += loss.item() * size_batch

            running_loss = running_loss / float(self.n_data_test)
            print ('Testing Loss: {}'.format(running_loss))
