'''Bla bla

'''
from torch import nn
from window import WindowMaker1D

class AudioModel1DAbdoli_16k_8k(nn.Module):
    '''Bla bla

    '''
    window_maker_width = 16000
    window_maker_stride = 8000

    def __init__(self, n_classes):
        super().__init__()

        self.n_classes = n_classes

        conv1d_step1 = nn.Conv1d(in_channels=1, out_channels=16,
                                 kernel_size=64, stride=2, padding=0)
        relu_step1 = nn.ReLU()
        bnorm_step1 = nn.BatchNorm1d(num_features=16)
        pool1d_step1 = nn.MaxPool1d(kernel_size=8, stride=8)
        conv1d_step2 = nn.Conv1d(in_channels=16, out_channels=32,
                                 kernel_size=32, stride=2, padding=0)
        relu_step2 = nn.ReLU()
        bnorm_step2 = nn.BatchNorm1d(num_features=32)
        pool1d_step2 = nn.MaxPool1d(kernel_size=8, stride=8)
        conv1d_step3 = nn.Conv1d(in_channels=32, out_channels=64,
                                 kernel_size=16, stride=2, padding=0)
        relu_step3 = nn.ReLU()
        bnorm_step3 = nn.BatchNorm1d(num_features=64)
        conv1d_step4 = nn.Conv1d(in_channels=64, out_channels=128,
                                 kernel_size=8, stride=2)
        relu_step4 = nn.ReLU()
        bnorm_step4 = nn.BatchNorm1d(num_features=128)
        self.representation_layer = nn.Sequential(conv1d_step1, relu_step1, bnorm_step1, pool1d_step1,
                                                  conv1d_step2, relu_step2, bnorm_step2, pool1d_step2,
                                                  conv1d_step3, relu_step3, bnorm_step3,
                                                  conv1d_step4, relu_step4, bnorm_step4)

        self.aggregation_layer = nn.AvgPool1d(kernel_size=8)

        self.classification_layer = nn.Sequential(nn.Linear(in_features=128, out_features=64),
                                                  nn.ReLU(),
                                                  nn.Linear(in_features=64, out_features=self.n_classes),
                                                  nn.Softmax(dim=1))

        self.window_maker = WindowMaker1D(window_width=self.window_maker_width, stride=self.window_maker_stride)
        self.ensemble_size = self.window_maker.ensemble_size
        self.non_ensemble_size = self.window_maker.non_ensemble_size

    def forward(self, x):
        '''Bla bla

        '''
        x = self.window_maker(x)
        x = x.unsqueeze(1)
        x = self.representation_layer(x)
        x = self.aggregation_layer(x)
        x = x.squeeze(-1)
        c = self.classification_layer(x)

        return c


class AudioModel1DAbdoli_32k_8k(nn.Module):
    '''Bla bla

    '''
    window_maker_width = 32000
    window_maker_stride = 8000

    def __init__(self, n_classes):
        super().__init__()

        self.n_classes = n_classes

        conv1d_step1 = nn.Conv1d(in_channels=1, out_channels=16,
                                 kernel_size=64, stride=2, padding=0)
        relu_step1 = nn.ReLU()
        bnorm_step1 = nn.BatchNorm1d(num_features=16)
        pool1d_step1 = nn.MaxPool1d(kernel_size=8, stride=8)
        conv1d_step2 = nn.Conv1d(in_channels=16, out_channels=32,
                                 kernel_size=32, stride=2, padding=0)
        relu_step2 = nn.ReLU()
        bnorm_step2 = nn.BatchNorm1d(num_features=32)
        pool1d_step2 = nn.MaxPool1d(kernel_size=8, stride=8)
        conv1d_step3 = nn.Conv1d(in_channels=32, out_channels=64,
                                 kernel_size=16, stride=2, padding=0)
        relu_step3 = nn.ReLU()
        bnorm_step3 = nn.BatchNorm1d(num_features=64)
        conv1d_step4 = nn.Conv1d(in_channels=64, out_channels=128,
                                 kernel_size=8, stride=2)
        relu_step4 = nn.ReLU()
        bnorm_step4 = nn.BatchNorm1d(num_features=128)
        conv1d_step5 = nn.Conv1d(in_channels=128, out_channels=256,
                                 kernel_size=4, stride=2)
        relu_step5 = nn.ReLU()
        bnorm_step5 = nn.BatchNorm1d(num_features=256)
        pool1d_step5 = nn.MaxPool1d(kernel_size=4, stride=4)
        self.representation_layer = nn.Sequential(conv1d_step1, relu_step1, bnorm_step1, pool1d_step1,
                                                  conv1d_step2, relu_step2, bnorm_step2, pool1d_step2,
                                                  conv1d_step3, relu_step3, bnorm_step3,
                                                  conv1d_step4, relu_step4, bnorm_step4,
                                                  conv1d_step5, relu_step5, bnorm_step5, pool1d_step5)

        self.aggregation_layer = nn.AvgPool1d(kernel_size=2)

        self.classification_layer = nn.Sequential(nn.Linear(in_features=256, out_features=256),
                                                  nn.ReLU(),
                                                  nn.Linear(in_features=256, out_features=self.n_classes),
                                                  nn.Softmax(dim=1))

        self.window_maker = WindowMaker1D(window_width=self.window_maker_width, stride=self.window_maker_stride)
        self.ensemble_size = self.window_maker.ensemble_size
        self.non_ensemble_size = self.window_maker.non_ensemble_size

    def forward(self, x):
        '''Bla bla

        '''
        x = self.window_maker(x)
        x = x.unsqueeze(1)
        x = self.representation_layer(x)
        x = self.aggregation_layer(x)
        x = x.squeeze(-1)
        c = self.classification_layer(x)

        return c