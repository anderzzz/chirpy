'''Handle train data

'''
import torch
from torch.utils.data import Dataset

from pydub import AudioSegment

class ChirpyDataset(Dataset):
    '''Bla bla

    '''
    def __init__(self, csv_file, audio_root):
        super(ChirpyDataset, self).__init__()

        self.csv_file = csv_file
        self.audio_root = audio_root

    def __len__(self):
        pass

    def __getitem__(self, item):
        pass

    def _create_audio_tensor(self, item):
        '''Bla bla

        '''
        # Insert read of mp3 and conversion into wav and then into array of defined sampling
        pass