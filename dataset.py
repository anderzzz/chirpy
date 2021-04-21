'''Handle train data

'''
import pandas as pd

import torch
from torch.utils.data import Dataset

from pydub import AudioSegment

from request_train_data import RawDataHandler

class ChirpyDataset(Dataset):
    '''Bla bla

    '''
    def __init__(self, db_rootdir, subfolder):
        super(ChirpyDataset, self).__init__()

        self.rawdata = RawDataHandler(db_rootdir, subfolder)
        self.audio_processor = AudioProcessor()

    def __len__(self):
        return self.rawdata.__len__()

    def __getitem__(self, item):

        if torch.is_tensor(item):
            item = item.tolist()

        db_item = self.rawdata.get_db_key_(item)
        audio_file_path = self.rawdata.get_audio_file_path_(db_item['catalogue_nr'])

        audio_tensor = self.audio_processor.convert(audio_file_path, 'torch')

        raise RuntimeError('BOOOO!')

        return None

class AudioProcessor(object):
    '''Bla bla

    '''
    def __init__(self):
        pass

    def convert(self, path_source_file, target):
        '''Bla bla

        '''
        if target == 'torch':
            return self._convert_torch(path_source_file)

        else:
            raise ValueError('Unknown target type for audio conversion: {}'.format(target))

    def _convert_torch(self, path):
        pass

def test1():
    dataset = ChirpyDataset('./test_db', 'audio')
    for i in range(len(dataset)):
        print (dataset[i])

test1()
