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

        self.audio_processor(audio_file_path)

        raise RuntimeError('BOOOO!')

        return None

class AudioProcessor(object):
    '''Process the raw input audio file such that it can be integrated with PyTorch

    Currently this wraps methods and objects of the `pydub` library. This library requires the
    command-line tool `ffmpeg` to be installed. If the script is missing, the conversion will
    crash with error. Installation on Mac: `brew install ffmpeg`.

    '''
    def __init__(self):
        self._audio = None

    def __call__(self, path_source_file):
        '''Bla bla

        '''
        if path_source_file.suffix == '.wav':
            source_type = 'wav'
        elif path_source_file.suffix == '.mp3':
            source_type = 'mp3'
        else:
            source_type = None

        self._audio = AudioSegment.from_file(str(path_source_file), format=source_type)
        print (self._audio)

def test1():
    dataset = ChirpyDataset('./test_db', 'audio')
    for i in range(len(dataset)):
        print (dataset[i])

test1()
