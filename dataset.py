'''Handle train data

'''
import torch
from torch.utils.data import Dataset

from pydub import AudioSegment

from request_train_data import RawDataHandler, label_maker_factory
from transforms import AudioTo1DTensor, AudioChunkifyTransform

class ChirpyDatasetFileTypeException(Exception):
    pass

class ChirpyDataset(Dataset):
    '''Bla bla

    '''
    def __init__(self, db_rootdir, subfolder, label_maker, transform):
        super(ChirpyDataset, self).__init__()

        self.rawdata = RawDataHandler(db_rootdir, subfolder)
        self.label_maker = label_maker
        self.transform = transform

    def __len__(self):
        return self.rawdata.__len__()

    def __getitem__(self, item):

        if torch.is_tensor(item):
            item = item.tolist()

        db_item = self.rawdata.get_db_key_(item)
        audio_file_path = self.rawdata.get_audio_file_path_(db_item['catalogue_nr'])
        if audio_file_path.suffix == '.wav':
            source_type = 'wav'
        elif audio_file_path.suffix == '.mp3':
            source_type = 'mp3'
        else:
            raise ChirpyDatasetFileTypeException('Inferred file format for file {} not supported'.format(str(audio_file_path)))

        self._audio = AudioSegment.from_file(str(audio_file_path), format=source_type)

        label = self.label_maker(db_item)
        sample = self.transform(self._audio)

        return {'label' : label, 'audio' : sample}

def test1():
    transforms = AudioTo1DTensor()
    label_maker = label_maker_factory.create('english name')
    dataset = ChirpyDataset('./test_db', 'audio',
                            label_maker=label_maker,
                            transform=transforms)
    for i in range(len(dataset)):
        print (dataset[i])
        raise RuntimeError('DUMP!')

def test2():
    transform = AudioChunkifyTransform(run_time=5000, method='pad', strict=False)
    label_maker = label_maker_factory.create('english name')
    dataset = ChirpyDataset('./test_db', 'audio',
                            label_maker=label_maker,
                            transform=transform)
    for i in range(len(dataset)):
        print (dataset[i])
        raise RuntimeError('Dummy')


test2()
