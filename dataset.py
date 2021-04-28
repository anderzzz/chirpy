'''Handle train data

'''
import torch
from torch.utils.data import Dataset

from pydub import AudioSegment

from rawdata import RawDataHandler
from transforms import AudioToTensorTransform, AudioChunkifyTransform, AudioRandomChunkTransform, Compose, \
    AudioAddWhiteNoiseTransform, AudioScaleVolumeTransform, AudioScaleVolumeRelativeMaxTransform

class ChirpyDatasetFileTypeException(Exception):
    pass

class ChirpyDatasetTransformationIncompletenessException(Exception):
    pass

class ChirpyDataset(Dataset):
    '''Bla bla

    '''
    def __init__(self, db_rootdir, subfolder, label_maker, transform, mask=None, force_mono=True):
        super(ChirpyDataset, self).__init__()

        self.rawdata = RawDataHandler(db_rootdir, subfolder)
        self.label_maker = label_maker
        self.transform = transform
        self.force_mono = force_mono

        if mask is None:
            self.mask = [True] * self.rawdata.__len__()
        else:
            self.mask = mask
        self._idxs = [ind for ind, val in enumerate(self.mask) if val]

    def __len__(self):
        return self.mask.count(True)

    def __getitem__(self, item):

        if torch.is_tensor(item):
            item = item.tolist()

        db_item = self.rawdata.get_db_key_(self._idxs[item])
        audio_file_path = self.rawdata.get_audio_file_path_(db_item['catalogue_nr'])
        if audio_file_path.suffix.lower() == '.wav':
            source_type = 'wav'
        elif audio_file_path.suffix.lower() == '.mp3':
            source_type = 'mp3'
        else:
            raise ChirpyDatasetFileTypeException('Inferred file format for file {} not supported'.format(str(audio_file_path)))

        self._audio = AudioSegment.from_file(str(audio_file_path), format=source_type)

        if self.force_mono:
            if self._audio.channels == 1:
                pass
            else:
                self._audio = self._audio.set_channels(1)

        label = self.label_maker(db_item)
        sample = self.transform(self._audio)

        if not isinstance(sample, torch.Tensor):
            raise ChirpyDatasetTransformationIncompletenessException('The audio must at least be transformed into ' + \
                                                                     'a PyTorch Tensor, see ' + \
                                                                     '`transforms.AudioToTensorTransform`')

        return {'label' : label, 'audio' : sample}

from utils import label_maker_factory
def test1():
    transforms = AudioToTensorTransform()
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

def test3():
    transform = AudioRandomChunkTransform(run_time=5000, append_method='silence')
    label_maker = label_maker_factory.create('english name')
    dataset = ChirpyDataset('./test_db', 'audio',
                            label_maker=label_maker,
                            transform=transform)
    for i in range(len(dataset)):
        print (dataset[i])
        raise RuntimeError('Dummy')

def test4():
    transform = Compose([AudioRandomChunkTransform(5000, append_method='silence'), AudioToTensorTransform()])
    label_maker = label_maker_factory.create('english name', label_container_source='test.json')
    dataset = ChirpyDataset('./test_db', 'audio',
                            label_maker=label_maker,
                            transform=transform)
    ss = []
    for i in range(len(dataset)):
        xx = dataset[i]['audio']
        xx = xx.double()
        print (xx.mean(), xx.std())
        ss.append(float(xx.std()))

    print (sum(ss) / len(ss))
    print (label_maker.label_map)
    with open('test.json', 'w') as fout:
        label_maker.to_json(fout)

def test5():
    transform = Compose([AudioRandomChunkTransform(5000, append_method='silence'), AudioScaleVolumeRelativeMaxTransform(0.5), AudioAddWhiteNoiseTransform(-20.0), AudioToTensorTransform()])
    label_maker = label_maker_factory.create('english name')
    dataset = ChirpyDataset('./test_db', 'audio',
                            label_maker=label_maker,
                            transform=transform)
    for i in range(len(dataset)):
        xx = dataset[i]['audio']
        print (xx)

#test5()
