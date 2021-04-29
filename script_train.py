'''Bla bla

'''
import pandas as pd
from torch.utils.data import random_split

from rawdata import RawDataHandler
from transforms import Compose, AudioToTensorTransform, AudioDownSampleTransform, AudioRandomChunkTransform, AudioNormalizeTransform
from dataset import ChirpyDataset
from models import AudioModel1DAbdoli_16k_8k
from ensemble_criterion import MajorityVoter
from learner import Learner

#
# Set constants for training and testing data
#
DB_ROOT = './db_April26'
DB_AUDIO_SUBFOLDER = 'audio'
WELL_SAMPLED_THRS = 30
COL2PREDICT = 'english_name'
AUDIO_RATE = 16000
AUDIO_CHUNK_RUNTIME = 5000
AUDIO_TENSOR_LENGTH = AUDIO_RATE * AUDIO_CHUNK_RUNTIME / 1000
FRAC_TEST = 0.20
BATCH_SIZE = 64

#
# Determine the rows in the database that are adequately sampled
#
df_db = pd.read_csv('{}/{}'.format(DB_ROOT, RawDataHandler.db_file_name))
df_bird_count = df_db.groupby(COL2PREDICT).count().iloc[:,0]
birds_well_sampled = df_bird_count.loc[df_bird_count > WELL_SAMPLED_THRS].index
mask = df_db[COL2PREDICT].isin(birds_well_sampled).to_list()

#
# Create bird to label and label to bird dictionary
#
bird2label = {bird : label for label, bird in enumerate(birds_well_sampled.to_list())}
label2bird = {label : bird for bird, label in bird2label.items()}

#
# Construct a label maker
#
label_maker = lambda db_row_dict : bird2label[db_row_dict[COL2PREDICT]]

#
# Initialize transforms of raw data before use in model
#
transforms = Compose([
    AudioDownSampleTransform(AUDIO_RATE),
    AudioRandomChunkTransform(AUDIO_CHUNK_RUNTIME, append_method='cycle'),
    AudioToTensorTransform('torch.FloatTensor'),
    AudioNormalizeTransform(0.0, 1200.0)
])

#
# Initialize dataset given masks and label_maker
#
dataset = ChirpyDataset(db_rootdir=DB_ROOT, subfolder=DB_AUDIO_SUBFOLDER, label_maker=label_maker,
                        transform=transforms, mask=mask)
n_test = int(FRAC_TEST * len(dataset))
n_train = len(dataset) - n_test
dataset_train, dataset_test = random_split(dataset, [n_train, n_test])

#
# Initialize model to train
#
model = AudioModel1DAbdoli_16k_8k(n_classes=len(birds_well_sampled))

#
# Define criterion
#
if model.non_ensemble_size(AUDIO_TENSOR_LENGTH) != 0:
    print ('Warning: Data settings will create data ensemble for model which does not contain entire audio')
criterion = MajorityVoter(ensemble_size=model.ensemble_size(AUDIO_TENSOR_LENGTH))

#
# Initialize the trainer module
#
train_me = Learner(data_train=dataset_train, data_test=dataset_test,
                   model=model, criterion=criterion,
                   optimizer='SGD', scheduler='StepLR',
                   data_key='audio', label_key='label',
                   loader_batch_size=BATCH_SIZE,
                   lr=0.01)
train_me.train(5)