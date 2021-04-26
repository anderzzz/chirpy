'''Script to request train data from web

'''
import requests
import urllib.parse
from pathlib import Path
import pandas as pd
import json
from time import sleep

from dataclasses import dataclass
from typing import Dict, List

from datetime import datetime
from copy import deepcopy

class WebXenoCantoException(Exception):
    pass

class RawDataHandlerError(Exception):
    pass

class LabelMakerMissingDataNameException(Exception):
    pass

class LabelMakerInvalidKeyException(Exception):
    pass

class LabelMakerMissingKeySourceException(Exception):
    pass

@dataclass
class WebConsts:
    base_url : str
    entity_name : str
    query : str
    query_args : Dict
    api_help_url : str

@dataclass
class XenoCantoRecordingConsts:
    data_names : Dict
    id_key : str
    recordings_key : str = 'recordings'
    url_description : str = 'https://www.xeno-canto.org/explore/api'

    @property
    def data_names_inv(self):
        return {v: k for k, v in self.data_names.items()}

class WebXenoCanto(object):
    '''Handler for audio data requested from Xeno Canto web page

    Args:
        TBD

    '''
    def __init__(self, web_consts, since=None):

        self.web_consts = web_consts
        if web_consts.base_url[-1] == '/':
            self.web_consts.base_url = web_consts.base_url[:-1]

        self.query_base = {}
        if not since is None:
            try:
                datetime.strptime(since, '%Y-%m-%d')
            except ValueError:
                raise ValueError("Incorrect data format, should be YYYY-MM-DD")

            self.query_base[self.web_consts.query_args['uploaded_since']] = since
        self.reset_payload()

    def __len__(self):
        return len(self.payloads)

    def n_entries_payloads(self):
        return list(map(len, self.payloads))

    def _get(self, url, params):
        return requests.get(url, params)

    def _get_records(self, params):
        return self._get('{}/{}'.format(self.web_consts.base_url,
                                        self.web_consts.entity_name),
                         params)

    def reset_payload(self):
        '''Bla bla

        '''
        self.payload = None
        self.payloads = []

    def get(self):
        '''Bla bla

        '''
        # The parameters contain characters that normally are turned into percent encoding that do not parse
        # right at Xeno Canto server. Hence the urlencode is used to make the parameter string
        response = self._get_records(
            urllib.parse.urlencode({self.web_consts.query : self.query_string}, safe=':+&=')
        )

        if response.status_code == 200:
            self.payload = response.json()
            self.payloads.append(self.payload)
            return self
        else:
            raise WebXenoCantoException('Exception in database get for command {}. Code: {}'.format(response.url, response.status_code))

    def get_all(self):
        '''Bla bla

        '''
        self.get()
        if self.payload['page'] < self.payload['numPages']:
            self.query(page=self.payload['page'] + 1, **self.current_query_kwargs).get_all()

        return self

    def query(self, page=None, **kwargs):
        '''Bla bla

        '''
        self.current_query_kwargs = kwargs

        query_payload = deepcopy(self.query_base)
        for key, val in kwargs.items():
            query_payload[self.web_consts.query_args[key]] = val

        self.query_string = ''
        for k, v in query_payload.items():
            self.query_string += '{}:{}+'.format(k, v)
        else:
            self.query_string = self.query_string[:-1]

        if not page is None:
            self.query_string += '&page={}'.format(page)

        return self


class RawDataHandler(object):
    '''Bla bla

    '''
    db_file_name = 'db.csv'

    def __init__(self, db_rootdir, subfolder, payload_consts=None, start_clean=False):

        self.db_rootdir = db_rootdir
        self.audio_subfolder = subfolder
        self.payload_consts = payload_consts

        self.db_audiodir = '{}/{}'.format(self.db_rootdir, self.audio_subfolder)
        Path(self.db_rootdir).mkdir(parents=True, exist_ok=True)
        Path(self.db_audiodir).mkdir(parents=True, exist_ok=True)

        self.db_file = self.db_rootdir + '/' + self.db_file_name
        if start_clean:
            Path(self.db_file).unlink(missing_ok=True)

    def __len__(self):
        with open(self.db_file) as fin:
            return sum(1 for _ in fin) - 1

    def get_db_key_(self, k_row):
        '''Bla bla

        '''
        df = pd.read_csv(self.db_file, nrows=1, skiprows=list(range(1, k_row + 1)), header=0)
        if df.shape[0] != 1:
            raise RawDataHandlerError('Row number {} does not return a single row. Corrupted database?'.format(k_row))
        return df.to_dict(orient='records')[0]

    def get_audio_file_path_(self, catalogue_nr):
        '''Return the file path to the audio file associated with the given catalogue number

        Args:
            catalogue_nr (int): The catalogue number for which to extract audio file path

        Returns:
            path (PosixPath): Path to the audio file, represented as `pathlib.PosixPath` object

        '''
        file = sorted(Path(self.db_audiodir).glob('{}.*'.format(catalogue_nr)))
        if len(file) != 1:
            raise RawDataHandlerError('Audio file for catalogue number {} returns {} files rather than one file'.format(catalogue_nr, len(file)))
        return file[0]

    def populate_metadata(self, payload, col_subset=None):
        '''Bla bla

        '''
        if col_subset is None:
            datacols = list(self.payload_consts.data_names.values())
        else:
            datacols = [self.payload_consts.data_names[key] for key in col_subset]

        df_payload = pd.DataFrame(payload[self.payload_consts.recordings_key], columns=datacols)
        df_payload = df_payload.rename(columns=self.payload_consts.data_names_inv).set_index(self.payload_consts.id_key)

        if not Path(self.db_file).exists():
            df_payload.to_csv(self.db_file)
        else:
            with open(self.db_file, 'a') as fcsv:
                data_str = df_payload.to_csv(header=False)
                fcsv.write(data_str)

    def deduplicate_db(self):
        '''Bla bla

        '''
        raise NotImplementedError('De-duplication not implemented yet')

    def download_audio(self, payload, sleep_seconds=None):
        '''Bla bla

        '''
        for rec in payload[self.payload_consts.recordings_key]:
            url_partial = rec[self.payload_consts.data_names['file_url']]
            url = 'https:{}'.format(url_partial)
            r = requests.get(url, allow_redirects=True)

            id_key = rec[self.payload_consts.data_names['catalogue_nr']]
            suffix = rec[self.payload_consts.data_names['file_name_original']].split('.')[-1]
            audio_file_name = '{}/{}.{}'.format(self.db_audiodir, id_key, suffix)

            if Path(audio_file_name).exists():
                print('Warning! Audio file {} exists, and is overwritten'.format(audio_file_name))

            with open(audio_file_name, 'wb') as f_audio_file:
                f_audio_file.write(r.content)

            if not sleep_seconds is None:
                sleep(sleep_seconds)

#
# Classes to map the semantics of a database entry to a numeric label that is to be
# predicted. A factory method implementation is used to make it easier to extend
#

class LabelMakerByOne(object):
    '''Bla bla

    '''
    def __init__(self, data_name, label_container_source=None, augment_source=True):
        if not data_name in web_xeno_canto_payload_consts.data_names.keys():
            raise LabelMakerInvalidKeyException('The label key {} is not part of the data name constants'.format(data_name))
        self.data_name = data_name

        if label_container_source is None:
            self.label_container = {}
            self.label_max = 0

        else:
            with open(label_container_source) as fin:
                self.label_container = json.load(fin)['key_to_label']
                self.label_max = len(self.label_container)

        self.augment_source = augment_source

    def __call__(self, db_item):
        '''Bla bla

        '''
        try:
            data_val = db_item[self.data_name]
        except KeyError:
            raise LabelMakerMissingDataNameException('Did not find {} in database item'.format(self.data_name))

        try:
            label = self.label_container[data_val]
        except KeyError:
            if self.augment_source:
                label = self.label_max
                self.label_container[data_val] = label
                self.label_max += 1
            else:
                raise LabelMakerMissingKeySourceException('The key {} not assigned label in source, and augmentation set as `False`.'.format(data_val))

        return label

    def to_json(self, fout):
        out = {'key_to_label' : self.label_container, 'date_created' : str(datetime.now())}
        json.dump(out, fout)

    @property
    def label_map(self):
        return self.label_container

    @property
    def label_map_inv(self):
        return {v : k for k, v in self.label_container.items()}

class LabelMakerByOneBuilder(object):
    '''Bla bla

    '''
    def __init__(self, data_name):
        self._instance = None
        self.data_name = data_name

    def __call__(self, **kwargs):
        self._instance = LabelMakerByOne(data_name=self.data_name, **kwargs)
        return self._instance

class LabelMakerFactory(object):
    '''Bla bla

    '''
    def __init__(self):
        self._builders = {}

    def register_builder(self, key, builder):
        '''Register a builder
        Args:
            key (str): Key to the builder, which can be invoked by `create` method
            builder: A Fungi Data Builder instance
        '''
        self._builders[key] = builder

    @property
    def keys(self):
        return self._builders.keys()

    def create(self, key, **kwargs):
        '''Method to create a fungi data set through a uniform interface
        Args:

        '''
        try:
            builder = self._builders[key]
        except KeyError:
            raise ValueError('Unregistered data builder: {}'.format(key))
        return builder(**kwargs)

label_maker_factory = LabelMakerFactory()
label_maker_factory.register_builder('english name', LabelMakerByOneBuilder(data_name='english_name'))

#
# Constants that embody knowledge of how Xeno Canto exposes data and returns it. If
# Xeno Canto changes something on their side, these constants (and only this?)
# need to change
#

web_xeno_canto_consts = WebConsts(
    base_url='https://www.xeno-canto.org/api/2',
    entity_name='recordings',
    query='query',
    query_args={'country' : 'cnt',
                'sound_type' : 'type',
                'recording_license' : 'lic',
                'recording_quality_equal' : 'q',
                'recording_quality_gt' : 'q_gt',
                'uploaded_since' : 'since'},
    api_help_url='https://www.xeno-canto.org/help/search'
)

web_xeno_canto_payload_consts = XenoCantoRecordingConsts(
    id_key = 'catalogue_nr',
    data_names = {
        'catalogue_nr' : 'id',
        'generic_name' : 'gen',
        'specific_name' : 'sp',
        'subspecies_name' : 'ssp',
        'english_name' : 'en',
        'country_recorded' : 'cnt',
        'location_name' : 'loc',
        'latitude' : 'lat',
        'longitude' : 'lng',
        'sound_type' : 'type',
        'recording_details_url' : 'url',
        'file_url' : 'file',
        'file_name_original' : 'file-name',
        'license' : 'lic',
        'quality_rating' : 'q',
        'length_recording' : 'length',
        'date_recording' : 'date',
        'time_of_day_recording' : 'time',
        'bird_seen' : 'bird-seen'
    }
)

def test1():
    xeno = WebXenoCanto(web_xeno_canto_consts)
    xeno.query(country='iceland', recording_quality_equal='A').get()

def test2():
    xeno = WebXenoCanto(web_xeno_canto_consts)
    xeno.query(country='iceland', recording_quality_gt='C').get_all()

def test3():
    xeno = WebXenoCanto(web_xeno_canto_consts)
    xeno.query(country='iceland', recording_quality_equal='A').get()

    db = RawDataHandler(db_rootdir='./test_db', subfolder='audio',
                        payload_consts=web_xeno_canto_payload_consts, start_clean=True)
    for payload in xeno.payloads:
        db.populate_metadata(payload, col_subset=['catalogue_nr', 'english_name', 'country_recorded', 'file_url'])
        db.download_audio(payload)

#test3()