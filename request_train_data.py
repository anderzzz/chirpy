'''Script to request train data from web

'''
import requests
import urllib.parse
from pathlib import Path
import pandas as pd
import csv

from dataclasses import dataclass
from typing import Dict, List

from datetime import datetime
from copy import deepcopy

class WebXenoCantoException(Exception):
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

        if start_clean:
            Path(self.db_rootdir + '/' + self.db_file_name).unlink(missing_ok=True)

    def populate_metadata(self, payload, col_subset=None):
        '''Bla bla

        '''
        if col_subset is None:
            datacols = list(self.payload_consts.data_names.values())
        else:
            datacols = [self.payload_consts.data_names[key] for key in col_subset]

        df_payload = pd.DataFrame(payload[self.payload_consts.recordings_key], columns=datacols)
        df_payload = df_payload.rename(columns=self.payload_consts.data_names_inv).set_index(self.payload_consts.id_key)

        db_file = self.db_rootdir + '/' + self.db_file_name
        if not Path(db_file).exists():
            df_payload.to_csv(db_file)
        else:
            with open(db_file, 'a') as fcsv:
                data_str = df_payload.to_csv(header=False)
                fcsv.write(data_str)

    def deduplicate_db(self):
        '''Bla bla

        '''
        raise NotImplementedError('De-duplication not implemented yet')

    def download_audio(self, payload):
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

test3()