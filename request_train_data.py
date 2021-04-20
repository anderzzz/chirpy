'''Script to request train data from web

'''
import requests
import urllib.parse

from dataclasses import dataclass
from typing import Dict
from datetime import datetime
from copy import deepcopy
import json

class WebXenoCantoException(Exception):
    pass

@dataclass
class WebConsts:
    base_url : str
    entity_name : str
    query : str
    query_args : Dict
    api_help_url : str

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

def test1():
    xeno = WebXenoCanto(web_xeno_canto_consts)
    xeno.query(country='iceland', recording_quality_equal='A').get()

def test2():
    xeno = WebXenoCanto(web_xeno_canto_consts)
    xeno.query(country='iceland', recording_quality_gt='C').get_all()

test2()