'''Bla bla

'''
#
# Classes to map the semantics of a database entry to a numeric label that is to be
# predicted. A factory method implementation is used to make it easier to extend
#
import json
from datetime import datetime

from rawdata import web_xeno_canto_payload_consts

class LabelMakerMissingDataNameException(Exception):
    pass

class LabelMakerInvalidKeyException(Exception):
    pass

class LabelMakerMissingKeySourceException(Exception):
    pass

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