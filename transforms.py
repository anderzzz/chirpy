'''Custom audio transforms

'''
import pydub

class Compose(object):
    pass

class AudioTo1DTensor(object):
    '''Bla bla

    '''
    def __init__(self):
        pass

    def __call__(self, label, audio):
        print (label)
        print (audio)
        print (audio.get_array_of_samples())
        raise RuntimeError

class AudioWhiteNoiseTransform(object):
    '''Bla bla

    '''
    def __init__(self):
        pass

    def __call__(self, sample):
        pass
