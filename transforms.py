'''Custom audio transforms. Most are wrappers to `pydub` functionality

'''
import torch
import pydub
from pydub.generators import WhiteNoise

import math
from numpy.random import randint

class AudioTransformInitializationException(Exception):
    pass

class AudioTransformChunkMethodException(Exception):
    pass

class Compose(object):
    '''Bla bla

    '''
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, audio):
        for transform in self.transforms:
            audio = transform(audio)

        return audio

class AudioToTensorTransform(object):
    '''Bla bla

    '''
    def __init__(self, dtype_cast=None):
        self.dtype_cast = dtype_cast

    def __call__(self, audio):
        assert isinstance(audio, pydub.AudioSegment)
        audio_tensor = torch.tensor(audio.get_array_of_samples())

        if not self.dtype_cast is None:
            audio_tensor = audio_tensor.type(dtype=self.dtype_cast)

        return audio_tensor

class AudioDownSampleTransform(object):
    '''Bla bla

    '''
    def __init__(self, rate_target):
        self.rate_target = rate_target

    def __call__(self, audio):
        assert isinstance(audio, pydub.AudioSegment)
        audio = audio.set_frame_rate(self.rate_target)

        return audio

class AudioNormalizeTransform(object):
    '''Normalize the audio data.

    Borrows from the corresponding functional in `torchvision`

    '''
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, audio):
        '''Bla bla

        '''
        if not isinstance(audio, torch.Tensor):
            raise TypeError('Input audio should be a torch tensor. Got {}.'.format(type(audio)))

        audio = audio.clone()

        dtype = audio.dtype
        mean = torch.as_tensor(self.mean, dtype=dtype, device=audio.device)
        std = torch.as_tensor(self.std, dtype=dtype, device=audio.device)
        if (std == 0).any():
            raise ValueError('std evaluated to zero after conversion to {}, leading to division by zero.'.format(dtype))
        if mean.ndim == 1:
            mean = mean.view(-1, 1, 1)
        if std.ndim == 1:
            std = std.view(-1, 1, 1)
        audio.sub_(mean).div_(std)

        return audio

class AudioChunkifyTransform(object):
    '''Bla bla

    '''
    # These constants have to correspond to the class methods that can be invoked through __call__
    accepted_methods = ['sequential', 'space_evenly', 'pad']

    def __init__(self, run_time, method='sequential', strict=True, **kwargs):
        self.run_time = run_time
        if not method in self.accepted_methods:
            raise AudioTransformInitializationException('Unknown method to transform audio into chunks: {}'.format(method))
        self.method = method
        self.strict = strict
        self.kwargs_method = kwargs

    def __call__(self, audio):
        '''Bla bla

        '''
        total_ms = len(audio)
        try:
            audio_chunks = getattr(self, self.method)(audio, total_ms, **self.kwargs_method)
        except AudioTransformChunkMethodException as err:
            if self.strict:
                raise
            else:
                print ('Warning: exception raised for {}: {}'.format(audio, err))
                audio_chunks = []

        return audio_chunks

    def _cmp_n_chunks(self, total_ms):
        n_chunks = total_ms // self.run_time
        if n_chunks == 0:
            raise AudioTransformChunkMethodException('Total run length {} less than specified chunk length {}'.format(total_ms, self.run_time))
        slack = total_ms % self.run_time
        return n_chunks, slack

    def sequential(self, audio, total_ms):
        '''Bla bla

        '''
        n_chunks, _ = self._cmp_n_chunks(total_ms)
        audio_ret = []
        for k_chunk in range(n_chunks):
            audio_ret.append(audio[k_chunk * self.run_time: (k_chunk + 1) * self.run_time])

        return audio_ret

    def space_evenly(self, audio, total_ms):
        '''Bla bla

        '''
        n_chunks, slack = self._cmp_n_chunks(total_ms)
        slack_divided = slack // (n_chunks + 1)
        audio_ret = []
        for k_chunk in range(n_chunks):
            print (k_chunk * self.run_time + (k_chunk + 1) * slack_divided, (k_chunk + 1) * (self.run_time + slack_divided))
            audio_ret.append(audio[k_chunk * self.run_time + (k_chunk + 1) * slack_divided: \
                                   (k_chunk + 1) * (self.run_time + slack_divided)])

        return audio_ret

    def pad(self, audio, total_ms):
        '''Bla bla

        '''
        n_chunks, _ = self._cmp_n_chunks(total_ms)
        silent_budget = (n_chunks + 1) * self.run_time - total_ms
        audio_ret = []
        start = 0
        for k_chunk in range(n_chunks + 1):
            padding_tot = silent_budget // (n_chunks + 1 - k_chunk)
            silent_budget -= padding_tot
            pad_pre = math.floor(padding_tot / 2)
            pad_post = math.ceil(padding_tot / 2)
            end = start + self.run_time - padding_tot
            audio_content = audio[start: end]
            start = end
            audio_ret.append(pydub.AudioSegment.silent(pad_pre) +
                             audio_content +
                             pydub.AudioSegment.silent(pad_post))

        return audio_ret


class AudioAddWhiteNoiseTransform(object):
    '''Bla bla

    '''
    def __init__(self, volume=-20.0, sample_rate=44100):
        self.volume = volume
        self.white_noise_generator = WhiteNoise(sample_rate=sample_rate)

    def __call__(self, audio):
        '''Bla bla

        '''
        audio_white_noise = self.white_noise_generator.to_audio_segment(duration=len(audio), volume=self.volume)
        return audio.overlay(audio_white_noise)

class AudioScaleVolumeTransform(object):
    '''Scale the volume of the audio.

    Since audio representations have a maximum possible amplitude, scaling up beyond some threshold will not
    further impact the final output. The gain that is applied is logarithmic, so the initialization parameter
    is centred around 0.0 and for well-behaved input audio the gain should not have to go outside the interval
    [-20.0, 20.0].

    Args:
        gain (float): The gain to apply to the audio. That means 0.0 means no change, positive values increases
            the amplitude of the input audio, and negative values decreases the amplitude. The gain is logarithmic.

    '''
    def __init__(self, gain):
        self.gain = gain

    def __call__(self, audio):
        '''Bla bla

        '''
        return audio.apply_gain(self.gain)

class AudioScaleVolumeRelativeMaxTransform(object):
    '''Bla bla

    '''
    def __init__(self, factor_of_max):
        assert factor_of_max <= 1.0
        assert factor_of_max > 0.0
        self.factor_of_max = factor_of_max

    def __call__(self, audio):
        '''Bla bla

        '''
        maxdbfs = audio.max_dBFS
        return audio.apply_gain(math.log10(self.factor_of_max) - maxdbfs)

class AudioRandomChunkTransform(object):
    '''Bla bla

    '''
    def __init__(self, run_time, append_method=None):
        self.run_time = run_time
        if not append_method in [None, 'silence', 'cycle']:
            raise AudioTransformChunkMethodException('Unknown append method {} given a shorter audio than run-time'.format(append_method))
        self.append_method = append_method

    def __call__(self, audio):
        '''Bla bla

        '''
        total_ms = len(audio)
        if total_ms < self.run_time:
            if self.append_method is None:
                raise AudioTransformChunkMethodException('Total run-time for {} is {}, which is less than specified run-time {}'.format(audio, total_ms, self.run_time))
            elif self.append_method == 'silence':
                silent_slack = self.run_time - total_ms
                start = 0
            elif self.append_method == 'cycle':
                audio = audio * math.ceil(self.run_time / total_ms)
                silent_slack = 0
                start = 0
            else:
                raise AudioTransformChunkMethodException('Unknown append method {} given a shorter audio than run-time'.format(self.append_method))
        else:
            silent_slack = 0
            start = randint(low=0, high=total_ms - self.run_time)

        audio_slice = audio[start: start + self.run_time - silent_slack]
        if silent_slack > 0:
            audio_slice += pydub.AudioSegment.silent(silent_slack)

        return audio_slice

class AudioRandomChunkCycledTransform(object):
    '''Bla bla

    '''
    def __init__(self, run_time):
        self.random_chunker = AudioRandomChunkTransform(run_time=run_time, strict=True)

    def __call__(self, audio):
        '''Bla bla

        '''
        if len(audio) < self.random_chunker.run_time:
            audio_ = math.ceil(self.random_chunker.run_time / len(audio)) * audio
        else:
            audio_ = audio
        self.random_chunker(audio_)

class AudioOverlayTransform(object):
    '''Bla bla

    '''
    def __init__(self):
        raise NotImplementedError()

    def __call__(self, audio):
        pass

class AudioRandomImpulseTransform(object):
    '''Bla bla

    '''
    def __init__(self):
        raise NotImplementedError()

    def __call__(self, audio):
        pass

class AudioTrafficOverlayTransform(AudioOverlayTransform):
    '''Bla bla

    '''
    def __init__(self):
        super().__init__()

    def __call__(self, audio):
        pass