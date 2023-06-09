from typing import Union

import os
import torch
import torchaudio
import math
import ffmpeg
import tempfile
import subprocess

class Suppress: # taken from https://stackoverflow.com/a/50691621
    def __init__(self, suppress_stdout=True, suppress_stderr=False):
        self.suppress_stdout = suppress_stdout
        self.suppress_stderr = suppress_stderr
        self.original_stdout = None
        self.original_stderr = None

    def __enter__(self):
        import sys, os
        devnull = open(os.devnull, "w")

        # Suppress streams
        if self.suppress_stdout:
            self.original_stdout = sys.stdout
            sys.stdout = devnull

        if self.suppress_stderr:
            self.original_stderr = sys.stderr
            sys.stderr = devnull

    def __exit__(self, *args, **kwargs):
        import sys
        # Restore streams
        if self.suppress_stdout:
            sys.stdout = self.original_stdout

        if self.suppress_stderr:
            sys.stderr = self.original_stderr

CACHE_PATH = 'len_cache.pt'
CACHE = {}

class LowSamplingRateError(Exception):
    pass

class SilentError(Exception):
    pass

def load_audio_from_path_ffmpeg_old(path: str, **input_kwargs):
    probe = ffmpeg.probe(path)
    native_sr = int(probe['streams'][0]['sample_rate'])
    expected_len = float(probe['streams'][0]['duration_ts'])
    
    data, err = (
        ffmpeg
        .input(path, **input_kwargs)
        .output('-', format='f32le', acodec='pcm_f32le', channels=1)
        .global_args('-hide_banner', '-loglevel', 'warning')
        .run(capture_stdout=True, capture_stderr=True)
    )
    with Suppress(True, True): # suppress non-writable buffer warning
        data = torch.frombuffer(data, dtype=torch.float32).clone() # [wavT]
    if data.numel() < expected_len * 0.98: # using this instead of 'if err' because ffmpeg uses stderr for some warnings
        raise Exception(err.decode('utf8'))
    return data, native_sr

def load_audio_from_path_ffmpeg(path:str, **input_kwargs):
    assert os.path.exists(path), f'File does not exist: {path}'
    
    # use subprocess instead of ffmpeg-python because ffmpeg-python doesn't run in parallel
    probe = ffmpeg.probe(path)
    native_sr = int(probe['streams'][0]['sample_rate'])
    expected_len = float(probe['streams'][0]['duration_ts'])
    
    cmd = [
        'ffmpeg',
        '-i', path,
        '-f', 'f32le',
        '-acodec', 'pcm_f32le',
        '-ac', '1',
        '-',
    ]
    with subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE) as proc:
        data, err = proc.communicate()
    with Suppress(True, True): # suppress non-writable buffer warning
        data = torch.frombuffer(data, dtype=torch.float32).clone() # [wavT]
    if data.numel() < expected_len * 0.98: # using this instead of 'if err' because ffmpeg uses stderr for some warnings
        raise Exception(err.decode('utf8'))
    return data, native_sr

# TODO: Check if https://stackoverflow.com/a/72102009 is possible
def load_audio_from_bytes_ffmpeg(bytes_: bytes, **input_kwargs):
    # write bytes to temp file and load using ffmpeg
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, 'tmpfile.opus')
        open(path, 'wb').write(bytes_)
        data, native_sr = load_audio_from_path_ffmpeg(path, **input_kwargs)
    return data, native_sr

# updated to force file to stay on RAM
def load_audio_from_bytes_ffmpeg_ramonly(bytes_: bytes, **input_kwargs):
    # assert os is linux
    assert os.name == 'posix',\
        'load_audio_from_bytes_ffmpeg_ramonly only works on linux (and is not generally recommended)'
    
    # create temp ramdisk
    tmpdir = tempfile.TemporaryDirectory(dir='/dev/shm')
    path = os.path.join(tmpdir.name, 'tmpfile.opus')
    open(path, 'wb').write(bytes_)
    data, native_sr = load_audio_from_path_ffmpeg(path, **input_kwargs)
    
    # delete temp ramdisk
    tmpdir.cleanup()
    return data, native_sr

def load_audio(path: Union[str, bytes], sr: int = 48000, min_sr: int = None,
        start_time: float = None,
        end_time: float = None,
        ):
    input_kwargs = dict(ss=start_time, to=end_time) if start_time is not None or end_time is not None else {}
    if isinstance(path, bytes):
        if os.name == 'posix':
            data, native_sr = load_audio_from_bytes_ffmpeg_ramonly(path, **input_kwargs)
        else:
            data, native_sr = load_audio_from_bytes_ffmpeg(path, **input_kwargs)
    else:
        data, native_sr = load_audio_from_path_ffmpeg(path, **input_kwargs)
    
    assert data.numel() > 100, f'Empty audio file {path}'
    
    if min_sr is not None and native_sr < min_sr:
        raise LowSamplingRateError(f'Expected native_sr greater than or equal to {min_sr:.0f}, got {native_sr:.0f}')
    
    if sr is not None and not (sr*0.95 <= native_sr <= sr*1.05):
        #data = librosa.core.resample(data, orig_sr=native_sr, target_sr=sr)
        data = torchaudio.transforms.Resample(native_sr, sr)(data)
        native_sr = sr
    
    data -= data.mean() # remove DC offset
    abs_max = data.abs().max()
    if abs_max == 0:
        raise SilentError(f'Audio is silent')
    assert math.isfinite(abs_max), f'Audio has infs/NaNs'
    if abs_max > 1.0: # Audio is clipping (normally due to resampling, but could be corrupted file or bad code)
        data /= abs_max
    
    return data[None, :, None], native_sr