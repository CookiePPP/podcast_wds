import json
from typing import Iterable, List

import torch
import io
from podcast_wds.load_audio import load_audio

def avg(*x):
    return sum(x) / len(x)

# util function for merging neighbouring segments to make longer samples for training.
def merge_segments(segments: List, max_time_gap=0.3, max_duration=20.0, merge_text='#'):
    segments_copy = segments.copy()
    segment = segments_copy.pop(0)
    new_segments = []
    while True:
        # segment = {"id": 0, "start": 0.0, "end": 7.0, "text": ".", "avg_logprob": -0.5346933022523538,
        #            "compression_ratio": 1.588235294117647}
        next_segment = segments_copy.pop(0)
        gap_between_segments = next_segment['start'] - segment['end']
        is_neighbour = gap_between_segments < max_time_gap
        would_be_too_long = next_segment['end'] - segment['start'] > max_duration
        
        # merge new segment into current segment
        if is_neighbour and not would_be_too_long:
            segment = {
                'start': segment['start'],
                'end': next_segment['end'],
                'text': segment['text'].strip() + merge_text + next_segment['text'].strip(),
                'avg_logprob': avg(segment['avg_logprob'], next_segment['avg_logprob']),
                'compression_ratio': avg(segment['compression_ratio'], next_segment['compression_ratio']),
                'id': segment['id'],
                'end_id': next_segment['id'],
            }
        # add current segment to list and start chain
        else:
            new_segments.append(segment)
            segment = next_segment
        
        if len(segments_copy) == 0:
            new_segments.append(segment)
            break
    return new_segments

# takes raw file bytes and returns list of {gt_wav, text, rf5_emb}
# converts `.opus` into raw waveform tensor(s)
# converts `.whisper` into transcripts+timestamps
# converts `.txt` into transcripts
# converts `.rf5_emb` into speaker embeddings
def decode_sample(sample:dict, should_merge_segments=True, max_time_gap=0.1, max_duration=12.0, merge_text='#'):
    sample = {k.split('.')[-1]: v for k, v in sample.items()} # 'opus.bak' -> 'bak' (use last file extension only)
    assert 'opus' in sample, f'expected "opus" in sample, got {sample.keys()}\n{sample.get("__key__", "")}'
    has_whisper = 'whisper' in sample
    has_text    = 'text'    in sample
    has_rf5_emb = 'rf5_emb' in sample

    wav_path = sample['__url__'] + '/' + sample['__key__']
    
    # load audio from bytes
    opus_bytes = sample['opus']
    wav, sr = load_audio(opus_bytes, sr=48000) # Float[1,T,1], 48000
    assert wav.isfinite().all(), f'got non-finite values in wav for {wav_path}'
    
    # load whisper from bytes
    if has_whisper:
        whisper_bytes = sample['whisper']
        whisper = json.loads(whisper_bytes.decode('utf-8'))
    
    # load text from bytes
    if has_text:
        text_bytes = sample['text']
        text = text_bytes.decode('utf-8').strip()
    
    # load speaker embedding from bytes
    if has_rf5_emb:
        rf5_emb_bytes = sample['rf5_emb']
        with io.BytesIO(rf5_emb_bytes) as f:
            rf5_emb = torch.load(f) # Dict[str, Float[256]]
        rf5_emb = {int(k.split(';;')[-1]): v for k, v in rf5_emb.items()} # '{path};;{id}' -> id
    
    # carry unknown fields from input
    carry = {}
    for k, v in sample.items():
        if k not in {'opus', 'whisper', 'text', 'rf5_emb'}:
            carry[k] = v
    
    # convert into list format (where each item is it's own independant sample)
    # e.g: [{'wav': Float[1,T,1], 'whisper': dict, 'text': str, 'rf5_emb': Float[256]}, ...]
    only_one_sample = (not has_whisper) or len(whisper['segments']) <= 1
    if only_one_sample:
        sample = {
            'gt_wav': wav,
            'n_samples': wav.shape[1],
            'duration': wav.shape[1] / sr,
            'sr': sr,
            **carry
        }
        if has_text:
            sample['text'] = text
            sample['text_is_asr'] = False
        elif has_whisper:
            sample['text'] = whisper['text']
            sample['text_is_asr'] = True
        if has_rf5_emb:
            sample['rf5_emb'] = rf5_emb[0]
        sample['chunk_id'] = 0
        sample['num_chunks'] = 1
        sample['wav_path'] = wav_path
        return [sample]
    else:
        assert has_whisper
        samples = []
        if should_merge_segments:
            segments = merge_segments(whisper['segments'], max_time_gap, max_duration, merge_text)
        else:
            segments = whisper['segments']
        for segment in segments:
            sample = {'sr': sr, **carry}
            sample['wav_path'] = wav_path
            
            # filter out extremely fast/slow speaking segments
            n_words = len(segment['text'].split())
            duration_sec = segment['end'] - segment['start']
            wpm = (n_words / duration_sec) * 60
            if wpm < 50 or wpm > 360:
                continue
            if duration_sec < 0.3:
                continue
            
            # slice the full wav to get this segment's audio
            start_time_sec = segment['start']
            end_time_sec   = segment['end']
            start_sample = int(start_time_sec * sr)
            end_sample   = int(end_time_sec   * sr)
            wav_segment = wav[:,start_sample:end_sample].data.clone() # avoid using views for multi-threading safety
            sample['gt_wav'] = wav_segment
            sample['n_samples'] = wav_segment.numel()
            sample['duration'] = wav_segment.numel() / sr
            
            # get text for this segment
            text_segment = segment['text']
            sample['text'] = text_segment
            sample['text_is_asr'] = True
            
            # get speaker embedding for this segment
            if has_rf5_emb and segment['id'] in rf5_emb:
                rf5_emb_segment = rf5_emb[segment['id']]
                sample['rf5_emb'] = rf5_emb_segment
            
            # save chunk id for possible downstream use
            # (e.g: concat segments to make longer samples, or use alternate segment from same file for conditioning)
            sample['chunk_id'] = segment['id']
            sample['num_chunks'] = len(whisper['segments'])
            
            # save any additional whisper data
            # (though this data cannot be estimated from text, so may not always be available)
            set_if_exists = lambda o_obj, k, i_obj: o_obj.__setitem__(k, i_obj[k]) if k in i_obj else None
            set_if_exists(sample,'avg_logprob'      , segment) # sample['avg_logprob'] = ... (if exists)
            set_if_exists(sample,'compression_ratio', segment) # sample['compression_ratio'] = ... (if exists)
            set_if_exists(sample,'no_speech_prob'   , segment) # sample['no_speech_prob'] = ... (if exists)
            
            samples.append(sample)
        del wav
        return samples

def decode_samples(samples:Iterable[dict], should_merge_segments=True, max_time_gap=0.1, max_duration=12.0, merge_text='#'):
    # generator that yields processed samples from raw samples
    kwargs = dict(should_merge_segments=should_merge_segments, max_time_gap=max_time_gap,
                  max_duration=max_duration, merge_text=merge_text)
    for sample in samples:
        decoded_sample_list = decode_sample(sample, **kwargs)
        while len(decoded_sample_list):
            yield decoded_sample_list.pop(-1) # using pop() to reduce memory consumption

def decode_samples_ignore_exc(samples:Iterable[dict], should_merge_segments=True, max_time_gap=0.1, max_duration=12.0, merge_text='#'):
    # generator that yields processed samples from raw samples
    kwargs = dict(should_merge_segments=should_merge_segments, max_time_gap=max_time_gap,
                  max_duration=max_duration, merge_text=merge_text)
    for sample in samples:
        try:
            decoded_sample_list = decode_sample(sample, **kwargs)
            while len(decoded_sample_list):
                yield decoded_sample_list.pop(-1) # using pop() to reduce memory consumption
        except KeyboardInterrupt:
            raise
        except Exception as e:
            continue