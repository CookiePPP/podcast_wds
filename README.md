## Intro

This repo contains the code to stream a CookiePPP formatted Text-to-Speech WebDataset (e.g`NOSPEAKER_PodcastsRank0-000000.tar`)

## Download + Install

```
git clone https://github.com/CookiePPP/podcast_wds
cd podcast_wds
python -m pip install -e .
```

## Scripts/Usage

[examples.py](https://github.com/CookiePPP/podcast_wds/blob/main/podcast_wds/examples.py) contains an example function which you would copy to your `dataset.py` and modify as needed

Example:

```python
from podcast_wds.examples import get_dataset_from_url
URLs = ["http://0.0.0.0:8888/NOSPEAKER_PodcastsRank0-000001.tar"]
# this should point to your file server and should be a list of each tar file you'll use.
# refer to https://github.com/webdataset/webdataset for AIStore/AWS/GCP formatted URLs
iterator = get_dataset_from_url(URLs)

for clip in iterator:
    print(
        clip['gt_wav'], # prints FloatTensor[1, wav_length, 1]
        clip['text']  , # prints string
    )
    # note - by default short clips are pre-merged and a hashtag is added at the boundary
    # e.g: clip['text'] = f"It's just Strawberry's honest opinion. #What's wrong with that?"
```

## Clip

As seen in the above example, the webdataset produces 'clips'. The clip format is shown below.

```python
clip = {
    'gt_wav': wav, # FloatTensor[1,n_samples,1]
    'n_samples': wav.shape[1],
    'duration': wav.shape[1] / sr, # duration in seconds
    'wav_path': f'{path-to-tar}/{path-inside-tar}',
    'sr': 48000, # all clips use 48khz, no exceptions
    'text': str, # the text associated with each clip
    'text_is_asr': bool, # is this text from a speech recognition model or written by a human?
    'rf5_emb': OPTIONAL[FloatTensor[256]], # some tars come with precomputed speaker embeddings from https://github.com/RF5/simple-speaker-embedding
    'chunk_id': int, # for audio files with multiple clips, an id is provided
    'num_chunks': int, # and the number of clips in the audio file is also provided.
}
```
