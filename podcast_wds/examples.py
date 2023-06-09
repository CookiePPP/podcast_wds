import webdataset as wds
from podcast_wds.decode_sample import decode_samples, decode_samples_ignore_exc
from functools import partial
from typing import Union, List

def get_dataset_from_url(url:Union[List[str], str], ignore_exc:bool = True, decode_kwargs:dict = {}):
    decode_fn = decode_samples_ignore_exc if ignore_exc else decode_samples
    dataset = wds.DataPipeline(
        wds.SimpleShardList(url),
        # at this point we have an iterator over all the urls
        wds.shuffle(100), # collect 100 urls and shuffle them
        # at this point, we have an iterator over the shards assigned to each worker
        wds.tarfile_to_samples(), # url -> dict of file bytes
        partial(decode_fn, **decode_kwargs), # dict of file bytes -> list of {wav:Tensor, text:str, rf5_emb:Tensor}
        wds.shuffle(1000), # collect 1000 decompressed samples and shuffle them
        # at this point, we have an list of decompressed training samples from each shard in this worker in sequence
        # either convert into batches here, or in the collate_fn of the dataloader
    )
    return dataset


if __name__ == '__main__':
    from tqdm import tqdm
    
    test_tarfile_path = [f"http://0.0.0.0:8000/NOSPEAKER_PodcastsRank0-{i:06}.tar" for i in range(500)]
    
    # test code runs without exceptions
    dataset = get_dataset_from_url(test_tarfile_path)
    for i, sample in enumerate(tqdm(dataset, smoothing=0.0)):
        if i > 1000:
            break # test first 1000 samples
    
    # performance test with multiple workers
    from torch.utils.data import DataLoader
    
    dataset = get_dataset_from_url(test_tarfile_path)
    dataloader = DataLoader(dataset, batch_size=1, num_workers=2, prefetch_factor=2, collate_fn=lambda x: x)
    for i, batch in enumerate(tqdm(dataloader, smoothing=0.0)):
        pass
    