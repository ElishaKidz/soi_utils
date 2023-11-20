from datasets.datasets import ImageDetectionDatasetCollection, collate_fn
from cloud.storage import download_folder
from pathlib import Path
import os
from torch.utils.data import DataLoader

if not os.path.exists("./test_datasets"):
    download_folder(Path('test_dataset'),'soi_experiments','annotations-example/test')
video_collection = ImageDetectionDatasetCollection(Path('test_dataset/test'), "annotations.json")
loader = DataLoader(video_collection, batch_size=2, collate_fn=collate_fn, shuffle=True)
for i, (_, dets) in enumerate(loader):
    print(f"{i}. Number of dets {len(dets[0])}")

for i in range(video_collection.num_videos()):
    print(len(video_collection.get_video_dataset(i)))