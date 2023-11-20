from SoiUtils.datasets.datasets import ImageDetectionDatasetCollection
from cloud.storage import download_folder
from pathlib import Path
download_folder(Path('test_dataset'),'soi_experiments','annotations-example/test')
video_collection = ImageDetectionDatasetCollection(Path('test_dataset'))
