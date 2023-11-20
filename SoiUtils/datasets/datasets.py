from torch.utils.data import Dataset, ConcatDataset
from pycocotools.coco import COCO
from pathlib import Path
from SoiUtils.datasets.base import ImageDetectionSample,Detection
import cv2 as cv

def collate_fn(batch):
    return tuple(zip(*batch))

class ImageDetectionDataset(Dataset):
    FRAMES_DIR_NAME = "frames"
    BBOX_FORMAT = 'coco'

    def __init__(self,dataset_root_dir:str, annotation_file_name: str, transforms = None):
        super().__init__()
        self.dataset_root_dir = Path(dataset_root_dir)
        all_dataset_info = COCO(self.dataset_root_dir/annotation_file_name)
        self.image_info = all_dataset_info.imgs
        self.image_ids = list(all_dataset_info.imgs.keys())
        self.imgToAnns = all_dataset_info.imgToAnns
        self.transforms = transforms

    def __getitem__(self, index:int) -> ImageDetectionSample:
        image_id = self.image_ids[index]
        image_file_path = str(self.dataset_root_dir/ImageDetectionDataset.FRAMES_DIR_NAME/self.image_info[image_id]['file_name'])
        image = cv.imread(image_file_path)
        detections = [Detection.load_generic_mode(bbox=detection_annotation['bbox'], cl=detection_annotation['category_id'], 
                                                  from_type=ImageDetectionDataset.BBOX_FORMAT, to_type="coco", image_size=image.shape[:2][::-1])
                       for detection_annotation in self.imgToAnns[index]]


        image_detection_sample = ImageDetectionSample(image=image,detections=detections)

        if self.transforms is not None:
            item = self.transforms(image_detection_sample)
        
        else:
            item = image_detection_sample
        
        return item.image, [det.__dict__ for det in item.detections]

    def __len__(self):
        return len(self.image_ids)
    

class ImageDetectionDatasetCollection(Dataset):
    def __init__(self, collection_root_dir: str, annotation_file_name: str, **kwargs) -> None:
        super().__init__()
        self.collection_root_dir = Path(collection_root_dir)
        self.collection_items_root_dirs = [f for f in self.collection_root_dir.iterdir() if f.is_dir()]
        self.kwargs = kwargs
        self.collection = ConcatDataset([ImageDetectionDataset(f, annotation_file_name, **self.kwargs) for f in self.collection_items_root_dirs])
    
    def __getitem__(self, index:int) -> ImageDetectionDataset:
        return self.collection[index]
    
    def __len__(self):
        return len(self.collection)

    def get_video_dataset(self, index):
        return self.collection.datasets[index]
    
    def num_videos(self):
        return len(self.collection_items_root_dirs)


