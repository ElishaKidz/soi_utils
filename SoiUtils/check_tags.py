import json
import os
from utils import video_from_images
import pybboxes.functional as pbf
import cv2
from tqdm import tqdm


def find_relevant_tags(frame_id, tags):
    """
    find all the tages relevant for a frame id
    @param frame_id: the id of the frame
    @param tags: the list of all video tasg
    @return: the relevant
    """
    relevante_tages = [tag for tag in tags if tag["image_id"]==frame_id]
    states = ['static' if tag['attributes']['static'] else 'move' for tag in relevante_tages]
    bbs = [tag["bbox"] for tag in tags if tag["image_id"]==frame_id]
    return bbs, states


def draw_bbs_on_image(frame, bboxes, texts, bbox_format='coco', box_color=(0, 255, 0)):
    """
    draw a list of bbs on an image
    @param frame: image
    @param bboxes: list of bboxes
    @param bbox_format: the format of the bboxes
    @param box_color: what color to draw the bboxes
    @return: a frame with drawn bboxes
    """
    for bbox, text in zip(bboxes, texts):
        x, y, width, height = pbf.convert_bbox(bbox, from_type=bbox_format, to_type='coco',
                                               image_size=frame.shape[:2][::-1], return_values=True)
        frame = cv2.rectangle(frame, (x, y), (x + width, y + height), color=box_color, thickness=2)
        cv2.putText(frame, f'{text}'.strip(), (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (36, 255, 12), 1)
    return frame


def create_taged_video(data_dir_path):
    """
    put the tagged bboxes over the correspunded frames of a segment
    the directory should be same as appeare at the GCP
    - segment
    -- annotations
    ---- thermal_classifier.json
    -- images
    ---- frame_0000000.PNG
    ---- frame_0000001.PNG
    .
    .
    .
    @param data_dir_path: path to relevant segment
    """
    annotations = json.load(open(f"{data_dir_path}/annotations/thermal_classifier.json"))
    annotations = annotations["annotations"]
    frames = sorted(os.listdir(f"{data_dir_path}/images"))
    renderd_frames = []
    for frame_name in tqdm(frames, total=len(frames)):
        frame = cv2.imread(f"{data_dir_path}/images/{frame_name}")
        frame_id = int(frame_name.split("_")[1].split(".")[0])
        bboxes, states = find_relevant_tags(frame_id, annotations)
        ren_frame = draw_bbs_on_image(frame, bboxes, states)
        renderd_frames.append(ren_frame)
    video_from_images(renderd_frames, video_save_path=f"{data_dir_path}/taged_video.mp4")


def create_taged_videos(dir_of_segments):
    """
    @param dir_of_segments: path to dir of segments directories
    """
    for segment in tqdm(os.listdir(dir_of_segments), total=len(os.listdir(dir_of_segments))):
        create_taged_video(f"{dir_of_segments}/{segment}")


#######################
# running example for functions
# create_taged_videos(r"C:\Users\orber\PycharmProjects\VeCToR_data")
# create_taged_video(r"C:\Users\orber\PycharmProjects\VeCToR_data\record_29_12_2023_04_48_secondary_segment_4.mp4")
#######################