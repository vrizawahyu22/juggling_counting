from supervision.draw.color import ColorPalette
from supervision.video.dataclasses import VideoInfo
from supervision.video.source import get_video_frames_generator
from supervision.video.sink import VideoSink
from supervision.tools.detections import Detections, BoxAnnotator
from utils import LineCounter, LineCounterAnnotator
from tqdm.notebook import tqdm
import numpy as np
import cv2
from ultralytics import YOLO


MODEL_POSE = "yolov8n-pose.pt"
model_pose = YOLO(MODEL_POSE)

MODEL = "yolov8n.pt"
model = YOLO(MODEL)

# dict maping class_id to class_name
CLASS_NAMES_DICT = model.model.names
# get class id ball
CLASS_ID = [32]

SOURCE_VIDEO_PATH = f"dataset/juggling2.mp4"
TARGET_VIDEO_PATH = f"dataset/juggling2_result.mp4"

# create VideoInfo instance
video_info = VideoInfo.from_video_path(SOURCE_VIDEO_PATH)
# create frame generator
generator = get_video_frames_generator(SOURCE_VIDEO_PATH)
# create instance of BoxAnnotator and LineCounterAnnotator
box_annotator = BoxAnnotator(color=ColorPalette(), thickness=1, text_thickness=1, text_scale=0.5)
# create LineCounter instance
line_counter = LineCounter(start=0, end=0)
# create instance of BoxAnnotator and LineCounterAnnotator
line_annotator = LineCounterAnnotator(thickness=3, text_thickness=3, text_scale=2)

# open target video file
with VideoSink(TARGET_VIDEO_PATH, video_info) as sink:
    # loop over video frames
    for frame in tqdm(generator, total=video_info.total_frames):
        results_poses = model_pose.track(frame, persist=True)
        annotated_frame = results_poses[0].plot()
        keypoints = results_poses[0].keypoints.xy.int().cpu().tolist()
        bboxes = results_poses[0].boxes.xyxy.cpu().numpy()

        results_ball = model.track(frame, persist=True, conf=0.1)
        tracker_ids = results_ball[0].boxes.id.int().cpu().numpy() if results_ball[0].boxes.id is not None else None
        detections = Detections(
            xyxy=results_ball[0].boxes.xyxy.cpu().numpy(),
            confidence=results_ball[0].boxes.conf.cpu().numpy(),
            class_id=results_ball[0].boxes.cls.cpu().numpy().astype(int),
            tracker_id=tracker_ids
        )

        # filtering out detections with unwanted classes
        mask = np.array([class_id in CLASS_ID for class_id in detections.class_id], dtype=bool)
        detections.filter(mask=mask, inplace=True)
        
        # updating line
        line_counter.update_line(bboxes[0], keypoints[0])
        # updating line counter
        line_counter.update(detections=detections)
        # annotate and display frame
        labels = [
            f"id:{track_id} {CLASS_NAMES_DICT[class_id]} {confidence:0.2f}"
            for _, confidence, class_id, track_id
            in detections
        ]
        annotated_frame = box_annotator.annotate(frame=annotated_frame, detections=detections, labels=labels)
        line_annotator.annotate(frame=annotated_frame, line_counter=line_counter)

        cv2.imshow("YOLOv8 Inference", annotated_frame)
        sink.write_frame(annotated_frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
        
cv2.destroyAllWindows()