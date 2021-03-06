import numpy as np
import os

from moviepy.editor import VideoFileClip

from detection import detection_cast, extract_detections, draw_detections
from metrics import iou_score


class Tracker:
    """Generate detections and build tracklets."""
    def __init__(self, return_images=True, lookup_tail_size=80, labels=None):
        self.return_images = return_images  # Return images or detections?
        self.frame_index = 0
        self.labels = labels  # Tracker label list
        self.detection_history = []  # Saved detection list
        self.last_detected = {}
        self.tracklet_count = 0  # Counter to enumerate tracklets

        # We will search tracklet at last lookup_tail_size frames
        self.lookup_tail_size = lookup_tail_size

    def new_label(self):
        """Get new unique label."""
        self.tracklet_count += 1
        return self.tracklet_count - 1

    def init_tracklet(self, frame):
        """Get new unique label for every detection at frame and return it."""
        # Use extract_detections and new_label
        extracted_detections = extract_detections(frame)
        for i in range(extracted_detections.shape[0]):
            extracted_detections[i, 0] = self.new_label()
        # print(list(extracted_detections))
        return (extracted_detections)

    @property
    def prev_detections(self):
        """Get detections at last lookup_tail_size frames from detection_history.

        One detection at one id.
        """

        detections = []
        # Write code here
        ids = set()
        history = self.detection_history[-self.lookup_tail_size:]
        all_detections = []
        for in_frame_det in history:
            for det in in_frame_det:
                all_detections.append(det)

        for detection in all_detections:
            id = detection[0]
            if id in ids:
                continue
            detections.append(detection)
            ids.add(id)
        return detection_cast(detections)

    def bind_tracklet(self, detections):
        """Set id at first detection column.

        Find best fit between detections and previous detections.

        detections: numpy int array Cx5 [[label_id, xmin, ymin, xmax, ymax]]
        return: binded detections numpy int array Cx5 [[tracklet_id, xmin, ymin, xmax, ymax]]
        """
        detections = detections.copy()
        prev_detections = self.prev_detections

        # Write code here
        # Step 1: calc pairwise detection IOU
        for i, detection in enumerate(list(detections)):
            max_iou = 0
            for prev_detection in list(prev_detections):
                current_score = iou_score(detection[1:], prev_detection[1:])
                if current_score > max_iou:
                    detections[i, 0] = prev_detection[0]
                    max_iou = current_score
            # unmatched
            if max_iou == 0:
                detections[i, 0] = self.new_label()
        # Step 2: sort IOU list

        # Step 3: fill detections[:, 0] with best match
        # One matching for each id

        # Step 4: assign new tracklet id to unmatched detections
        return detection_cast(detections)

    def save_detections(self, detections):
        """Save last detection frame number for each label."""
        detections = np.array(detections)
        for label in detections[:, 0]:
            self.last_detected[label] = self.frame_index

    def update_frame(self, frame):
        if not self.frame_index:
            # First frame should be processed with init_tracklet function
            detections = self.init_tracklet(frame)
        else:
            # Every Nth frame should be processed with CNN (very slow)
            # First, we extract detections
            detections = extract_detections(frame, labels=self.labels)
            # Then bind them with previous frames
            # Replacing label id to tracker id is performing in bind_tracklet function
            detections = self.bind_tracklet(detections)

        # After call CNN we save frame number for each detection
        self.save_detections(detections)
        # Save detections and frame to the history, increase frame counter
        self.detection_history.append(detections)
        self.frame_index += 1

        # Return image or raw detections
        # Image usefull to visualizing, raw detections to metric
        if self.return_images:
            return draw_detections(frame, detections)
        else:
            return detections


def main():
    dirname = os.path.dirname(__file__)
    input_clip = VideoFileClip(os.path.join(dirname, 'data', 'test.mp4'))

    tracker = Tracker()
    input_clip.fl_image(tracker.update_frame).preview()


if __name__ == '__main__':
    main()
