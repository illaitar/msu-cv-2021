def iou_score(bbox1, bbox2):
    """Jaccard index or Intersection over Union.

    https://en.wikipedia.org/wiki/Jaccard_index

    bbox: [xmin, ymin, xmax, ymax]
    """

    assert len(bbox1) == 4
    assert len(bbox2) == 4
    import numpy as np
    m = max(np.max(bbox1), np.max(bbox2))
    field = np.zeros((m, m))
    field[bbox1[0]:bbox1[2], bbox1[1]:bbox1[3]] += 1
    field[bbox2[0]:bbox2[2], bbox2[1]:bbox2[3]] += 1

    return np.count_nonzero(field == 2) / np.count_nonzero(field > 0)


def motp(obj, hyp, threshold=0.5):
    """Calculate MOTP

    obj: list
        Ground truth frame detections.
        detections: numpy int array Cx5 [[id, xmin, ymin, xmax, ymax]]

    hyp: list
        Hypothetical frame detections.
        detections: numpy int array Cx5 [[id, xmin, ymin, xmax, ymax]]

    threshold: IOU threshold
    """

    def to_dict(list_of_bboxes):
        dict_out = dict()
        for bbox in list_of_bboxes:
            dict_out[bbox[0]] = bbox[1:]
        return dict_out

    dist_sum = 0  # a sum of IOU distances between matched objects and hypotheses
    match_count = 0

    matches = {}  # matches between object IDs and hypothesis IDs

    # For every frame
    for frame_obj, frame_hyp in zip(obj, hyp):
        # Write code here
        # Step 1: Convert frame detections to dict with IDs as keys
        #print('frame_obj', frame_obj, 'frame_hyp', frame_hyp)
        dict_obj = to_dict(frame_obj)
        dict_hyp = to_dict(frame_hyp)
        #print('dict_obj', dict_obj, 'dict_hyp', dict_hyp)
        # Step 2: Iterate over all previous matches
        # If object is still visible, hypothesis still exists
        # and IOU distance > threshold - we've got a match
        for id in matches.keys():
            if id in set(dict_obj.keys()) and \
                    matches[id] in set(dict_hyp.keys()) and \
                    iou_score(dict_obj[id], dict_hyp[matches[id]]) > threshold:
                # Update the sum of IoU distances and match count
                dist_sum += iou_score(dict_obj[id], dict_hyp[matches[id]])
                match_count += 1
                # Delete matched detections from frame detections
                dict_obj.pop(id)
                dict_hyp.pop(matches[id])

        # Step 3: Calculate pairwise detection IOU between remaining frame detections
        # Save IDs with IOU > threshold
        saved_ious = []
        for id_obj in dict_obj.keys():
            for id_hyp in dict_hyp.keys():
                iou = iou_score(dict_obj[id_obj], dict_hyp[id_hyp])
                if iou > threshold:
                    saved_ious.append((id_obj, id_hyp, iou))
        saved_ious = sorted(saved_ious, key=lambda x: x[2])

        # Step 4: Iterate over sorted pairwise IOU
        # Update the sum of IoU distances and match count
        # Delete matched detections from frame detections
        # Step 5: Update matches with current matched IDs
        new_match = dict()
        for iou in saved_ious:
            new_match[iou[0]] = (iou[1], iou[2])
        for id_obj in new_match.keys():
            matches[id_obj] = new_match[id_obj][0]
            dist_sum += new_match[id_obj][1]
            match_count += 1



    # Step 6: Calculate MOTP
    MOTP = dist_sum/match_count

    return MOTP


def motp_mota(obj, hyp, threshold=0.5):
    """Calculate MOTP/MOTA

    obj: list
        Ground truth frame detections.
        detections: numpy int array Cx5 [[id, xmin, ymin, xmax, ymax]]

    hyp: list
        Hypothetical frame detections.
        detections: numpy int array Cx5 [[id, xmin, ymin, xmax, ymax]]

    threshold: IOU threshold
    """
    def to_dict(list_of_bboxes):
        dict_out = dict()
        for bbox in list_of_bboxes:
            dict_out[bbox[0]] = bbox[1:]
        return dict_out
    dist_sum = 0  # a sum of IOU distances between matched objects and hypotheses
    match_count = 0
    missed_count = 0
    false_positive = 0
    mismatch_error = 0

    matches = {}  # matches between object IDs and hypothesis IDs
    g = 0
    # For every frame
    for frame_obj, frame_hyp in zip(obj, hyp):
        g+=len(frame_obj)
        # Step 1: Convert frame detections to dict with IDs as keys
        dict_obj = to_dict(frame_obj)
        dict_hyp = to_dict(frame_hyp)
        for id in matches.keys():
            if id in set(dict_obj.keys()) and \
                    matches[id] in set(dict_hyp.keys()) and \
                    iou_score(dict_obj[id], dict_hyp[matches[id]]) > threshold:
                # Update the sum of IoU distances and match count
                dist_sum += iou_score(dict_obj[id], dict_hyp[matches[id]])
                match_count += 1
                # Delete matched detections from frame detections
                dict_obj.pop(id)
                dict_hyp.pop(matches[id])

        # Step 3: Calculate pairwise detection IOU between remaining frame detections
        # Save IDs with IOU > threshold
        saved_ious = []
        for id_obj in dict_obj.keys():
            for id_hyp in dict_hyp.keys():
                iou = iou_score(dict_obj[id_obj], dict_hyp[id_hyp])
                if iou > threshold:
                    saved_ious.append((id_obj, id_hyp, iou))
        saved_ious = sorted(saved_ious, key=lambda x: x[2])

        # Step 4: Iterate over sorted pairwise IOU
        # Update the sum of IoU distances and match count
        # Delete matched detections from frame detections

        # Step 5: If matched IDs contradict previous matched IDs - increase mismatch error

        # Step 6: Update matches with current matched IDs

        # Step 7: Errors
        # All remaining hypotheses are considered false positives
        # All remaining objects are considered misses

        new_match = dict()
        for iou in saved_ious:
            new_match[iou[0]] = (iou[1], iou[2])
            dict_obj.pop(iou[0])
            dict_hyp.pop(iou[1])

        for id_obj in new_match.keys():
            if id_obj in set(matches.keys()):
                mismatch_error += 1
            matches[id_obj] = new_match[id_obj][0]
            dist_sum += new_match[id_obj][1]
            match_count += 1

        false_positive += len(dict_hyp.keys())
        missed_count += len(dict_obj.keys())
    # Step 8: Calculate MOTP and MOTA
    MOTP = dist_sum / match_count
    MOTA = 1 - (missed_count + false_positive + mismatch_error) / g

    return MOTP, MOTA
