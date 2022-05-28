import torch
from torch.nn import Sequential
import numpy as np
import torch.nn as nn 

# ============================== 1 Classifier model ============================


def get_cls_model(input_shape):
    """
    :param input_shape: tuple (n_rows, n_cols, n_channels)
            input shape of image for classification
    :return: nn model for classification
    """
    from torch import nn
    rows, cols, ch = input_shape
    return Sequential(
        nn.Conv2d(1, 8, kernel_size=3, padding=1),  nn.MaxPool2d(2),
        nn.ReLU(),
        nn.Conv2d(8, 4, kernel_size=3, padding=1),  nn.MaxPool2d(2),
        nn.ReLU(),
        nn.Flatten(),
        nn.Linear(rows//4 * cols // 4 * 4, 100),

        nn.ReLU(),

        nn.Linear(100,20),
        nn.ReLU(),
         nn.Linear(20,2)
    )


def fit_cls_model(X, y, epochs=55, save_model=False):
    """
    :param X: 4-dim tensor with training images
    :param y: 1-dim tensor with labels for training
    :return: trained nn model
    """
    from torch import nn, optim, save
    model = get_cls_model((40, 100, 1))
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    model.train()
    for i in range(epochs):
        optimizer.zero_grad()
        y_pred = model(X)
        loss = criterion(y_pred, y)
        print("loss on epoch", i, ":", loss.item())
        loss.backward()
        optimizer.step()
    #model = get_detection_model(model)
    if save_model:
        save(model, "classifier_model.pth")
    return model.eval()


# ============================ 2 Classifier -> FCN =============================
def get_detection_model(cls_model):
    """
    :param cls_model: trained cls model
    :return: fully convolutional nn model with weights initialized from cls
             model
    """
    res_m = []
    flag = False
    for idx, layer in enumerate(list(cls_model.children())):
        if isinstance(layer, nn.Flatten):
            flag = True
            continue
        if isinstance(layer, nn.Linear) and flag:
            flag = False
            A, b = layer.parameters()
            conv = nn.Conv2d(4, b.shape[0], (10, 25), (1, 3))
            weights, _ = conv.parameters()
            weights = torch.reshape(A, weights.shape)
            conv.weight = torch.nn.Parameter(weights)
            res_m.append(conv)
        else:
            if isinstance(layer, nn.Linear):
                A, b = layer.parameters()
                conv = nn.Conv2d(A.shape[1], A.shape[0], 1, 1)
                weights, _ = conv.parameters()
                weights = torch.reshape(A, weights.shape)
                conv.weight = torch.nn.Parameter(weights)
                res_m.append(conv)
            else:
                res_m.append(layer)
    return Sequential(*res_m)


# ============================ 3 Simple detector ===============================
def get_detections(detection_model, dictionary_of_images):
    """
    :param detection_model: trained fully convolutional detector model
    :param dictionary_of_images: dictionary of images in format
        {filename: ndarray}
    :return: detections in format {filename: detections}. detections is a N x 5
        array, where N is number of detections. Each detection is described
        using 5 numbers: [row, col, n_rows, n_cols, confidence].
    """
    import torch
    import matplotlib.pyplot as plt
    from matplotlib import patches

    detections = {}

    padded_images = []
    filenames = []
    for filename, image in dictionary_of_images.items():
        padded_image = torch.zeros((1, 220, 370))
        h, w = image.shape[:2]
        padded_image[..., :h, :w] = torch.from_numpy(image)
        padded_images.append(padded_image)
        filenames.append(filename)

    padded_image = torch.stack(padded_images, 0)

    with torch.no_grad():
        outputs = detection_model(padded_image)
        probs = torch.nn.Softmax(dim=1)
        detections = probs(outputs)
    
    outputs = {}
    for filename, detection in zip(filenames, detections):
        rows, cols = torch.where(detection[1] > 0.85)
        if len(rows) < 5:
            _, indices = detection[1].ravel().topk(5)
            _, w = detection[1].shape
            rows = indices // w
            cols = indices % w

        mult = 4

        outputs[filename] = [
            (row * mult, col * mult, 40, 100, detection[1, row, col]) 
            for row, col in zip(rows, cols)]

    return outputs


# =============================== 5 IoU ========================================
def calc_iou(first_bbox, second_bbox):
    """
    :param first bbox: bbox in format (row, col, n_rows, n_cols)
    :param second_bbox: bbox in format (row, col, n_rows, n_cols)
    :return: iou measure for two given bboxes
    """
    x1, y1, n_rows, n_cols = first_bbox
    x2 = x1 + n_rows
    y2 = y1 + n_cols
    x3, y3, n_rows, n_cols = second_bbox
    x4 = x3 + n_rows
    y4 = y3 + n_cols
    boxA = [x1, y1, x2, y2]
    boxB = [x3, y3, x4, y4]
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interArea = abs(max((xB - xA, 0)) * max((yB - yA), 0))
    if interArea == 0:
        return 0
    boxAArea = abs((boxA[2] - boxA[0]) * (boxA[3] - boxA[1]))
    boxBArea = abs((boxB[2] - boxB[0]) * (boxB[3] - boxB[1]))
    iou = interArea / float(boxAArea + boxBArea - interArea)
    return iou


# =============================== 6 AUC ========================================
def calc_auc(pred_bboxes, gt_bboxes):
    """
    :param pred_bboxes: dict of bboxes in format {filename: detections}
        detections is a N x 5 array, where N is number of detections. Each
        detection is described using 5 numbers: [row, col, n_rows, n_cols,
        confidence].
    :param gt_bboxes: dict of bboxes in format {filenames: bboxes}. bboxes is a
        list of tuples in format (row, col, n_rows, n_cols)
    :return: auc measure for given detections and gt
    """
    global_tp = []
    global_det = []
    gt_len = 0
    for name, det in pred_bboxes.items():
        gt = gt_bboxes[name].copy()                                                                                                                                                                                      
        gt_len += len(gt)
        tp = []
        fp = []
        for pred_bbox in sorted(det, key = lambda x: x[4], reverse=True):
            matched = []
            for i, gt_bbox in enumerate(gt):
                iou = calc_iou(pred_bbox[:4], gt_bbox[:4])
                if iou >= 0.5:
                    matched.append((i, iou))
            matched = sorted(matched, key = lambda x: x[1], reverse=True)
            if len(matched) > 0:
                tp.append(pred_bbox[4])
                del gt[matched[0][0]]
            else:
                fp.append(pred_bbox[4])
        global_tp.extend(tp)
        global_det.extend(tp)
        global_det.extend(fp)

    gtp = np.asarray(sorted(global_tp))
    gdet = np.asarray(sorted(global_det))
    p = []
    r = []
    for c in gdet:
        tp = (gtp >= c).sum()
        det = (gdet >= c).sum()
        recall = tp / gt_len
        prec = tp / det
        p.append(prec)
        r.append(recall)
    r.append(0)
    p.append(1)
    auc = 0
    for i in range(len(r) - 1):
        width = r[i] - r[i + 1]
        auc += width * (p[i + 1] + p[i]) / 2
    return auc


# =============================== 7 NMS ========================================
def nms(detections_dictionary, iou_thr=0.7):
    """
    :param detections_dictionary: dict of bboxes in format {fnames: detections}
        detections is a N x 5 array, where N is number of detections. Each
        detection is described using 5 numbers: [row, col, n_rows, n_cols,
        conf].
    :param iou_thr: IoU threshold for nearby detections
    :return: dict in same format as detections_dictionary where close detections
        are deleted
    """
    for fnames, detection in detections_dictionary.items():
        detection = sorted(detection, key=lambda item: item[4], reverse=True)
        to_delete = []
        for i in range(len(detection)):
            for j in range(i + 1, len(detection)):
                if i not in to_delete and calc_iou(detection[i][:4], detection[j][:4]) > iou_thr:
                    to_delete.append(j)
        for prediction in set(to_delete):
            detections_dictionary[fnames].remove(detection[prediction])
    return detections_dictionary
