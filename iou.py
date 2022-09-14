import numpy as np


def boxes(n,                      # number of boxes to generate
          img_w, img_h,           # width and height to fit the boxes into
          min_box_w, max_box_w,   # limits to boxes widths
          min_box_h, max_box_h,   # limits to boxes heights
          rng):                   # numpy.random.Generator object to be used
    """Random boxes generator.

    Returns a float32 numpy array (F-contiguous) representing bounding boxes
    according to the following specifications:

    '[...] of shape `(n, 4)` where `n` stands for the number of boxes in the
    given array, and `4` stands for `(x1, y1, x2, y2)`. `(x1, y1)` forms the
    topmost corner of the box and `(x2, y2)` forms the bottommost corner.`

    `[...] assume well defined boxes (`y1 < y2` and `x1 < X2`).`

    The boxes widths and heights are generated uniformly at random from the
    ranges [`min_box_w`, `max_box_w`] and [`min_box_h`, `max_box_h`
    respectively. The origin of each box is generated uniformly at random with
    domain restricted to the subspace of `img_w × img_h` valid for the given
    box width and height.
    """

    widths = rng.integers(min_box_w, max_box_w, size=n, endpoint=True)
    heights = rng.integers(min_box_h, max_box_h, size=n, endpoint=True)

    corners_x1 = rng.random(size=n) * (img_w - 1 - widths)
    corners_y1 = rng.random(size=n) * (img_h - 1 - heights)

    corners_x2 = corners_x1 + widths
    corners_y2 = corners_y1 + heights

    return np.round((corners_x1, corners_y1, corners_x2, corners_y2))\
        .T.astype(np.float32)


# single pair of boxes IoU
def iou(box1, box2):
    box1_x1, box1_y1, box1_x2, box1_y2 = box1
    box2_x1, box2_y1, box2_x2, box2_y2 = box2

    box1_delta_x = (box1_x2 - box1_x1)
    box1_delta_y = (box1_y2 - box1_y1)
    box2_delta_x = (box2_x2 - box2_x1)
    box2_delta_y = (box2_y2 - box2_y1)

    x_overlap = max(
        0,
        min(
            box1_delta_x,
            box2_delta_x,
            box1_x2 - box2_x1,
            box2_x2 - box1_x1,
        ),
    )
    y_overlap = max(
        0,
        min(
            box1_delta_y,
            box2_delta_y,
            box1_y2 - box2_y1,
            box2_y2 - box1_y1,
        ),
    )

    intersection_area = x_overlap * y_overlap

    box1_area = box1_delta_x * box1_delta_y
    box2_area = box2_delta_x * box2_delta_y

    union_area = box1_area + box2_area - intersection_area
    iou = intersection_area / union_area

    return iou


# nested loops reference version IoU matrix
def iou_matrix_looped_ref(boxes_1, boxes_2):
    iou_matrix = np.empty((boxes_1.shape[0], boxes_2.shape[0]), dtype=np.float32)

    for n, box1 in enumerate(boxes_1):
        for m, box2 in enumerate(boxes_2):
            iou_matrix[n, m] = iou(box1, box2)
    return iou_matrix


# manually vectorized IoU matrix using numpy (ver1)
def iou_matrix_np_opt1(boxes_1, boxes_2):
    n = boxes_1.shape[0]
    m = boxes_2.shape[0]

    boxes_1_x1, boxes_1_y1 = boxes_1[:, 0:1], boxes_1[:, 1:2]
    boxes_1_x2, boxes_1_y2 = boxes_1[:, 2:3], boxes_1[:, 3:4]
    boxes_2_x1, boxes_2_y1 = boxes_2[:, 0:1], boxes_2[:, 1:2]
    boxes_2_x2, boxes_2_y2 = boxes_2[:, 2:3], boxes_2[:, 3:4]

    boxes_1_delta_x = (boxes_1_x2 - boxes_1_x1)
    boxes_1_delta_y = (boxes_1_y2 - boxes_1_y1)
    boxes_2_delta_x = (boxes_2_x2 - boxes_2_x1)
    boxes_2_delta_y = (boxes_2_y2 - boxes_2_y1)

    boxes_1_area = boxes_1_delta_x * boxes_1_delta_y
    boxes_2_area = boxes_2_delta_x * boxes_2_delta_y

    x_overlap_matrix = np.zeros((4, n, m))
    x_overlap_matrix[0] = np.repeat(boxes_1_delta_x, m, axis=1)
    x_overlap_matrix[1] = np.repeat(boxes_2_delta_x.T, n, axis=0)
    x_overlap_matrix[2] = np.repeat(boxes_1_x2, m, axis=1) - np.repeat(boxes_2_x1.T, n, axis=0)
    x_overlap_matrix[3] = np.repeat(boxes_2_x2.T, n, axis=0) - np.repeat(boxes_1_x1, m, axis=1)
    x_overlap_matrix = np.clip(np.min(x_overlap_matrix, axis=0), 0, None)

    y_overlap_matrix = np.zeros((4, n, m))
    y_overlap_matrix[0] = np.repeat(boxes_1_delta_y, m, axis=1)
    y_overlap_matrix[1] = np.repeat(boxes_2_delta_y.T, n, axis=0)
    y_overlap_matrix[2] = np.repeat(boxes_1_y2, m, axis=1) - np.repeat(boxes_2_y1.T, n, axis=0)
    y_overlap_matrix[3] = np.repeat(boxes_2_y2.T, n, axis=0) - np.repeat(boxes_1_y1, m, axis=1)
    y_overlap_matrix = np.clip(np.min(y_overlap_matrix, axis=0), 0, None)

    intersection_area_matrix = x_overlap_matrix * y_overlap_matrix
    boxes_combined_area_matrix = np.repeat(boxes_1_area, m, axis=1) + np.repeat(boxes_2_area.T, n, axis=0)

    union_area_matrix = boxes_combined_area_matrix - intersection_area_matrix
    iou_matrix = intersection_area_matrix / union_area_matrix

    return iou_matrix


# manually vectorized IoU matrix using numpy (ver2)
def iou_matrix_np_opt2(boxes_1, boxes_2):
    n = boxes_1.shape[0]
    m = boxes_2.shape[0]

    boxes_1_x1, boxes_1_y1 = boxes_1[:, 0].reshape(-1, 1), boxes_1[:, 1].reshape(-1, 1)
    boxes_1_x2, boxes_1_y2 = boxes_1[:, 2].reshape(-1, 1), boxes_1[:, 3].reshape(-1, 1)
    boxes_2_x1, boxes_2_y1 = boxes_2[:, 0].reshape(-1, 1), boxes_2[:, 1].reshape(-1, 1)
    boxes_2_x2, boxes_2_y2 = boxes_2[:, 2].reshape(-1, 1), boxes_2[:, 3].reshape(-1, 1)

    boxes_1_delta_x = (boxes_1_x2 - boxes_1_x1)
    boxes_1_delta_y = (boxes_1_y2 - boxes_1_y1)
    boxes_2_delta_x = (boxes_2_x2 - boxes_2_x1)
    boxes_2_delta_y = (boxes_2_y2 - boxes_2_y1)
    boxes_1_area = boxes_1_delta_x * boxes_1_delta_y
    boxes_2_area = boxes_2_delta_x * boxes_2_delta_y

    x_overlap_matrix = np.zeros((4, n, m))
    x_overlap_matrix[0] = np.repeat(boxes_1_delta_x, m, axis=1)
    x_overlap_matrix[1] = np.repeat(boxes_2_delta_x.T, n, axis=0)
    x_overlap_matrix[2] = np.repeat(boxes_1_x2, m, axis=1) - np.repeat(boxes_2_x1.T, n, axis=0)
    x_overlap_matrix[3] = np.repeat(boxes_2_x2.T, n, axis=0) - np.repeat(boxes_1_x1, m, axis=1)
    x_overlap_matrix = np.min(x_overlap_matrix, axis=0)
    x_overlap_mask = (x_overlap_matrix <= 0)

    y_overlap_matrix = np.zeros((4, n, m))
    y_overlap_matrix[0] = np.repeat(boxes_1_delta_y, m, axis=1)
    y_overlap_matrix[1] = np.repeat(boxes_2_delta_y.T, n, axis=0)
    y_overlap_matrix[2] = np.repeat(boxes_1_y2, m, axis=1) - np.repeat(boxes_2_y1.T, n, axis=0)
    y_overlap_matrix[3] = np.repeat(boxes_2_y2.T, n, axis=0) - np.repeat(boxes_1_y1, m, axis=1)
    y_overlap_matrix = np.min(y_overlap_matrix, axis=0)
    y_overlap_mask = (y_overlap_matrix <= 0)

    xy_overlap_mask = x_overlap_mask | y_overlap_mask
    x_overlap_matrix = np.ma.array(x_overlap_matrix, mask=xy_overlap_mask)
    y_overlap_matrix = np.ma.array(y_overlap_matrix, mask=xy_overlap_mask)

    intersection_area_matrix = x_overlap_matrix * y_overlap_matrix
    boxes_combined_area_matrix = np.ma.array(np.repeat(boxes_1_area, m, axis=1) + np.repeat(boxes_2_area.T, n, axis=0), mask=xy_overlap_mask)

    union_area_matrix = boxes_combined_area_matrix - intersection_area_matrix

    iou_matrix = intersection_area_matrix / union_area_matrix

    return iou_matrix


# manually vectorized IoU matrix using numpy (ver3)
def iou_matrix_np_opt3(boxes_1, boxes_2):
    n = boxes_1.shape[0]
    m = boxes_2.shape[0]

    # step 1
    boxes_1_x1, boxes_1_y1 = boxes_1[:, 0].reshape(1, -1), boxes_1[:, 1].reshape(1, -1)
    boxes_1_x2, boxes_1_y2 = boxes_1[:, 2].reshape(1, -1), boxes_1[:, 3].reshape(1, -1)
    boxes_2_x1, boxes_2_y1 = boxes_2[:, 0].reshape(1, -1), boxes_2[:, 1].reshape(1, -1)
    boxes_2_x2, boxes_2_y2 = boxes_2[:, 2].reshape(1, -1), boxes_2[:, 3].reshape(1, -1)

    # step 2
    boxes_1_delta_x = (boxes_1_x2 - boxes_1_x1)
    boxes_1_delta_y = (boxes_1_y2 - boxes_1_y1)
    boxes_2_delta_x = (boxes_2_x2 - boxes_2_x1)
    boxes_2_delta_y = (boxes_2_y2 - boxes_2_y1)

    # step 3
    boxes_1_area = boxes_1_delta_x * boxes_1_delta_y
    boxes_2_area = boxes_2_delta_x * boxes_2_delta_y

    # step 4
    x_overlap_matrix = np.zeros((4, n, m))
    x_overlap_matrix[0] = np.repeat(boxes_1_delta_x, m, axis=0).T
    x_overlap_matrix[1] = np.repeat(boxes_2_delta_x, n, axis=0)
    x_overlap_matrix[2] = np.repeat(boxes_1_x2, m, axis=0).T - np.repeat(boxes_2_x1, n, axis=0)
    x_overlap_matrix[3] = np.repeat(boxes_2_x2, n, axis=0) - np.repeat(boxes_1_x1, m, axis=0).T
    x_overlap_matrix = np.clip(np.min(x_overlap_matrix, axis=0), 0, None)

    # step 5
    y_overlap_matrix = np.zeros((4, n, m))
    y_overlap_matrix[0] = np.repeat(boxes_1_delta_y, m, axis=0).T
    y_overlap_matrix[1] = np.repeat(boxes_2_delta_y, n, axis=0)
    y_overlap_matrix[2] = np.repeat(boxes_1_y2, m, axis=0).T - np.repeat(boxes_2_y1, n, axis=0)
    y_overlap_matrix[3] = np.repeat(boxes_2_y2, n, axis=0) - np.repeat(boxes_1_y1, m, axis=0).T
    y_overlap_matrix = np.clip(np.min(y_overlap_matrix, axis=0), 0, None)

    # step 6
    intersection_area_matrix = x_overlap_matrix * y_overlap_matrix
    boxes_combined_area_matrix = np.repeat(boxes_1_area, m, axis=0).T + np.repeat(boxes_2_area, n, axis=0)

    # step 7
    union_area_matrix = boxes_combined_area_matrix - intersection_area_matrix

    # step 8
    iou_matrix = intersection_area_matrix / union_area_matrix

    return iou_matrix
