import numpy as np
import numba as nb


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
    iou_matrix = np.empty(
        (boxes_1.shape[0], boxes_2.shape[0]),
        dtype=np.float32,
    )

    for n, box1 in enumerate(boxes_1):
        for m, box2 in enumerate(boxes_2):
            iou_matrix[n, m] = iou(box1, box2)

    return iou_matrix


# make input arrays of shapes (n, 4) and (m, 4) broadcastable to (n, m, 4)
def reshape_input(func):
    def wrapper(arr1, arr2):
        result = func(
            arr1.reshape(-1, 1, 4),
            arr2.reshape(1, -1, 4),
        )
        return result
    return wrapper


# numpy-vectorized reference IoU matrix
iou_matrix_np_vect_ref = reshape_input(
    np.vectorize(iou, signature='(n),(n)->()'),
)
iou_matrix_np_vect_ref.__name__ = 'iou_matrix_np_vect_ref'


# manually vectorized IoU matrix using numpy only
def iou_matrix_np_opt(boxes_1, boxes_2):
    X1, Y1, X2, Y2 = 0, 1, 2, 3

    boxes_1_x1 = np.ascontiguousarray(boxes_1[:, X1]).reshape(-1, 1)
    boxes_1_y1 = np.ascontiguousarray(boxes_1[:, Y1]).reshape(-1, 1)
    boxes_1_x2 = np.ascontiguousarray(boxes_1[:, X2]).reshape(-1, 1)
    boxes_1_y2 = np.ascontiguousarray(boxes_1[:, Y2]).reshape(-1, 1)

    boxes_2_x1 = np.ascontiguousarray(boxes_2[:, X1]).reshape(-1, 1)
    boxes_2_y1 = np.ascontiguousarray(boxes_2[:, Y1]).reshape(-1, 1)
    boxes_2_x2 = np.ascontiguousarray(boxes_2[:, X2]).reshape(-1, 1)
    boxes_2_y2 = np.ascontiguousarray(boxes_2[:, Y2]).reshape(-1, 1)

    boxes_1_delta_x = (boxes_1_x2 - boxes_1_x1)
    boxes_1_delta_y = (boxes_1_y2 - boxes_1_y1)
    boxes_2_delta_x = (boxes_2_x2 - boxes_2_x1)
    boxes_2_delta_y = (boxes_2_y2 - boxes_2_y1)

    boxes_1_area = boxes_1_delta_x * boxes_1_delta_y
    boxes_2_area = boxes_2_delta_x * boxes_2_delta_y

    min_within_delta_x = np.minimum(
        boxes_1_delta_x,
        boxes_2_delta_x.T,
    )
    min_across_delta_x = np.minimum(
        boxes_1_x2 - boxes_2_x1.T,
        boxes_2_x2.T - boxes_1_x1,
    )
    x_overlap = np.clip(
        np.minimum(
            min_within_delta_x,
            min_across_delta_x,
        ),
        0,
        None,
    )
    min_within_delta_y = np.minimum(
        boxes_1_delta_y,
        boxes_2_delta_y.T,
    )
    min_across_delta_y = np.minimum(
        boxes_1_y2 - boxes_2_y1.T,
        boxes_2_y2.T - boxes_1_y1,
    )
    y_overlap = np.clip(
        np.minimum(
            min_within_delta_y,
            min_across_delta_y,
        ),
        0,
        None,
    )

    intersection_area = x_overlap * y_overlap
    boxes_combined_area = boxes_1_area + boxes_2_area.T
    union_area = boxes_combined_area - intersection_area

    iou_matrix = intersection_area / union_area

    return iou_matrix


# numba-vectorized IoU matrix
@reshape_input
@nb.guvectorize(
    [(nb.float64[:], nb.float64[:], nb.float32[:])],
    '(n),(n)->()',
    nopython=True,
)
def iou_matrix_nb_vect(box1, box2, iou):
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

    iou[0] = intersection_area / union_area


iou_matrix_nb_vect.__name__ = 'iou_matrix_nb_vect'
