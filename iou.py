#!/usr/bin/env python
# coding: utf-8

import time
import numpy as np

IMG_WIDTH = 2000
IMG_HEIGHT = 2000

MIN_BOX_WIDTH = 500
MIN_BOX_HEIGHT = 500
MAX_BOX_WIDTH = 800
MAX_BOX_HEIGHT = 800


# quick and dirty random bounding boxes
def get_boxes(num_boxes, img_width, img_height, min_box_width, min_box_height, max_box_width, max_box_height, seed=0):
    rng = np.random.default_rng(seed)
    max_half_box_width = max_box_width / 2
    max_half_box_height = max_box_height / 2

    centers_x = rng.random(size=num_boxes, dtype=np.float32) * (img_width - max_box_width) + max_half_box_width
    centers_y = rng.random(size=num_boxes, dtype=np.float32) * (img_height - max_box_height) + max_half_box_height
    widths = rng.random(size=num_boxes, dtype=np.float32) * (max_box_width - min_box_width) + min_box_width
    heights = rng.random(size=num_boxes, dtype=np.float32) * (max_box_height - min_box_height) + min_box_height

    corners_x1 = centers_x - (widths / 2)
    corners_y1 = centers_y - (heights / 2)
    corners_x2 = img_width - widths
    corners_y2 = img_height - heights

    corners_x1 = centers_x - (widths / 2)
    corners_y1 = centers_y - (heights / 2)
    corners_x2 = centers_x + (widths / 2)
    corners_y2 = centers_y + (heights / 2)

    return np.round(np.array(list(zip(corners_x1, corners_y1, corners_x2, corners_y2))))


# non-vectorized IoU
def get_iou(box1, box2):
    box1_x1, box1_y1, box1_x2, box1_y2 = box1
    box2_x1, box2_y1, box2_x2, box2_y2 = box2

    box1_delta_x = (box1_x2 - box1_x1)
    box1_delta_y = (box1_y2 - box1_y1)
    box2_delta_x = (box2_x2 - box2_x1)
    box2_delta_y = (box2_y2 - box2_y1)

    x_overlap = max(0, min(box1_delta_x, box2_delta_x, box1_x2 - box2_x1, box2_x2 - box1_x1))
    y_overlap = max(0, min(box1_delta_y, box2_delta_y, box1_y2 - box2_y1, box2_y2 - box1_y1))

    intersection_area = x_overlap * y_overlap

    box1_area = box1_delta_x * box1_delta_y
    box2_area = box2_delta_x * box2_delta_y

    union_area = box1_area + box2_area - intersection_area
    iou = intersection_area / union_area

    return iou


# vectorized IoU
def get_iou_matrix1(boxes_1, boxes_2):
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


# vectorized IoU using masked arrays
def get_iou_matrix2(boxes_1, boxes_2):
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


# vectorized IoU with intermediate timers
def get_iou_matrix1_timed(boxes_1, boxes_2):
    n = boxes_1.shape[0]
    m = boxes_2.shape[0]

    # step 1
    start = time.perf_counter()
    boxes_1_x1, boxes_1_y1 = boxes_1[:, 0:1], boxes_1[:, 1:2]
    boxes_1_x2, boxes_1_y2 = boxes_1[:, 2:3], boxes_1[:, 3:4]
    boxes_2_x1, boxes_2_y1 = boxes_2[:, 0:1], boxes_2[:, 1:2]
    boxes_2_x2, boxes_2_y2 = boxes_2[:, 2:3], boxes_2[:, 3:4]
    print(f'step 1: {time.perf_counter() - start:.4f}')

    # step 2
    start = time.perf_counter()
    boxes_1_delta_x = (boxes_1_x2 - boxes_1_x1)
    boxes_1_delta_y = (boxes_1_y2 - boxes_1_y1)
    boxes_2_delta_x = (boxes_2_x2 - boxes_2_x1)
    boxes_2_delta_y = (boxes_2_y2 - boxes_2_y1)
    print(f'step 2: {time.perf_counter() - start:.4f}')

    # step 3
    start = time.perf_counter()
    boxes_1_area = boxes_1_delta_x * boxes_1_delta_y
    boxes_2_area = boxes_2_delta_x * boxes_2_delta_y
    print(f'step 3: {time.perf_counter() - start:.4f}')

    # step 4
    start = time.perf_counter()
    x_overlap_matrix = np.zeros((4, n, m))
    x_overlap_matrix[0] = np.repeat(boxes_1_delta_x, m, axis=1)
    x_overlap_matrix[1] = np.repeat(boxes_2_delta_x.T, n, axis=0)
    x_overlap_matrix[2] = np.repeat(boxes_1_x2, m, axis=1) - np.repeat(boxes_2_x1.T, n, axis=0)
    x_overlap_matrix[3] = np.repeat(boxes_2_x2.T, n, axis=0) - np.repeat(boxes_1_x1, m, axis=1)
    x_overlap_matrix = np.clip(np.min(x_overlap_matrix, axis=0), 0, None)
    print(f'step 4: {time.perf_counter() - start:.4f}')

    # step 5
    start = time.perf_counter()
    y_overlap_matrix = np.zeros((4, n, m))
    y_overlap_matrix[0] = np.repeat(boxes_1_delta_y, m, axis=1)
    y_overlap_matrix[1] = np.repeat(boxes_2_delta_y.T, n, axis=0)
    y_overlap_matrix[2] = np.repeat(boxes_1_y2, m, axis=1) - np.repeat(boxes_2_y1.T, n, axis=0)
    y_overlap_matrix[3] = np.repeat(boxes_2_y2.T, n, axis=0) - np.repeat(boxes_1_y1, m, axis=1)
    y_overlap_matrix = np.clip(np.min(y_overlap_matrix, axis=0), 0, None)
    print(f'step 5: {time.perf_counter() - start:.4f}')

    # step 6
    start = time.perf_counter()
    intersection_area_matrix = x_overlap_matrix * y_overlap_matrix
    boxes_combined_area_matrix = np.repeat(boxes_1_area, m, axis=1) + np.repeat(boxes_2_area.T, n, axis=0)
    print(f'step 6: {time.perf_counter() - start:.4f}')

    # step 7
    start = time.perf_counter()
    union_area_matrix = boxes_combined_area_matrix - intersection_area_matrix
    print(f'step 7: {time.perf_counter() - start:.4f}')

    # step 8
    start = time.perf_counter()
    iou_matrix = intersection_area_matrix / union_area_matrix
    print(f'step 8: {time.perf_counter() - start:.4f}')

    return iou_matrix


# vectorized IoU with intermediate timers, optimized
def get_iou_matrix1_opt1_timed(boxes_1, boxes_2):
    n = boxes_1.shape[0]
    m = boxes_2.shape[0]

    # step 1
    start = time.perf_counter()
    boxes_1_x1, boxes_1_y1 = boxes_1[:, 0].reshape(1, -1), boxes_1[:, 1].reshape(1, -1)
    boxes_1_x2, boxes_1_y2 = boxes_1[:, 2].reshape(1, -1), boxes_1[:, 3].reshape(1, -1)
    boxes_2_x1, boxes_2_y1 = boxes_2[:, 0].reshape(1, -1), boxes_2[:, 1].reshape(1, -1)
    boxes_2_x2, boxes_2_y2 = boxes_2[:, 2].reshape(1, -1), boxes_2[:, 3].reshape(1, -1)
    print(f'step 1: {time.perf_counter() - start:.4f}')

    # step 2
    start = time.perf_counter()
    boxes_1_delta_x = (boxes_1_x2 - boxes_1_x1)
    boxes_1_delta_y = (boxes_1_y2 - boxes_1_y1)
    boxes_2_delta_x = (boxes_2_x2 - boxes_2_x1)
    boxes_2_delta_y = (boxes_2_y2 - boxes_2_y1)
    print(f'step 2: {time.perf_counter() - start:.4f}')

    # step 3
    start = time.perf_counter()
    boxes_1_area = boxes_1_delta_x * boxes_1_delta_y
    boxes_2_area = boxes_2_delta_x * boxes_2_delta_y
    print(f'step 3: {time.perf_counter() - start:.4f}')

    # step 4
    start = time.perf_counter()
    x_overlap_matrix = np.zeros((4, n, m))
    x_overlap_matrix[0] = np.repeat(boxes_1_delta_x, m, axis=0).T
    x_overlap_matrix[1] = np.repeat(boxes_2_delta_x, n, axis=0)
    x_overlap_matrix[2] = np.repeat(boxes_1_x2, m, axis=0).T - np.repeat(boxes_2_x1, n, axis=0)
    x_overlap_matrix[3] = np.repeat(boxes_2_x2, n, axis=0) - np.repeat(boxes_1_x1, m, axis=0).T
    x_overlap_matrix = np.clip(np.min(x_overlap_matrix, axis=0), 0, None)
    print(f'step 4: {time.perf_counter() - start:.4f}')

    # step 5
    start = time.perf_counter()
    y_overlap_matrix = np.zeros((4, n, m))
    y_overlap_matrix[0] = np.repeat(boxes_1_delta_y, m, axis=0).T
    y_overlap_matrix[1] = np.repeat(boxes_2_delta_y, n, axis=0)
    y_overlap_matrix[2] = np.repeat(boxes_1_y2, m, axis=0).T - np.repeat(boxes_2_y1, n, axis=0)
    y_overlap_matrix[3] = np.repeat(boxes_2_y2, n, axis=0) - np.repeat(boxes_1_y1, m, axis=0).T
    y_overlap_matrix = np.clip(np.min(y_overlap_matrix, axis=0), 0, None)
    print(f'step 5: {time.perf_counter() - start:.4f}')

    # step 6
    start = time.perf_counter()
    intersection_area_matrix = x_overlap_matrix * y_overlap_matrix
    boxes_combined_area_matrix = np.repeat(boxes_1_area, m, axis=0).T + np.repeat(boxes_2_area, n, axis=0)
    print(f'step 6: {time.perf_counter() - start:.4f}')

    # step 7
    start = time.perf_counter()
    union_area_matrix = boxes_combined_area_matrix - intersection_area_matrix
    print(f'step 7: {time.perf_counter() - start:.4f}')

    # step 8
    start = time.perf_counter()
    iou_matrix = intersection_area_matrix / union_area_matrix
    print(f'step 8: {time.perf_counter() - start:.4f}')

    return iou_matrix


def get_iou_matrix1_opt1(boxes_1, boxes_2):
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


def example_benchmark1():
    """Used to get the scalability charts

    - dont't overdo the `num_boxes` warning 1: nested loops approach is more than 200 times slower than vectorized approaches
    - dont't overdo the `num_boxes` warning 2: vectorized approaches have higher memory requirements
    """

    # 16 GB of RAM on my system is more than enough to process 14000 x 14000 bounding boxes
    print(f'sqrt(num_boxes);time/s')
    for num_boxes in range(200, 14001, 200):
        print(f'{num_boxes}', end=';')
        boxes_1 = get_boxes(num_boxes, IMG_WIDTH, IMG_HEIGHT, MIN_BOX_WIDTH, MIN_BOX_HEIGHT, MAX_BOX_WIDTH, MAX_BOX_HEIGHT, seed=0)
        boxes_2 = get_boxes(num_boxes, IMG_WIDTH, IMG_HEIGHT, MIN_BOX_WIDTH, MIN_BOX_HEIGHT, MAX_BOX_WIDTH, MAX_BOX_HEIGHT, seed=1)

        # # nested loops version
        # start = time.perf_counter()
        # iou_matrix = np.zeros((num_boxes, num_boxes))
        # for n, box1 in enumerate(boxes_1):
        #     for m, box2 in enumerate(boxes_2):
        #         iou_matrix[n, m] = get_iou(box1, box2)
        # print(f'{time.perf_counter() - start:.4f}')

        # vectorized version 1 - the best performer
        start = time.perf_counter()
        get_iou_matrix1(boxes_1, boxes_2)
        print(f'{time.perf_counter() - start:.4f}')

        # # vectorized version using masked arrays - more than 10 % slower than `get_iou_matrix1`
        # start = time.perf_counter()
        # get_iou_matrix2(boxes_1, boxes_2)
        # print(f'{time.perf_counter() - start:.4f}')


def get_timings():
    """Used to examine the time needed for various substeps within a function call
    """
    num_boxes_1 = 2000
    num_boxes_2 = 2000
    boxes_1 = get_boxes(num_boxes_1, IMG_WIDTH, IMG_HEIGHT, MIN_BOX_WIDTH, MIN_BOX_HEIGHT, MAX_BOX_WIDTH, MAX_BOX_HEIGHT, seed=0)
    boxes_2 = get_boxes(num_boxes_2, IMG_WIDTH, IMG_HEIGHT, MIN_BOX_WIDTH, MIN_BOX_HEIGHT, MAX_BOX_WIDTH, MAX_BOX_HEIGHT, seed=1)

    # vectorized IoU with intermediate timers
    start_total = time.perf_counter()
    iou_matrix1 = get_iou_matrix1(boxes_1, boxes_2)
    print(f'get_iou_matrix1 (vectorized solution) for 2k x 2k boxes: {time.perf_counter() - start_total:.4f} s')

    # vectorized IoU with intermediate timers, optimization attempt 1
    start_total = time.perf_counter()
    iou_matrix1_opt1 = get_iou_matrix1_opt1(boxes_1, boxes_2)
    print(f'get_iou_matrix1_opt1 (vectorized/optimized solution) for 2k x 2k boxes: {time.perf_counter() - start_total:.4f} s')

    print(f'{np.allclose(iou_matrix1, iou_matrix1_opt1) = }')
    print(f'sparsity = {np.sum((iou_matrix1 == 0))/iou_matrix1.size:.2%}')


if __name__ == '__main__':
    # example_benchmark1()
    get_timings()
