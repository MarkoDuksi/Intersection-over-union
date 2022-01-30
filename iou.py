#!/usr/bin/env python
# coding: utf-8

import time
import numpy as np

IMG_WIDTH = 9000
IMG_HEIGHT = 9000

MIN_BOX_WIDTH = 50
MIN_BOX_HEIGHT = 50
MAX_BOX_WIDTH = 80
MAX_BOX_HEIGHT = 80


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


def example_benchmark():
    """Used to get the scalability charts

    - dont't overdo the `num_boxes` warning 1: nested loops approach is more than 200 times slower than vectorized approaches
    - dont't overdo the `num_boxes` warning 2: vectorized approaches have O(n^2) space complexity with respect to `num_boxes
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


if __name__ == '__main__':
    example_benchmark()
