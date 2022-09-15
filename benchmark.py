from time import perf_counter

import numpy as np

import iou


def benchmark(func, min_num_boxes, max_num_boxes, step, params, seed=0):
    # reinitialize the generator on each call for repeatability
    rng = np.random.default_rng(seed=seed)

    print(f'\nbenchmarking `{func.__name__}`:')
    print('-----------------------------------------------------')
    print(f'number of boxes    overlap fraction    execution time')
    for num_boxes in range(min_num_boxes, max_num_boxes + 1, step):
        num_boxes_string = f'{num_boxes}x{num_boxes}'
        print(f'{num_boxes_string: ^15}', end='    ')

        boxes_1 = iou.boxes(num_boxes, *params, rng)
        boxes_2 = iou.boxes(num_boxes, *params, rng)
        start = perf_counter()
        iou_mat = func(boxes_1, boxes_2)
        time_string = f'{perf_counter() - start:.4f}s'

        print(f'{np.sum((iou_mat == 0))/iou_mat.size: ^16.1%}', end='    ')
        print(f' {time_string: ^13}')
    print()


if __name__ == '__main__':
    IMG_WIDTH = 1000
    IMG_HEIGHT = 1000

    dense_params = (
        IMG_WIDTH,
        IMG_HEIGHT,
        IMG_WIDTH // 1.9, IMG_WIDTH // 1.1,
        IMG_HEIGHT // 1.9, IMG_HEIGHT // 1.1,
    )

    sparse_params = (
        IMG_WIDTH,
        IMG_HEIGHT,
        IMG_WIDTH // 20, IMG_WIDTH // 10,
        IMG_HEIGHT // 20, IMG_HEIGHT // 10,
    )

    # benchmark overlapping bounding boxes
    benchmark(iou.iou_matrix_looped_ref, 50, 500, 50, dense_params)
    benchmark(iou.iou_matrix_np_vect_ref, 50, 500, 50, dense_params)
    # increase box count 100 times (10*10) for best performers
    benchmark(iou.iou_matrix_np_opt, 500, 5000, 500, dense_params)
    benchmark(iou.iou_matrix_nb_vect, 500, 5000, 500, dense_params)

    # benchmark mostly non-overlapping bounding boxes
    benchmark(iou.iou_matrix_looped_ref, 50, 500, 50, sparse_params)
    benchmark(iou.iou_matrix_np_vect_ref, 50, 500, 50, sparse_params)
    # increase box count 100 times (10*10) for best performers
    benchmark(iou.iou_matrix_np_opt, 500, 5000, 500, sparse_params)
    benchmark(iou.iou_matrix_nb_vect, 500, 5000, 500, sparse_params)
