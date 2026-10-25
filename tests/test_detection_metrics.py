import numpy as np

from saferestore.metrics import average_precision_101, box_iou


def test_box_iou_known_values() -> None:
    assert box_iou(np.array([0, 0, 10, 10]), np.array([0, 0, 10, 10])) == 1.0
    assert box_iou(np.array([0, 0, 10, 10]), np.array([20, 20, 30, 30])) == 0.0
    assert np.isclose(
        box_iou(np.array([0, 0, 10, 10]), np.array([5, 5, 15, 15])),
        25 / 175,
    )


def test_average_precision_perfect_and_empty() -> None:
    assert average_precision_101(np.array([]), np.array([])) == 0.0
    recall = np.array([0.5, 1.0])
    precision = np.array([1.0, 1.0])
    assert np.isclose(average_precision_101(recall, precision), 1.0)

