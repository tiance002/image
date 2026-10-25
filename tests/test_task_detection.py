"""真实GT流的坐标、忽略区域及冻结检测器梯度契约。"""
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from saferestore.data.detection_gt import ACDCDetectionDataset, collate_detection_batch, letterbox_rgb
from saferestore.losses.task_detection import FrozenYoloTaskLoss, IgnoreRegionBCE, ignore_anchor_mask


def test_letterbox_and_flip_keep_rgb_and_gt_geometry():
    image = np.zeros((100, 200, 3), np.uint8)
    image[:] = [255, 0, 0]
    tensor, boxes, ignore = letterbox_rgb(image, [[20, 10, 100, 60]], [[0, 0, 20, 20]], 640)
    assert tensor.shape == (3, 640, 640)
    assert tensor[:, 320, 320].tolist() == [1, 0, 0]
    assert torch.allclose(boxes, torch.tensor([[.3, .425, .4, .25]]))
    assert torch.allclose(ignore, torch.tensor([[0, .25, .1, .35]]))
    _, flipped, ignored = letterbox_rgb(image, [[20, 10, 100, 60]], [[0, 0, 20, 20]], 640, True)
    assert torch.allclose(flipped, torch.tensor([[.7, .425, .4, .25]]))
    assert torch.allclose(ignored, torch.tensor([[.9, .25, 1, .35]]))


def test_ignore_regions_cover_center_and_intersection_without_cross_batch():
    points = torch.tensor([[5., 5.], [30., 30.], [50., 50.]])
    predictions = torch.tensor([[[4., 4., 6., 6.], [9., 9., 31., 31.], [45., 45., 55., 55.]]]).repeat(2, 1, 1)
    regions = torch.tensor([[0., 0., 10., 10.]])
    mask = ignore_anchor_mask(points, predictions, regions, torch.tensor([0]), 2)
    assert mask.tolist() == [[True, True, False], [False, False, False]]
    bce = IgnoreRegionBCE()
    bce.mask = mask
    logits = torch.zeros((2, 3, 2), requires_grad=True)
    targets = torch.zeros_like(logits)
    targets[0, 0, 1] = .7  # 忽略区域的已匹配真阳性不能被删掉。
    bce(logits, targets).sum().backward()
    assert logits.grad[0, 0, 0] == 0
    assert logits.grad[0, 0, 1] != 0
    assert logits.grad[0, 1].abs().sum() == 0
    assert logits.grad[1].abs().sum() > 0


def write_dataset(tmp_path, split="train"):
    import cv2
    name = f"rain/{split}/sequence/frame_rgb_anon.png"
    path = tmp_path / name
    path.parent.mkdir(parents=True)
    cv2.imencode(".png", np.zeros((100, 200, 3), np.uint8))[1].tofile(path)
    payload = {"images": [{"id": 1, "file_name": name, "width": 200, "height": 100}],
               "categories": [{"id": key, "name": value} for key, value in
                              {24: "person", 25: "rider", 26: "car", 27: "truck", 28: "bus", 31: "train", 32: "motorcycle", 33: "bicycle"}.items()],
               "annotations": [{"id": 1, "image_id": 1, "category_id": 28, "bbox": [10, 10, 50, 50], "iscrowd": 0},
                               {"id": 2, "image_id": 1, "category_id": 25, "bbox": [100, 10, 30, 50], "iscrowd": 0},
                               {"id": 3, "image_id": 1, "category_id": 26, "bbox": [130, 10, 30, 50], "iscrowd": 1}]}
    annotation = tmp_path / "gt.json"
    annotation.write_text(json.dumps(payload), encoding="utf-8")
    return annotation


def test_dataset_maps_bus_and_masks_crowd_rider(tmp_path):
    annotation = write_dataset(tmp_path)
    dataset = ACDCDetectionDataset(annotation, tmp_path, image_size=128)
    item = dataset[0]
    assert item["cls"].tolist() == [[5.]]
    assert len(item["ignore_boxes"]) == 2
    batch = collate_detection_batch([item, item])
    assert batch["batch_idx"].tolist() == [0, 1]
    assert batch["ignore_batch_idx"].tolist() == [0, 0, 1, 1]
    with pytest.raises(ValueError, match="重复"):
        ACDCDetectionDataset([annotation, annotation], tmp_path)


@pytest.mark.parametrize("split", ["val", "test", "train_ref"])
def test_dataset_rejects_nontrain_weather(tmp_path, split):
    annotation = write_dataset(tmp_path, split)
    with pytest.raises(ValueError, match="禁止"):
        ACDCDetectionDataset(annotation, tmp_path)


def test_missing_weights_fail_without_download(tmp_path):
    with pytest.raises(FileNotFoundError, match="禁止自动下载"):
        FrozenYoloTaskLoss(tmp_path / "missing.pt")


def test_real_native_yolo_loss_has_image_gradient_frozen_bn_and_class_mask():
    weights = Path(__file__).resolve().parents[1] / "yolov8n.pt"
    if not weights.is_file():
        pytest.skip("本地权重缺失，不下载")
    torch.set_num_threads(1)
    task = FrozenYoloTaskLoss(weights)
    before = {key: value.clone() for key, value in task.detector.state_dict().items()}
    task.train()
    assert all(not module.training for module in task.detector.modules())
    assert task.criterion.class_weights.flatten().nonzero().flatten().tolist() == [0, 1, 2, 3, 5, 6, 7]
    image = torch.rand((1, 3, 128, 128), requires_grad=True)
    batch = {"batch_idx": torch.tensor([0.]), "cls": torch.tensor([[5.]]),
             "bboxes": torch.tensor([[.5, .5, .5, .5]])}
    captured = {}
    def capture_scores(module, inputs, output):
        captured["scores"] = output[1]["scores"]
        captured["scores"].retain_grad()
    hook = task.detector.register_forward_hook(capture_scores)
    loss, stats = task(image, batch)
    loss.backward()
    hook.remove()
    assert torch.isfinite(loss) and image.grad.abs().sum() > 0
    assert torch.isfinite(image.grad).all()
    assert set(("box_loss", "cls_loss", "dfl_loss")).issubset(stats)
    unsupported = [index for index in range(80) if index not in [0, 1, 2, 3, 5, 6, 7]]
    assert captured["scores"].grad[:, unsupported].abs().sum() == 0
    assert captured["scores"].grad[:, [0, 1, 2, 3, 5, 6, 7]].abs().sum() > 0
    assert all(parameter.grad is None for parameter in task.detector.parameters())
    assert all(torch.equal(before[key], value) for key, value in task.detector.state_dict().items())
