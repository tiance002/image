"""损失函数。"""

from .perceptual import DetectorFeatureLoss
from .restoration import SafeRestorationLoss

__all__ = ["DetectorFeatureLoss", "SafeRestorationLoss"]
