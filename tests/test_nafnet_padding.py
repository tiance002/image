import torch
import pytest
from saferestore.models.nafnet import build_nafnet


@pytest.mark.parametrize('shape',[(33,49),(32,48),(540,16)])
def test_residual_returns_original_shape_and_identity_at_initialization(shape):
    model=build_nafnet(variant='tiny').eval()
    image=torch.rand(1,3,*shape)
    with torch.no_grad():result=model(image)
    assert result.shape==image.shape
    assert torch.equal(result,image)


def test_nondivisible_shape_keeps_image_gradient():
    model=build_nafnet(variant='tiny').eval()
    image=torch.rand(1,3,33,49,requires_grad=True)
    model(image).square().mean().backward()
    assert torch.isfinite(image.grad).all()
    assert image.grad.abs().sum()>0
