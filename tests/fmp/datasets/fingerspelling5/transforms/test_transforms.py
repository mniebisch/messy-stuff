import numpy as np
import pytest
from torchvision import tv_tensors

from fmp import transforms


def test_pad_to_size_square():
    image = np.zeros((3, 128, 128))
    image = tv_tensors.Image(image)

    size = 256
    padder = transforms.PadToSize(size=size)

    image_padded = padder(image)

    assert image_padded.shape[-2:] == (size, size)


def test_pad_to_size_rectangle():
    image = np.zeros((3, 128, 128))
    image = tv_tensors.Image(image)

    size = (256, 512)
    padder = transforms.PadToSize(size=size)

    image_padded = padder(image)

    assert image_padded.shape[-2:] == size


def test_pad_to_size_no_padding():
    image = np.zeros((3, 128, 128))
    image = tv_tensors.Image(image)

    size = 128
    padder = transforms.PadToSize(size=size)

    image_padded = padder(image)

    assert (image_padded == image).all()


def test_pad_to_size_raise_height_padding():
    image = np.zeros((3, 512, 128))
    image = tv_tensors.Image(image)

    size = 256
    padder = transforms.PadToSize(size=size)

    with pytest.raises(ValueError, match="Image size"):
        padder(image)


def test_pad_to_size_raise_width_padding():
    image = np.zeros((3, 128, 512))
    image = tv_tensors.Image(image)

    size = 256
    padder = transforms.PadToSize(size=size)

    with pytest.raises(ValueError, match="Image size"):
        padder(image)
