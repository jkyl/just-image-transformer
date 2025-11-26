import io
from typing import Literal, TypedDict

import numpy as np
import PIL.Image
from datasets import Image as HFImage
from datasets import load_dataset
from grain.python import MapDataset, MapTransform, ReadOptions
from PIL import PngImagePlugin

PngImagePlugin.MAX_TEXT_CHUNK = 100 * (1024 * 1024)  # ty: ignore[invalid-assignment]


class RawImagenetSample(TypedDict):
    image: dict[str, bytes]
    label: int


class DecodeImage(MapTransform):
    def __init__(
        self,
        size: int,
        resampling: PIL.Image.Resampling = PIL.Image.Resampling.LANCZOS,
    ):
        self.size = size
        self.resampling = resampling

    def map(self, element: RawImagenetSample) -> tuple[np.ndarray, int]:
        image_bytes = element["image"]["bytes"]
        image = PIL.Image.open(io.BytesIO(image_bytes)).convert("RGB")
        width, height = image.size
        short_edge = min(width, height)
        left = (width - short_edge) // 2
        upper = (height - short_edge) // 2
        right = left + short_edge
        lower = upper + short_edge
        image = image.crop((left, upper, right, lower))
        image = image.resize((self.size, self.size), resample=self.resampling)
        image = np.ascontiguousarray(image, dtype=np.uint8)
        return image, element["label"]


class Normalize(MapTransform):
    def map(self, element: tuple[np.ndarray, int]) -> tuple[np.ndarray, int]:
        image, label = element
        image = image.astype(np.float32)
        image /= 127.5
        image -= 1
        return image, label


def imagenet(
    image_size: int,
    batch_size: int,
    seed: int,
    num_workers: int = 0,
    prefetch_buffer_size: int = 1,
    split: Literal["train", "validation", "test"] = "train",
):
    ds = load_dataset("ILSVRC/imagenet-1k", split=split)
    ds = ds.cast_column("image", HFImage(decode=False))
    ds = MapDataset.source(ds)  # ty: ignore[invalid-argument-type]
    ds = ds.seed(seed).shuffle().repeat(reseed_each_epoch=True)
    ds = ds.map(DecodeImage(size=image_size))
    ds = ds.batch(batch_size, drop_remainder=True)
    ds = ds.map(Normalize())
    ds = ds.to_iter_dataset(
        ReadOptions(num_threads=num_workers, prefetch_buffer_size=prefetch_buffer_size)
    )
    return ds
