from dataclasses import dataclass
from typing import TypeVar, cast, get_type_hints

from omegaconf import MISSING, OmegaConf

T = TypeVar("T")


def schema(cls: T) -> T:
    for field_name in get_type_hints(cls):
        if hasattr(cls, field_name):
            attr = getattr(cls, field_name)
            if not callable(attr) and not isinstance(attr, property):
                raise TypeError("Default values are not allowed in schema.")
        else:
            setattr(cls, field_name, MISSING)
    return cls


@dataclass(frozen=True)
@schema
class Model:
    patch_size: int
    bottleneck_dim: int
    num_classes: int | None
    num_layers: int
    data_dim: int
    dim: int
    hidden_dim: int
    num_heads: int
    qk_norm: bool
    num_temb_freqs: int
    seed: int

    def __post_init__(self):
        assert self.dim % (2 * self.num_heads) == 0


@dataclass(frozen=True)
@schema
class Optimizer:
    lr: float
    weight_decay: float
    b1: float
    b2: float
    eps: float
    warmup_steps: int
    max_grad_norm: float


@dataclass(frozen=True)
@schema
class DataLoader:
    image_size: int
    batch_size: int
    num_workers: int
    prefetch_buffer_size: int
    split: str
    seed: int


@dataclass(frozen=True)
@schema
class Training:
    wandb_project: str
    seed: int
    checkpoint_directory: str
    save_interval: int
    resume_from: str | None
    max_checkpoints_to_keep: int | None


@dataclass(frozen=True)
@schema
class Config:
    model: Model
    optimizer: Optimizer
    dataloader: DataLoader
    training: Training

    @classmethod
    def from_yaml(cls, path: str) -> "Config":
        schema = OmegaConf.structured(cls)
        conf_from_yaml = OmegaConf.load(path)
        merged_conf = OmegaConf.merge(conf_from_yaml, schema)
        return cast("Config", OmegaConf.to_object(merged_conf))
