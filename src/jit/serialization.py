from collections.abc import Callable
from pathlib import Path
from typing import TypeVar

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx
from jaxtyping import Array, PyTree

# Separator used in state dict keys.
default_sep = "."


def pytree_to_state_dict(
    tree: PyTree[Array], separator: str = default_sep
) -> dict[str, np.ndarray]:
    return {
        jax.tree_util.keystr(path, simple=True, separator=separator): np.asarray(leaf)
        for path, leaf in jax.tree_util.tree_leaves_with_path(tree)
    }


def state_dict_to_pytree(
    state_dict: dict[str, np.ndarray], separator: str = default_sep
) -> PyTree[Array]:
    tree = {}
    for key, value in state_dict.items():
        parts = key.split(separator)
        current_node = tree
        for part in parts[:-1]:
            if part not in current_node:
                current_node[part] = {}
            current_node = current_node[part]
        current_node[parts[-1]] = jnp.asarray(value)
    return tree


class NpzArchive:
    def __init__(self, path: str | Path):
        self._path = Path(path).resolve()
        if self._path.suffix != ".npz":
            raise ValueError(
                f"NpzArchive path must end with .npz (got {self._path.suffix})"
            )

    def save(self, **data: np.ndarray) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(self._path, allow_pickle=False, **data)

    def load(self, *, mmap: bool = False) -> dict[str, np.ndarray]:
        return np.load(self._path, allow_pickle=False, mmap_mode="r" if mmap else None)


def save(obj: nnx.Pytree, checkpoint_file: str | Path) -> None:
    npz_archive = NpzArchive(checkpoint_file)
    _, state = nnx.split(obj)
    tree = nnx.to_pure_dict(state)
    state_dict = pytree_to_state_dict(tree)
    npz_archive.save(**state_dict)


T = TypeVar("T", bound=nnx.Pytree)


def load(checkpoint_file: str | Path, constructor: Callable[[], T]) -> T:
    npz_archive = NpzArchive(checkpoint_file)
    state_dict = npz_archive.load()
    restored_tree = state_dict_to_pytree(state_dict)
    abstract_obj = nnx.eval_shape(constructor)
    graphdef, abstract_state = nnx.split(abstract_obj)
    nnx.replace_by_pure_dict(abstract_state, restored_tree)
    obj = nnx.merge(graphdef, abstract_state)
    return obj


def restore(checkpoint_file: str | Path, target: nnx.Pytree) -> None:
    npz_archive = NpzArchive(checkpoint_file)
    state_dict = npz_archive.load()
    restored_tree = state_dict_to_pytree(state_dict)
    _, state = nnx.split(target)
    nnx.replace_by_pure_dict(state, restored_tree)
    nnx.update(target, state)
