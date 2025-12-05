import ctypes
from collections.abc import Callable
from pathlib import Path
from shutil import which
from typing import TypeVar

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx
from jaxtyping import Array, PyTree

try:
    cudart = ctypes.CDLL("libcudart.so")
except OSError:
    cudart = ctypes.CDLL("libcudart.so.12")


# Separator used in state dict keys.
default_sep = "."


def device_to_host(device_array: Array) -> np.ndarray:
    # Short circuit on non-WSL systems: np.asarray works fine.
    if which("wslinfo") is None:
        return np.asarray(device_array)

    # We are on WSL which has issues virtualizing CUDA memory that cause OOM on device->host.
    device_array.block_until_ready()
    cuda_interface = device_array.__cuda_array_interface__  # ty: ignore[unresolved-attribute]
    device_ptr = cuda_interface["data"][0]
    shape = device_array.shape
    dtype = device_array.dtype
    size_bytes = device_array.nbytes
    host_array = np.empty(shape, dtype=dtype)
    host_ptr = host_array.ctypes.data
    cudaMemcpyDeviceToHost = 2
    error_code = cudart.cudaMemcpy(
        ctypes.c_void_p(host_ptr),
        ctypes.c_void_p(device_ptr),
        ctypes.c_size_t(size_bytes),
        ctypes.c_int(cudaMemcpyDeviceToHost),
    )
    if error_code != 0:
        cudart.cudaGetErrorString.restype = ctypes.c_char_p
        error_msg = cudart.cudaGetErrorString(error_code).decode("utf-8")
        raise RuntimeError(f"cudaMemcpy failed: {error_msg} (code {error_code})")

    return host_array


def pytree_to_state_dict(
    tree: PyTree[Array], separator: str = default_sep
) -> dict[str, np.ndarray]:
    return {
        jax.tree_util.keystr(path, simple=True, separator=separator): device_to_host(
            leaf
        )
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
