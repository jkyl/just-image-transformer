# FSDP Checkpointing Implementation

## Background

This codebase uses JAX's **global view** paradigm with explicit sharding. The training loop, loss reduction, gradient handling, and optimizer state are all correct due to automatic all-reduce semantics.

The main gap is **checkpointing** - specifically, restoring to sharded topologies.

## Current Behavior

**Save:** `np.asarray(sharded_array)` gathers to host - works correctly.

**Restore:** Creates unsharded arrays, doesn't reshard to target topology. Training still works (all_gather handles it) but loses FSDP memory savings.

## Goal

Support flexible restore scenarios:
1. **Single device** - for inference, debugging
2. **Same topology** - resume training on same mesh
3. **Different topology** - scale up/down device count

Simpler than orbax while still handling topology flexibility.

## Approach

Use the **target model's sharding specs** as the source of truth. The target is already initialized with correct sharding for the current mesh.

### Implementation

Modify `restore()` to reshard loaded values:

```python
def restore(checkpoint_file: str | Path, target: nnx.Pytree) -> None:
    npz_archive = NpzArchive(checkpoint_file)
    state_dict = npz_archive.load()

    # Get target's state to extract sharding specs
    _, target_state = nnx.split(target)
    target_dict = nnx.to_pure_dict(target_state)

    # Reshard each loaded value to match target's sharding
    def reshard_to_target(loaded: np.ndarray, target_arr: Array) -> Array:
        target_sharding = jax.typeof(target_arr).sharding
        return reshard(jnp.asarray(loaded), target_sharding)

    restored_tree = jax.tree.map(
        reshard_to_target,
        state_dict_to_pytree(state_dict),
        target_dict,
    )

    _, state = nnx.split(target)
    nnx.replace_by_pure_dict(state, restored_tree)
    nnx.update(target, state)
```

### Key Points

1. **No stored sharding metadata needed** - target model has the correct specs
2. **Topology-agnostic** - works for any source→target topology change
3. **Minimal changes** - only modify `restore()`, save unchanged
4. **WSL path** - may need to gather before using CUDA interface (separate issue)

## Files to Modify

| File | Function | Change |
|------|----------|--------|
| `src/jit/serialization.py` | `restore()` | Reshard loaded values to target sharding |
| `src/jit/serialization.py` | `state_dict_to_pytree()` | Return numpy arrays (don't convert to jax yet) |
| `src/jit/serialization.py` | `device_to_host()` | Handle sharded arrays in WSL path |

## Verification

1. Initialize model on N devices
2. Save checkpoint
3. Restore to same model - verify shardings match via `jax.typeof(param.value).sharding`
4. Run forward pass - verify outputs match pre-save
5. (Optional) Test restore to different device count
