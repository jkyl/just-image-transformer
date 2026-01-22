# Claude Onboarding

Diffusion transformer for image generation, implemented in JAX with Flax NNX.

## Key Concept: JAX Explicit Sharding (Global View)

This repo uses JAX's **explicit sharding** with the **global view** paradigm. This is fundamentally different from PyTorch distributed:

- **Single process** sees the full logical shape of all arrays
- `jnp.mean()`, `jnp.sum()` on sharded arrays **automatically all-reduce**
- No manual `dist.all_reduce()` calls needed
- Optimizer state **inherits sharding** from parameters

Read this first: https://docs.jax.dev/en/latest/notebooks/explicit-sharding.html

## FSDP Sharding Strategy

The codebase implements **Hybrid Sharded Data Parallel (HSDP)** with a 2D mesh:

```python
# model.py
hsdp = ("dp", "fsdp")  # Shard over ALL devices

# train.py
mesh = jax.make_mesh(
    (data_size, fsdp_size),  # e.g., (1, 8) for 8 GPUs on one node
    hsdp,                     # axis names: ("dp", "fsdp")
    axis_types=(AxisType.Explicit, AxisType.Explicit),
)
jax.set_mesh(mesh)
```

Axis names use standard parallelism terminology:
- `"dp"` axis: data parallel (across nodes)
- `"fsdp"` axis: fully sharded data parallel (within node, up to 8 devices)
- `hsdp` = `("dp", "fsdp")` = hybrid sharded data parallel (all devices)

### Sharding Patterns

| What | Sharding | Why |
|------|----------|-----|
| Parameters | `P("fsdp")` | Shard within node for memory savings |
| Activations | `P(hsdp)` | Shard across all devices |
| Input data | `P(hsdp)` | Match activation sharding |

### Key Functions

```python
# Gather sharded param to replicated, cast to bf16 (used before compute)
def all_gather_bf16(param: nnx.Param) -> BFloat16[Array, "..."]:
    return reshard(param.value.astype(jnp.bfloat16), P())

# Attention is wrapped with shard_map to run independently per shard
default_attn_fn = make_sharded_attn_fn(
    cast(AttnFn, partial(jax.nn.dot_product_attention, implementation="cudnn"))
)
```

## Source Files

| File | Purpose |
|------|---------|
| `src/jit/model.py` | Model architecture (DiffusionTransformer, JustImageTransformer) |
| `src/jit/train.py` | Training loop, mesh setup, optimizer |
| `src/jit/dataset.py` | ImageNet data loading via grain |
| `src/jit/serialization.py` | Checkpoint save/restore (needs work for FSDP) |
| `src/jit/inference.py` | Sampling/generation |
| `src/jit/server.py` | Flask server for inference |
| `config/*.yaml` | Model configs (B/L/G/H sizes) |

## Commands

```bash
make          # Run linter (ruff) and type checker (ty)
make lint     # Just linting
make typecheck # Just type checking
```

## Known Gaps

See `fsdp-plan.md` for checkpointing implementation plan. The `restore()` function doesn't reshard loaded values to match target topology.

## Shape Annotations

The codebase uses jaxtyping for shape documentation:

```
B = batch size
T = sequence length
N = number of attention heads
H = dimensions of each attention head
D = model dimension = N * H
F = number of rope frequencies = H // 2
C = number of input/output dims
M = number of feedforward hidden dims
L = number of layers
```
