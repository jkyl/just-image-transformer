"""
Shapes legend:

B = batch size
T = sequence length
N = number of attention heads
H = dimensions of each attention head
D = model dimension = N * H
F = number of rope frequencies = H // 2
C = number of input/output dims
M = number of feedforward hidden dims
L = number of layers

cf.
  * https://docs.kidger.site/jaxtyping/api/array/
  * https://github.com/jax-ml/jax/blob/4c6bb8/jax/_src/nn/functions.py#L1075-L1084
  * https://jax-ml.github.io/scaling-book/transformers/#transformer-accounting
"""

from functools import partial
from typing import Protocol

import jax
import jax.numpy as jnp
from beartype import beartype
from flax import nnx
from jax.ad_checkpoint import checkpoint_name as ckpt
from jax.nn.initializers import glorot_normal, variance_scaling
from jax.sharding import PartitionSpec as P
from jax.sharding import auto_axes, reshard
from jaxtyping import (
    Array,
    BFloat16,
    Complex64,
    Float32,
    Int32,
    PRNGKeyArray,
    jaxtyped,
)

# Runtime type checker for JAX functions.
typechecked = jaxtyped(typechecker=beartype)


# Shorthand for sharding over all devices.
fsdp = ("data", "hsdp")


class AttnFn(Protocol):
    def __call__(
        self,
        query: BFloat16[Array, "B T N H"],
        key: BFloat16[Array, "B T N H"],
        value: BFloat16[Array, "B T N H"],
    ) -> BFloat16[Array, "B T N H"]: ...


def make_sharded_attn_fn(attn_fn: AttnFn) -> AttnFn:
    return jax.shard_map(
        attn_fn,
        in_specs=(P(fsdp), P(fsdp), P(fsdp)),
        out_specs=P(fsdp),
    )


# Sensible default for bidirectional attention on GPU.
default_attn_fn = make_sharded_attn_fn(
    partial(jax.nn.dot_product_attention, implementation="cudnn")
)


# Reuse the same default eps in functional and modular RMS norms.
default_rms_norm_eps = 1e-6


@typechecked
def rms_norm(
    x: BFloat16[Array, "... dim"],
    eps: float = default_rms_norm_eps,
) -> BFloat16[Array, "... dim"]:
    return x * (
        jax.lax.rsqrt((x.astype(jnp.float32) ** 2).mean(-1, keepdims=True) + eps)
    ).astype(jnp.bfloat16)


class RMSNorm(nnx.Module):
    def __init__(self, dim: int, eps: float = default_rms_norm_eps):
        self.weight = nnx.Param(jnp.ones(dim, out_sharding=P()))
        self.eps = eps

    @typechecked
    def __call__(self, x: BFloat16[Array, "... dim"]) -> BFloat16[Array, "... dim"]:
        return rms_norm(x, self.eps) * self.weight.value.astype(jnp.bfloat16)


@typechecked
def apply_rope(
    x: BFloat16[Array, "B T N H"],
    rope: Complex64[Array, "T N F"],
) -> BFloat16[Array, "B T N H"]:
    @auto_axes
    def _apply_rope(_x, _rope):
        x_complex = _x.astype(jnp.float32).view(dtype=jnp.complex64)
        x_rotated = x_complex * _rope
        return x_rotated.view(dtype=jnp.float32).astype(jnp.bfloat16)

    return _apply_rope(x, rope, out_sharding=jax.typeof(x).sharding)


@typechecked
def all_gather_bf16(param: nnx.Param[Float32[Array, "..."]]) -> BFloat16[Array, "..."]:
    return reshard(param.value.astype(jnp.bfloat16), P())


class MultiHeadAttention(nnx.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        attn_fn: AttnFn,
        *,
        rngs: nnx.Rngs,
    ):
        self.dim = dim
        self.num_heads = num_heads
        self.attn_fn = attn_fn
        self.W_qkv = nnx.Param(
            glorot_normal(in_axis=0, out_axis=(1, 2, 3))(
                key=rngs(),
                shape=(dim, 3, self.num_heads, self.head_dim),
                out_sharding=P("hsdp"),
            )
        )
        self.W_out = nnx.Param(
            glorot_normal(in_axis=(0, 1), out_axis=2)(
                key=rngs(),
                shape=(self.num_heads, self.head_dim, self.dim),
                out_sharding=P(None, "hsdp"),
            )
        )

    @property
    def head_dim(self) -> int:
        return self.dim // self.num_heads

    @typechecked
    def __call__(
        self,
        x: BFloat16[Array, "B T D"],
        rope: Complex64[Array, "T N F"] | None,
        qk_norm: bool,
    ) -> BFloat16[Array, "B T D"]:
        q, k, v = jnp.einsum(
            "BTD, D3NH -> 3BTNH",
            x,
            all_gather_bf16(self.W_qkv),
            out_sharding=P(None, fsdp),
        )
        if rope is not None:
            q, k = map(lambda t: apply_rope(t, rope), (q, k))
        if qk_norm:
            q, k = map(rms_norm, (q, k))

        attn = ckpt(self.attn_fn(q, k, v), "attn")
        return jnp.einsum(
            "BTNH, NHD -> BTD",
            attn,
            all_gather_bf16(self.W_out),
            out_sharding=P(fsdp),
        )


class FeedForward(nnx.Module):
    def __init__(self, dim: int, hidden_dim: int, *, rngs: nnx.Rngs):
        self.W_up = nnx.Param(
            glorot_normal(in_axis=0, out_axis=(1, 2))(
                rngs(), (dim, 2, hidden_dim), out_sharding=P("hsdp")
            )
        )
        self.W_down = nnx.Param(
            glorot_normal(in_axis=0, out_axis=1)(
                rngs(), (hidden_dim, dim), out_sharding=P("hsdp")
            )
        )

    @typechecked
    def __call__(self, x: BFloat16[Array, "B T D"]) -> BFloat16[Array, "B T D"]:
        h, gate = jnp.einsum(
            "BTD, D2M -> 2BTM",
            x,
            all_gather_bf16(self.W_up),
            out_sharding=P(None, fsdp),
        )
        h = gate * nnx.silu(h)
        return jnp.dot(
            h,
            all_gather_bf16(self.W_down),
            out_sharding=P(fsdp),
        )


class TransformerBlock(nnx.Module):
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        num_heads: int,
        attn_fn: AttnFn,
        *,
        rngs: nnx.Rngs,
    ):
        self.attn_norm = RMSNorm(dim)
        self.attn = MultiHeadAttention(dim, num_heads, attn_fn, rngs=rngs)
        self.ff_norm = RMSNorm(dim)
        self.ff = FeedForward(dim, hidden_dim, rngs=rngs)

    @typechecked
    def __call__(
        self,
        x: BFloat16[Array, "B T D"],
        rope: Complex64[Array, "T N F"] | None,
        qk_norm: bool,
    ) -> BFloat16[Array, "B T D"]:
        x += self.attn(self.attn_norm(x), rope, qk_norm)
        x += self.ff(self.ff_norm(x))
        return x


class Backbone(nnx.Module):
    def __init__(
        self,
        num_layers: int,
        dim: int,
        hidden_dim: int,
        num_heads: int,
        attn_fn: AttnFn,
        *,
        rngs: nnx.Rngs,
    ):
        @nnx.vmap
        def create_stack(key: PRNGKeyArray):
            return TransformerBlock(
                dim, hidden_dim, num_heads, attn_fn, rngs=nnx.Rngs(key)
            )

        self.blocks = create_stack(jax.random.split(rngs(), num_layers))

    @typechecked
    def __call__(
        self,
        x: BFloat16[Array, "B T D"],
        rope: Complex64[Array, "L T N F"] | None,
        qk_norm: bool,
    ) -> BFloat16[Array, "B T D"]:
        @nnx.scan(in_axes=(0, nnx.Carry, 0), out_axes=nnx.Carry)
        @nnx.remat(
            prevent_cse=False,
            policy=jax.checkpoint_policies.save_from_both_policies(
                jax.checkpoint_policies.dots_saveable,
                jax.checkpoint_policies.save_only_these_names("attn"),
            ),
        )
        def forward(layer, _x, per_layer_rope):
            return layer(_x, per_layer_rope, qk_norm)

        return forward(self.blocks, x, rope)


class TimestepEmbedding(nnx.Module):
    def __init__(self, dim: int, num_freqs: int, *, rngs: nnx.Rngs):
        self.weight = nnx.Param(
            glorot_normal()(rngs(), (2 * num_freqs, dim), out_sharding=P("hsdp"))
        )
        self.freqs = nnx.Param(jnp.zeros(num_freqs, out_sharding=P()))

    @typechecked
    def __call__(self, t: Float32[Array, " B"]) -> BFloat16[Array, "B D"]:
        embeddings = jnp.exp(1j * self.freqs.value * t[:, None])
        return jnp.dot(
            embeddings.view(dtype=jnp.float32).astype(jnp.bfloat16),
            all_gather_bf16(self.weight),
            out_sharding=P(fsdp),
        )


class Rope(nnx.Module):
    def __init__(self, num_layers: int, num_heads: int, head_dim: int, num_dims: int):
        self.num_dims = num_dims
        self.freqs = nnx.Param(
            jnp.zeros(
                (num_layers, num_dims, num_heads, head_dim // 2),
                out_sharding=P(),
            )
        )

    @typechecked
    def __call__(self, *sizes: int) -> Complex64[Array, "L T N F"]:
        assert len(sizes) == self.num_dims
        coords = jnp.stack(
            jnp.meshgrid(
                *[jnp.arange(s, dtype=jnp.float32) for s in sizes], indexing="ij"
            )
        )
        phase = jnp.einsum("d..., LdNF -> L...NF", coords, self.freqs.value)
        embeddings = jnp.exp(1j * phase)
        num_layers, _, num_heads, feat_dim = self.freqs.shape
        return jnp.reshape(embeddings, (num_layers, -1, num_heads, feat_dim))


def pad_sequence(
    x: Array,
    amt: int,
    fill: float | complex = 0.0,
    left: bool = True,
    axis: int = 1,
) -> Array:
    pad = [
        (amt if left and (hit := a == axis) else 0, amt if not left and hit else 0)
        for a in range(x.ndim)
    ]
    return jnp.pad(x, pad, constant_values=fill)


class DiffusionTransformer(nnx.Module):
    def __init__(
        self,
        *,
        input_dim: int,
        output_dim: int,
        num_layers: int,
        dim: int,
        hidden_dim: int,
        num_heads: int,
        attn_fn: AttnFn,
        qk_norm: bool,
        num_temb_freqs: int | None,
        num_coord_dims: int,
        rngs: nnx.Rngs,
    ):
        self.num_layers = num_layers
        self.dim = dim
        self.num_heads = num_heads
        self.qk_norm = qk_norm
        self.rope = Rope(
            num_layers, num_heads, dim // num_heads, num_dims=num_coord_dims
        )
        self.timestep_embedding = (
            TimestepEmbedding(dim, num_temb_freqs, rngs=rngs)
            if num_temb_freqs is not None
            else None
        )
        self.W_in = nnx.Param(
            glorot_normal()(rngs(), (input_dim, dim), out_sharding=P("hsdp"))
        )
        self.backbone = Backbone(
            num_layers=num_layers,
            dim=dim,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            attn_fn=attn_fn,
            rngs=rngs,
        )
        self.final_norm = RMSNorm(dim)
        self.W_out = nnx.Param(
            glorot_normal()(rngs(), (dim, output_dim), out_sharding=P("hsdp"))
        )

    @property
    def head_dim(self) -> int:
        return self.dim // self.num_heads

    @property
    def num_freqs(self) -> int:
        return self.head_dim // 2

    @typechecked
    def __call__(
        self,
        x: BFloat16[Array, "B *spatial Cin"],
        t: Float32[Array, " B"] | None,
        cond: BFloat16[Array, "B D"] | None = None,
    ) -> BFloat16[Array, "B *spatial Cout"]:
        # Read off the input shape and rearrange to a sequence.
        b, *spatial, c = x.shape
        x = jnp.reshape(x, (b, -1, c), out_sharding=P(fsdp))

        # Up-project the input onto the model dim.
        x = jnp.dot(
            x,
            all_gather_bf16(self.W_in),
            out_sharding=P(fsdp),
        )

        # Build sequence with optional timestep and conditioning embeddings.
        prefix_tokens = []
        if t is not None:
            assert self.timestep_embedding is not None
            temb = self.timestep_embedding(t)
            prefix_tokens.append(temb[:, None])
        if cond is not None:
            prefix_tokens.append(cond[:, None])

        if prefix_tokens:
            seq = jnp.concatenate(prefix_tokens + [x], axis=1)
            num_prefix = len(prefix_tokens)
        else:
            seq = x
            num_prefix = 0

        # Determine padding: 4 cases based on number of prefix tokens (0, 1, or 2).
        # Goal is to make sequence length even and provide attention sink.
        if num_prefix == 0:
            pad = 2  # No prefix tokens: pad with 2
        elif num_prefix == 1:
            pad = 1  # One prefix token: pad with 1 to make even
        else:  # num_prefix == 2
            pad = 2  # Two prefix tokens: pad with 2 for sink

        # Add padding to make the sequence even and provide an attention sink.
        seq = pad_sequence(seq, pad)

        # Left-pad the rope embeddings with 1+0j (0-degree rotation).
        rope = pad_sequence(self.rope(*spatial), pad + num_prefix, 1 + 0j)

        # Apply the transformer, and slice off the padding and prefix token positions.
        x = self.final_norm(
            self.backbone(seq, rope, self.qk_norm)[:, pad + num_prefix :]
        )

        # Project back to the data dim and rearrange back to original shape.
        x = jnp.dot(
            x,
            all_gather_bf16(self.W_out),
            out_sharding=P(fsdp),
        )
        x = jnp.reshape(
            x,
            (b, *spatial, -1),
            out_sharding=P(fsdp),
        )
        return x


class JustImageTransformer(DiffusionTransformer):
    def __init__(
        self,
        *,
        patch_size: int,
        data_dim: int,
        bottleneck_dim: int,
        num_classes: int | None,
        num_layers: int,
        dim: int,
        hidden_dim: int,
        num_heads: int,
        qk_norm: bool,
        num_temb_freqs: int | None,
        rngs: nnx.Rngs,
    ):
        self.patch_size = patch_size
        self.data_dim = data_dim
        self.bottleneck_dim = bottleneck_dim
        self.num_classes = num_classes
        super().__init__(
            input_dim=bottleneck_dim,
            output_dim=bottleneck_dim,
            num_layers=num_layers,
            dim=dim,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            qk_norm=qk_norm,
            num_temb_freqs=num_temb_freqs,
            num_coord_dims=2,
            attn_fn=default_attn_fn,
            rngs=rngs,
        )
        self.bottleneck_down = nnx.Param(
            glorot_normal()(
                rngs(), (self.patch_dim, bottleneck_dim), out_sharding=P("hsdp")
            )
        )
        self.bottleneck_up = nnx.Param(
            glorot_normal()(
                rngs(), (bottleneck_dim, self.patch_dim), out_sharding=P("hsdp")
            )
        )
        if num_classes is not None:
            self.class_embedding = nnx.Param(
                variance_scaling(1.0, "fan_in", "normal", out_axis=0)(
                    rngs(), (num_classes + 1, dim), out_sharding=P()
                )
            )
        else:
            self.class_embedding = None

    @property
    def patch_dim(self) -> int:
        return self.patch_size**2 * self.data_dim

    def patchify(
        self, x: BFloat16[Array, "B height width channels"]
    ) -> BFloat16[Array, "B h w patch_dim"]:
        b, h, w, c = x.shape
        ps = self.patch_size
        x = jnp.reshape(
            x,
            (b, h // ps, ps, w // ps, ps, c),
            out_sharding=P(fsdp),
        )
        x = jnp.transpose(x, (0, 1, 3, 2, 4, 5))
        x = jnp.reshape(
            x,
            (b, h // ps, w // ps, ps * ps * c),
            out_sharding=P(fsdp),
        )
        return x

    def unpatchify(
        self, x: BFloat16[Array, "B h w patch_dim"]
    ) -> BFloat16[Array, "B height width channels"]:
        b, h, w, _ = x.shape
        ps = self.patch_size
        c = self.data_dim
        x = jnp.reshape(
            x,
            (b, h, w, ps, ps, c),
            out_sharding=P(fsdp),
        )
        x = jnp.transpose(x, (0, 1, 3, 2, 4, 5))
        x = jnp.reshape(
            x,
            (b, h * ps, w * ps, c),
            out_sharding=P(fsdp),
        )
        return x

    @typechecked
    def __call__(
        self,
        x: BFloat16[Array, "B height width channels"],
        t: Float32[Array, " B"] | None,
        c: Int32[Array, " B"] | None,
    ) -> BFloat16[Array, "B height width channels"]:
        if c is not None:
            assert self.class_embedding is not None
            cond = (
                self.class_embedding.value.at[c]
                .get(out_sharding=P(fsdp))
                .astype(jnp.bfloat16)
            )
        else:
            cond = None
        x = self.patchify(x)
        x = jnp.dot(
            x,
            all_gather_bf16(self.bottleneck_down),
            out_sharding=P(fsdp),
        )
        x = super().__call__(x, t, cond)
        x = jnp.dot(
            x,
            all_gather_bf16(self.bottleneck_up),
            out_sharding=P(fsdp),
        )
        x = self.unpatchify(x)
        x = jnp.tanh(x)
        return x
