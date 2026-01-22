# Expert Choice MoE FeedForward

## Overview

Implement an expert choice MoE variant of the FeedForward module, matching existing code style.

## Expert Choice MoE Algorithm

Unlike token-choice MoE (where each token picks top-k experts), expert choice MoE has each expert pick its top-k tokens. This naturally load-balances across experts.

**Key difference - softmax axis:**
- Token choice: `softmax(logits, axis=experts)` - each token's scores sum to 1 over experts
- Expert choice: `softmax(logits, axis=tokens)` - each expert's scores sum to 1 over tokens

**Steps:**
1. Compute router logits: `[B, T, E]` where E = num_experts
2. Softmax over tokens (axis=1): each expert has a probability distribution over tokens
3. Each expert selects top-k tokens based on its token probabilities (k = capacity)
4. Selected tokens are processed by their assigned expert(s)
5. Outputs are combined, weighted by router scores
6. Tokens not selected by any expert output zeros (pass through via residual)

## Design Decisions

- **Zero output for unselected tokens**: Yes - residual connection handles it (`x + 0 = x`)
- **Capacity factor**: Controls how many tokens each expert processes (`capacity = capacity_factor * T / E`)
- **Router**: Linear projection to E logits, softmax over **tokens** (not experts)
- **Expert architecture**: Same as FeedForward (SwiGLU)

## Implementation

```python
class MoEFeedForward(nnx.Module):
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        num_experts: int,
        capacity_factor: float = 1.0,
        *,
        rngs: nnx.Rngs,
    ):
        self.num_experts = num_experts
        self.capacity_factor = capacity_factor

        # Router: projects to num_experts logits
        self.W_router = nnx.Param(
            glorot_normal()(rngs(), (dim, num_experts), out_sharding=P("fsdp"))
        )

        # Expert weights: [E, D, 2, M] for up-projection (SwiGLU)
        # E and 2 are batch dims (experts, h/gate), D is in, M is out
        self.W_up = nnx.Param(
            glorot_normal(in_axis=1, out_axis=3)(
                rngs(), (num_experts, dim, 2, hidden_dim), out_sharding=P(None, "fsdp")
            )
        )

        # Expert weights: [E, M, D] for down-projection
        # E is batch dim (experts), M is in, D is out
        self.W_down = nnx.Param(
            glorot_normal(in_axis=1, out_axis=2)(
                rngs(), (num_experts, hidden_dim, dim), out_sharding=P(None, "fsdp")
            )
        )

    @typechecked
    def __call__(self, x: BFloat16[Array, "B T D"]) -> BFloat16[Array, "B T D"]:
        B, T, D = x.shape
        E = self.num_experts
        capacity = int(self.capacity_factor * T / E)

        # Router logits: [B, T, E]
        router_logits = jnp.dot(x, all_gather_bf16(self.W_router))

        # Expert choice: softmax over TOKENS (axis=1), not experts
        # Each expert has a probability distribution over tokens
        router_logits_transposed = router_logits.transpose(0, 2, 1)  # [B, E, T]
        expert_scores = jax.nn.softmax(router_logits_transposed, axis=-1)  # [B, E, T]

        # Get top-k token indices per expert
        top_scores, top_indices = jax.lax.top_k(expert_scores, capacity)  # [B, E, K], [B, E, K]

        # Gather selected tokens for each expert: [B, E, K, D]
        # Use advanced indexing
        batch_idx = jnp.arange(B)[:, None, None]
        selected_tokens = x[batch_idx, top_indices]  # [B, E, K, D]

        # Process through experts (batched einsum)
        W_up = all_gather_bf16(self.W_up)      # [E, D, 2, M]
        W_down = all_gather_bf16(self.W_down)  # [E, M, D]

        # Up-project: [B, E, K, D] @ [E, D, 2, M] -> [2, B, E, K, M]
        h, gate = jnp.einsum("BEKD, ED2M -> 2BEKM", selected_tokens, W_up)
        h = gate * nnx.silu(h)  # [B, E, K, M]

        # Down-project: [B, E, K, M] @ [E, M, D] -> [B, E, K, D]
        expert_out = jnp.einsum("BEKM, EMD -> BEKD", h, W_down)

        # Weight by router scores
        expert_out = expert_out * top_scores[..., None]  # [B, E, K, D]

        # Scatter back to output (accumulate where tokens selected by multiple experts)
        output = jnp.zeros_like(x)
        output = output.at[batch_idx, top_indices].add(expert_out)

        return output
```

## Sharding Considerations

Each expert has the same FSDP sharding as the non-MoE FeedForward:

| Weight | Non-MoE Shape | Non-MoE Sharding | MoE Shape | MoE Sharding |
|--------|---------------|------------------|-----------|--------------|
| W_up | `(D, 2, M)` | `P("fsdp")` | `(E, D, 2, M)` | `P(None, "fsdp")` |
| W_down | `(M, D)` | `P("fsdp")` | `(E, M, D)` | `P(None, "fsdp")` |
| W_router | - | - | `(D, E)` | `P("fsdp")` |

The `P(None, "fsdp")` pattern means: replicate the expert axis (E), shard the model dim (D) - exactly matching the non-MoE sharding per expert.

Operations need explicit `out_sharding` annotations for the intermediate tensors.

## Files to Modify

| File | Change |
|------|--------|
| `src/jit/model.py` | Add `MoEFeedForward` class |

## Verification

1. Run `make` to verify types and linting
2. Test that output shape matches input shape
3. Verify tokens not selected by any expert get zero output
4. Compare parameter count: `E * (D*2*M + M*D) + D*E` vs dense `D*2*M + M*D`
