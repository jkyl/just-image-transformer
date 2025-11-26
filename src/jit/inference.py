from collections.abc import Callable
from dataclasses import dataclass
from functools import partial
from typing import Literal

import jax
import jax.numpy as jnp
import numpy as np
import PIL.Image
from flax import nnx
from jax.experimental import io_callback
from jaxtyping import Array, Float32, Int32
from tqdm.auto import tqdm

from config import Config

from .model import JustImageTransformer, typechecked
from .serialization import load
from .train import init


@dataclass(frozen=True)
class InferenceArgs:
    config: Config
    checkpoint: str
    output: str
    seed: int
    label: int
    cfg_strength: float
    num_steps: int
    schedule: Literal["linear", "logit_normal"]


@typechecked
def cfg(
    x_t: Float32[Array, "B ..."],
    t: Float32[Array, " B"],
    model: JustImageTransformer,
    label: Int32[Array, " B"],
    strength: Float32[Array, ""] | float,
) -> Float32[Array, "B ..."]:
    assert model.num_classes is not None
    label = jnp.concatenate([label, jnp.full_like(label, model.num_classes)])
    x_t = jnp.concatenate([x_t, x_t])
    t = jnp.concatenate([t, t])
    xhat = model(x_t.astype(jnp.bfloat16), t, label).astype(jnp.float32)
    xhat_cond, xhat_uncond = jnp.split(xhat, 2)
    xhat = xhat_uncond + strength * (xhat_cond - xhat_uncond)
    return xhat


@typechecked
def euler(
    model_fn: Callable[
        [Float32[Array, "B ..."], Float32[Array, "B ..."]], Float32[Array, "B ..."]
    ],
    epsilon: Float32[Array, "B ..."],
    x: Float32[Array, "B ..."],
    t: Float32[Array, " B"],
) -> tuple[Float32[Array, "B ..."], Float32[Array, "B ..."]]:
    alpha = t.reshape(-1, *((x.ndim - 1) * [1]))
    sigma = 1 - alpha
    x_t = alpha * x + sigma * epsilon
    xhat = model_fn(x_t, t)
    ehat = (x_t - alpha * xhat) / sigma
    return ehat, xhat


@typechecked
def linear_schedule(num_steps: int) -> Float32[Array, " num_steps"]:
    return jnp.linspace(0, 1, num_steps, endpoint=False)


@typechecked
def logit_normal_schedule(
    num_steps: int, mu: float = -0.8, sigma: float = 0.8
) -> Float32[Array, " num_steps"]:
    quantiles = jnp.linspace(0, 1, num_steps, endpoint=False)
    s = mu + sigma * jax.scipy.special.ndtri(quantiles)
    t = jax.nn.sigmoid(s)
    return t


# Container so that we can access schedules by name.
schedules: dict[str, Callable[[int], Float32[Array, " num_steps"]]] = {
    "linear": linear_schedule,
    "logit_normal": logit_normal_schedule,
}


@nnx.jit(static_argnames=("num_steps", "schedule", "callback"))
@typechecked
def generate(
    model: JustImageTransformer,
    epsilon: Float32[Array, "B ..."],
    label: Int32[Array, " B"],
    cfg_strength: Float32[Array, ""] | float,
    num_steps: int,
    schedule: Callable[[int], Float32[Array, " num_steps"]],
    callback: Callable | None = None,
) -> Float32[Array, "B ..."]:
    batch_size = epsilon.shape[0]
    timesteps = schedule(num_steps)
    timesteps = jnp.tile(timesteps[:, None], (1, batch_size))

    @nnx.scan(in_axes=(nnx.Carry, 0), out_axes=nnx.Carry)
    def forward(carry, t):
        model, (epsilon, x) = carry
        ehat, xhat = euler(
            partial(cfg, model=model, label=label, strength=cfg_strength),
            epsilon,
            x,
            t,
        )
        if callback is not None:
            io_callback(callback, None)
        return model, (ehat, xhat)

    _, (_, xhat) = forward((model, (epsilon, jnp.zeros_like(epsilon))), timesteps)
    return xhat


def prepare_batch(
    seed: int,
    label: int,
    batch_size: int,
    image_size: int,
    noise_scale_base_resolution: int = 256,
) -> tuple[
    Float32[Array, "batch_size image_size image_size 3"], Int32[Array, " batch_size"]
]:
    key = jax.random.PRNGKey(seed)
    shape = (batch_size, image_size, image_size, 3)
    noise_scale = image_size / noise_scale_base_resolution
    epsilon = noise_scale * jax.random.normal(key, shape)
    labels = jnp.full((batch_size,), label)
    return epsilon, labels


def postprocess_image(result: Float32[Array, "height width 3"]) -> PIL.Image.Image:
    img = np.asarray(result)
    img = np.clip(img * 128 + 128, 0, 255).astype(np.uint8)
    img = PIL.Image.fromarray(img)
    return img


def inference(args: InferenceArgs) -> None:
    model = load(args.checkpoint, lambda: init(args.config, rngs=nnx.Rngs(0))[0])
    epsilon, labels = prepare_batch(
        seed=args.seed,
        label=args.label,
        batch_size=1,
        image_size=config.dataloader.image_size,
    )
    with tqdm(total=args.num_steps, desc="Generating") as progbar:
        result = generate(
            model,
            epsilon,
            labels,
            args.cfg_strength,
            args.num_steps,
            schedules[args.schedule],
            progbar.update,
        )
    img = postprocess_image(result[0])
    img.save(args.output)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=str)
    parser.add_argument("checkpoint", type=str)
    parser.add_argument("output", type=str)
    parser.add_argument("--seed", type=int, default=555)
    parser.add_argument("--label", type=int, default=123)
    parser.add_argument("--cfg_strength", type=float, default=3.0)
    parser.add_argument("--num_steps", type=int, default=50)
    parser.add_argument(
        "--schedule", choices=("linear", "logit_normal"), default="linear"
    )
    cli_args = vars(parser.parse_args())
    config = Config.from_yaml(cli_args.pop("config"))
    args = InferenceArgs(config, **cli_args)
    inference(args)
