from collections.abc import Callable
from dataclasses import asdict
from functools import partial
from operator import add
from pathlib import Path
from shutil import rmtree

import jax
import jax.numpy as jnp
import optax
import wandb
from flax import nnx
from jax.experimental import io_callback
from jaxtyping import Array, Float32, Int32, PRNGKeyArray, PyTree
from omegaconf import OmegaConf
from tqdm.auto import tqdm

from config import Config

from .dataset import imagenet
from .model import JustImageTransformer, typechecked
from .serialization import restore, save


def tree_norm(tree: PyTree[Float32[Array, "..."]]) -> Float32[Array, ""]:
    return jnp.sqrt(
        jax.tree.reduce_associative(add, jax.tree.map(lambda g: jnp.vdot(g, g), tree)),
    )


def param_name(path: jax.tree_util.KeyPath) -> str:
    return jax.tree_util.keystr(path, simple=True, separator=".").removesuffix(".value")


def tqdm_callback(progbar: tqdm, **kwargs: Float32[Array, ""]) -> None:
    kwargs.pop("global_step")  # don't log the step.
    message = ", ".join(
        [
            f"{k}={v:.2e}"
            for k, v in kwargs.items()
            if k == "grad_norm/total"
            or not k.startswith("grad_norm/")  # don't log per-parameter grad norms.
        ],
    )
    progbar.set_postfix_str(message, refresh=False)
    progbar.update()


def wandb_callback(**kwargs: Float32[Array, ""]) -> None:
    step = kwargs.pop("global_step").item()
    wandb.log({k: v.item() for k, v in kwargs.items()}, step=step)


@nnx.jit(static_argnums=0)
def init(
    config: Config,
    *,
    rngs: nnx.Rngs,
) -> tuple[JustImageTransformer, nnx.Optimizer]:
    model = JustImageTransformer(
        patch_size=config.model.patch_size,
        data_dim=config.model.data_dim,
        bottleneck_dim=config.model.bottleneck_dim,
        num_classes=config.model.num_classes,
        num_layers=config.model.num_layers,
        dim=config.model.dim,
        hidden_dim=config.model.hidden_dim,
        num_heads=config.model.num_heads,
        qk_norm=config.model.qk_norm,
        num_temb_freqs=config.model.num_temb_freqs,
        rngs=rngs,
    )
    tx = optax.chain(
        optax.clip_by_global_norm(config.optimizer.max_grad_norm),
        optax.adamw(
            optax.warmup_constant_schedule(
                init_value=(1e-4 * config.optimizer.lr),
                peak_value=config.optimizer.lr,
                warmup_steps=config.optimizer.warmup_steps,
            ),
            b1=config.optimizer.b1,
            b2=config.optimizer.b2,
            eps=config.optimizer.eps,
            weight_decay=config.optimizer.weight_decay,
        ),
    )
    optimizer = nnx.Optimizer(model, tx, wrt=nnx.Param)
    return model, optimizer


def logit_normal(
    key: PRNGKeyArray, shape: tuple[int, ...], mu: float = -0.8, sigma: float = 0.8
) -> Float32[Array, "..."]:
    noise = jax.random.normal(key, shape=shape)
    s = mu + sigma * noise
    t = jax.nn.sigmoid(s)
    return t


@nnx.jit(static_argnames=("callbacks",))
@typechecked
def train_step(
    model: JustImageTransformer,
    optimizer: nnx.Optimizer,
    batch: tuple[Float32[Array, "B *spatial C"], Int32[Array, " B"]],
    key: PRNGKeyArray,
    callbacks: tuple[Callable, ...],
) -> None:
    x, label = batch
    batch_size = x.shape[0]
    e_key, t_key, c_key = jax.random.split(key, 3)
    noise_scale = x.shape[1] / 256
    epsilon = noise_scale * jax.random.normal(e_key, x.shape)
    t = logit_normal(t_key, (batch_size,))
    alpha = t.reshape(-1, *((x.ndim - 1) * [1]))
    sigma = 1 - alpha
    x_t = alpha * x + sigma * epsilon
    weight = jnp.maximum(sigma, 0.05) ** -2.0
    should_drop = jax.random.uniform(c_key, (batch_size,)) < 0.1
    label = jnp.where(should_drop, model.num_classes, label)

    @nnx.value_and_grad
    def loss_fn(_model: JustImageTransformer) -> Float32[Array, ""]:
        xhat = _model(x_t.astype(jnp.bfloat16), t, label).astype(jnp.float32)
        loss = jnp.mean(weight * jnp.square(xhat - x))
        return loss

    loss, grads = loss_fn(model)
    optimizer.update(model, grads)
    logs = {
        "loss/train": loss,
        "grad_norm/total": tree_norm(grads),
        **{
            f"grad_norm/{param_name(path)}": jnp.linalg.norm(grad)
            for path, grad in jax.tree.leaves_with_path(grads)
        },
    }
    for callback in callbacks:
        io_callback(
            callback,
            None,
            global_step=optimizer.step.value,
            **logs,  # ty: ignore[invalid-argument-type]
        )


def train(config: Config, notes: str | None = None) -> None:
    devices = jax.devices()
    num_devices = len(devices)
    fsdp_size = min(num_devices, 8)
    data_size = num_devices // fsdp_size
    mesh = jax.make_mesh((data_size, fsdp_size), ("data", "hsdp"))
    jax.set_mesh(mesh)
    print(f"Created mesh: {mesh}")

    model, optimizer = init(config, rngs=nnx.Rngs(config.model.seed))
    print(model)

    if config.training.resume_from is not None:
        ckpt = Path(config.training.resume_from)
        restore(ckpt / "model.npz", model)
        restore(ckpt / "optimizer.npz", optimizer)

    run = wandb.init(  # noqa: F841
        project=config.training.wandb_project,
        config=asdict(config),
        notes=notes,
    )

    def checkpoint() -> None:
        ckpt = Path(
            config.training.checkpoint_directory.format(
                project=config.training.wandb_project,
                run=run.name or run.id,
                step=optimizer.step.value.item(),
            )
        )
        print(f"saving new checkpoint {ckpt}")
        save(model, ckpt / "model.npz")
        save(optimizer, ckpt / "optimizer.npz")
        all_ckpts = sorted(
            [d for d in ckpt.parent.iterdir() if d.is_dir() and d.name.isdigit()]
        )
        if config.training.max_checkpoints_to_keep is not None:
            for old_ckpt in all_ckpts[: -config.training.max_checkpoints_to_keep]:
                print(f"removing old checkpoint {old_ckpt}")
                rmtree(old_ckpt)

    callbacks = (
        partial(tqdm_callback, tqdm()),
        wandb_callback,
    )
    key = jax.random.PRNGKey(config.training.seed)
    for step, batch in enumerate(imagenet(**asdict(config.dataloader))):
        if step % config.training.save_interval == 0:
            checkpoint()
        train_step(
            model,
            optimizer,
            batch,
            jax.random.fold_in(key, optimizer.step.value),
            callbacks,
        )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=str, help="path to the config file")
    parser.add_argument("--notes", type=str, default=None, help="wandb notes")
    args = parser.parse_args()
    config = Config.from_yaml(args.config)
    print(f"{args.config}:\n{'-' * 60}\n{OmegaConf.to_yaml(config)}\n{'-' * 60}")
    train(config, args.notes)
