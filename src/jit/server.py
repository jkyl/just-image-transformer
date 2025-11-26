import base64
import io
import json
import queue
import threading
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from typing import Generator

from flask import Flask, Response, render_template, request, stream_with_context
from flax import nnx

from config import Config

from .inference import generate, postprocess_image, prepare_batch, schedules
from .model import JustImageTransformer
from .serialization import load
from .train import init

app = Flask(__name__)


# Supports arbitrary batch size, but the layout may be awkward (always two columns wide).
BATCH_SIZE = 4


@dataclass(kw_only=True)
class GenerationArgs:
    seed: int
    label: int
    num_steps: int
    cfg_strength: float
    schedule: str

    def __post_init__(self):
        for field_name, field_type in self.__annotations__.items():
            current_value = getattr(self, field_name)
            setattr(self, field_name, field_type(current_value))


# Share the same defaults between warmup and the frontend to not trigger recompile
# (some arguments are compile-time static, like num_steps and schedule).
GENERATION_DEFAULTS = GenerationArgs(
    seed=555,
    label=169,
    num_steps=50,
    cfg_strength=2.5,
    schedule="linear",
)


@dataclass(kw_only=True)
class State:
    model: JustImageTransformer | None = None
    config: Config | None = None
    labels: list[tuple[int, str]] | None = None

    @property
    def initialized(self) -> bool:
        return (
            self.model is not None
            and self.config is not None
            and self.labels is not None
        )


# Initialize empty global state, to be filled in on startup.
STATE = State()


def download_imagenet_labels() -> list[tuple[int, str]]:
    url = "https://storage.googleapis.com/download.tensorflow.org/data/imagenet_class_index.json"
    with urllib.request.urlopen(url) as response:
        data = json.loads(response.read().decode())
    labels = [(int(k), v[1]) for k, v in data.items()]
    labels.sort(key=lambda x: x[0])
    return labels


def initialize_state(config_path: str, checkpoint_path: str):
    assert not STATE.initialized
    STATE.config = Config.from_yaml(config_path)
    STATE.model: JustImageTransformer = load(
        checkpoint_path, lambda: init(STATE.config, rngs=nnx.Rngs(0))[0]
    )
    STATE.labels = download_imagenet_labels()
    assert STATE.initialized


def warmup():
    assert STATE.initialized
    epsilon, labels = prepare_batch(
        seed=GENERATION_DEFAULTS.seed,
        label=GENERATION_DEFAULTS.label,
        batch_size=BATCH_SIZE,
        image_size=STATE.config.dataloader.image_size,  # ty: ignore[possibly-missing-attribute]
    )
    generate(
        STATE.model,
        epsilon,
        labels,
        cfg_strength=GENERATION_DEFAULTS.cfg_strength,
        num_steps=GENERATION_DEFAULTS.num_steps,
        schedule=schedules[GENERATION_DEFAULTS.schedule],
        callback=CALLBACK,
    ).block_until_ready()


class CallbackRouter:
    def __init__(self):
        self._queues = {}
        self._lock = threading.Lock()

    @contextmanager
    def register(self, q: queue.Queue):
        tid = threading.get_ident()
        with self._lock:
            self._queues[tid] = q
        try:
            yield self
        finally:
            with self._lock:
                self._queues.pop(tid, None)

    def __call__(self):
        tid = threading.get_ident()
        with self._lock:
            q = self._queues.get(tid)
        if q:
            try:
                q.put_nowait({"type": "progress"})
            except queue.Full:
                pass

    def __hash__(self) -> int:
        return hash(self.__class__.__name__)

    def __eq__(self, other) -> bool:
        return isinstance(other, CallbackRouter)


# Singleton callback so that we don't trigger recompile on each request.
CALLBACK = CallbackRouter()


# Maintain the thread pool across calls. Consider increasing max_workers if GPU allows;
# individual jobs should still be distinguishable via their thread ID which maps to their
# progress queues.
EXECUTOR = ThreadPoolExecutor(max_workers=4)


@app.route("/generate", methods=["POST"])
def generate_images():
    assert STATE.initialized
    if not request.json:
        return {"error": "No arguments received"}
    try:
        args = GenerationArgs(**request.json)
    except Exception as e:
        return {"error": f"Error parsing arguments: {e}"}, 400

    # Sample epsilon and construct labels batch.
    epsilon, labels = prepare_batch(
        seed=args.seed,
        label=args.label,
        batch_size=BATCH_SIZE,
        image_size=STATE.config.dataloader.image_size,  # ty: ignore[possibly-missing-attribute]
    )

    # Queue for this request that contains progress, images, and errors.
    results_queue = queue.Queue()

    def worker():
        with CALLBACK.register(results_queue):
            try:
                result = generate(
                    STATE.model,
                    epsilon,
                    labels,
                    args.cfg_strength,
                    args.num_steps,
                    schedules[args.schedule],
                    callback=CALLBACK,
                )
                images_b64 = []
                for i in range(BATCH_SIZE):
                    pil_img = postprocess_image(result[i])
                    img_io = io.BytesIO()
                    pil_img.save(img_io, "PNG")
                    img_io.seek(0)
                    b64_data = base64.b64encode(img_io.read()).decode("utf-8")
                    images_b64.append(b64_data)

                results_queue.put({"type": "result", "images": images_b64})
            except Exception as e:
                results_queue.put({"error": str(e)})

    # Submit the job to the thread pool.
    EXECUTOR.submit(worker)

    def stream_output() -> Generator[str, None, None]:
        while True:
            item = results_queue.get()
            yield json.dumps(item) + "\n"

            if "images" in item or "error" in item:
                break

    return Response(
        stream_with_context(stream_output()), mimetype="application/x-ndjson"
    )


@app.route("/", methods=["GET"])
def index():
    assert STATE.initialized
    return render_template(
        "index.html",
        labels=STATE.labels,
        defaults=asdict(GENERATION_DEFAULTS),
        schedules=list(schedules.keys()),
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=str, help="Path to config.yaml")
    parser.add_argument("checkpoint", type=str, help="Path to model checkpoint")
    parser.add_argument("--port", type=int, default=5000)
    args = parser.parse_args()
    initialize_state(args.config, args.checkpoint)
    warmup()
    app.run(host="0.0.0.0", port=args.port)
