export XLA_FLAGS="--xla_gpu_enable_command_buffer="
export XLA_PYTHON_CLIENT_PREALLOCATE=false
uv run -m src.jit.server \
    config/jit_L_32.yaml \
    "$@"
