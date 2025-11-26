export XLA_FLAGS="--xla_gpu_enable_command_buffer="
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.95
export HF_DATASETS_DISABLE_PROGRESS_BARS=1
uv run -m src.jit.train config/jit_L_32.yaml "$@"
