import os

# Simulate 8 devices on CPU. Must run before jax is imported.
os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=8")
os.environ.setdefault("JAX_PLATFORMS", "cpu")
