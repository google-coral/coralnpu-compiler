#!/usr/bin/env -S uv run --script
#
# /// script
# requires-python = ">=3.12,<3.14"
# dependencies = ["jax==0.8.0", "gemma", "tensorboard", "typeguard==4.4.1", "kauldron==1.3.0"]
# ///

import argparse
from pathlib import Path

import jax
import jax.numpy as jnp

import sys

print(sys.executable)
print(sys.version)

print(f"jax version={jax.__version__}")
jax.config.update("jax_platforms", "coralnpu_plugin")

result = jax.jit(lambda x, y: x + y)(
    jnp.array([1, 2, 3, 4, 5, 6, 7, 8], dtype=jnp.int32),
    jnp.array([10, 20, 30, 40, 50, 60, 70, 80], dtype=jnp.int32),
)

print("output of adding two arrays =", result)
