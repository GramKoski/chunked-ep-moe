"""Small shard_map helpers."""

import jax
import jax.numpy as jnp


def _axes(axis_name):
  return tuple(axis_name) if isinstance(axis_name, (tuple, list)) else (axis_name,)


def varying(x, axis_name):
  """Marks x as varying over axis_name so it can be a loop carry next to per-device values."""
  have = jax.typeof(x).manual_axis_type.varying
  missing = tuple(a for a in _axes(axis_name) if a not in have)
  return jax.lax.pcast(x, missing, to="varying") if missing else x


def zeros(shape, dtype, axis_name):
  return varying(jnp.zeros(shape, dtype), axis_name)
