"""Data loading and pre-processing utilities."""
import os
from typing import Tuple, Callable, Sequence, Text, Dict, Union

import jax.numpy as np
from jax import random

from utils import utils

import pandas as pd
import pickle

#import pdb


DataSynth = Tuple[Dict[Text, Union[np.ndarray, float, None]],
                  np.ndarray, np.ndarray]
DataReal = Dict[Text, Union[np.ndarray, float, None]]
ArrayTup = Tuple[np.ndarray, np.ndarray]

Equations = Dict[Text, Callable[..., np.ndarray]]


# =============================================================================
# NOISE SOURCES
# =============================================================================
def std_normal_1d(key: np.ndarray, num: int) -> np.ndarray:
  """Generate a Gaussian for the confounder."""
  return random.normal(key, (num,))


def std_normal_2d(key: np.ndarray, num: int) -> ArrayTup:
  """Generate a multivariate Gaussian for the noises e_X, e_Y."""
  key1, key2 = random.split(key)
  return random.normal(key1, (num,)), random.normal(key2, (num,))

def normal_2d(k):
  """Generate a multivariate Gaussian for the noises e_X, e_Y with mean k"""
  def std_normal_2d(key: np.ndarray, num: int):
    key1, key2 = random.split(key)
    return k + random.normal(key1, (num,)), k + random.normal(key2, (num,))
  return std_normal_2d

def normal_3d(k):
  """Generate a multivariate Gaussian for the noises e_X, e_Y with mean k"""
  def std_normal_3d(key: np.ndarray, num: int) -> ArrayTup:
    key1, key2 = random.split(key)
    key2, key3 = random.split(key2)
    return k+random.normal(key1, (num,)), k+random.normal(key2, (num,)), k+random.normal(key3, (num,))
  return std_normal_3d

def std_normal_3d(key: np.ndarray, num: int) -> ArrayTup:
  """Generate a multivariate normal."""
  key1, key2 = random.split(key)
  key2, key3 = random.split(key2)
  return random.normal(key1, (num,)), random.normal(key2, (num,)), random.normal(key3, (num,))


# =============================================================================
# SYNTHETIC STRUCTURAL EQUATIONS
# =============================================================================


structural_equations = {
  "lin1": {
    "noise": std_normal_2d,
    "confounder": std_normal_1d,
    "f_z": std_normal_1d,
    "f_x": lambda z, c, ex: 0.5 * z + 3 * c + ex,
    "f_y": lambda x, c, ey: x - 6 * c + ey,
  },
  "lin2": {
    "noise": std_normal_2d,
    "confounder": std_normal_1d,
    "f_z": std_normal_1d,
    "f_x": lambda z, c, ex: 3.0 * z + 0.5 * c + ex,
    "f_y": lambda x, c, ey: x - 6 * c + ey,
  },
  "quad1": {
    "noise": std_normal_2d,
    "confounder": std_normal_1d,
    "f_z": std_normal_1d,
    "f_x": lambda z, c, ex: 0.5 * z + 3 * c + ex,
    "f_y": lambda x, c, ey: 0.3 * x ** 2 - 1.5 * x * c + ey,
  },
  "quad2": {
    "noise": std_normal_2d,
    "confounder": std_normal_1d,
    "f_z": std_normal_1d,
    "f_x": lambda z, c, ex: 3.0 * z + 0.5 * c + ex,
    "f_y": lambda x, c, ey: 0.3 * x ** 2 - 1.5 * x * c + ey,
  },
  "lin1-2d": {
    "ex": std_normal_2d,
    "ey": std_normal_1d,
    "confounder": std_normal_2d,
    "f_z": std_normal_2d,
    "f_x": lambda z, c, ex:  0.5 * z + 2 * c + ex, 
    "f_y": lambda x1, x2, c1, c2, ey: x1 +  x2 - 3 * (x1 + x2) * (c1 + c2) + ey,
  },
   "lin2-2d": {
    "ex": std_normal_2d,
    "ey": std_normal_1d,
    "confounder": std_normal_2d,
    "f_z": std_normal_2d,
    "f_x": lambda z, c, ex: 2 * z + 1 * c + ex, 
    "f_y": lambda x1, x2, c1, c2, ey: 5*x1 + 6*x2 - x1 * (c1 + c2) + ey,
  },
  "lin3-2d": {
    "ex": std_normal_2d,
    "ey": std_normal_1d,
    "confounder": std_normal_2d,
    "f_z": std_normal_2d,
    "f_x": lambda z, c, ex:  z + 2 * c + ex, 
    "f_y": lambda x1, x2, c1, c2, ey: 2*x1 + x2 - 1 * (c1 + c2) + ey,
  },
  "quad1-2d": {
    "ex": std_normal_2d,
    "ey": std_normal_1d,
    "confounder": std_normal_2d,
    "f_z": std_normal_2d,
    "f_x": lambda z, c, ex:  z + 2 * c + ex, 
    "f_y": lambda x1, x2, c1, c2, ey: 2*x1**2 + 2*x2**2 - 1 * (c1 + c2) + ey,
  },
  "quad2-2d": {
    "ex": std_normal_2d,
    "ey": std_normal_1d,
    "confounder": std_normal_2d,
    "f_z": std_normal_2d,
    "f_x": lambda z, c, ex: 2 * z + 1 * c + ex, 
    "f_y": lambda x1, x2, c1, c2, ey: 5*x1**2 + 5*x2**2 - (x1 + x2) * (c1 + c2) + ey,
  },
  "lin1-3d": {
    "ex": std_normal_3d,
    "ey": std_normal_1d,
    "confounder": std_normal_3d,
    "f_z": std_normal_3d,
    "f_x": lambda z, c, ex: z + 2 * c + ex, 
    "f_y": lambda x1, x2, x3, c1, c2, c3, ey: x1 + x2 + 2 * x3 + 2 * (c1 + c2) + ey,
  },
  "lin2-3d": {
    "ex": std_normal_3d,
    "ey": std_normal_1d,
    "confounder": std_normal_3d,
    "f_z": std_normal_3d,
    "f_x": lambda z, c, ex: 2 * z + 1 * c + ex, 
    "f_y": lambda x1, x2, x3, c1, c2, c3, ey: x1 + x2 + 2 * x3 - 0.5 * (x1 + x2) * (c1 + c2) + ey,
  },
  "quad1-3d": {
    "ex": std_normal_3d,
    "ey": std_normal_1d,
    "confounder": std_normal_3d,
    "f_z": std_normal_3d,
    "f_x": lambda z, c, ex:  z + 2 * c + ex, 
    "f_y": lambda x1, x2, x3, c1, c2, c3, ey:  x1**2 + x2**2 + 2 * x3 - 2 * (c1 + c2 + c3) + ey,
  },
  "quad2-3d": {
    "ex": std_normal_3d,
    "ey": std_normal_1d,
    "confounder": std_normal_3d,
    "f_z": std_normal_3d,
    "f_x": lambda z, c, ex: 2 * z + 1 * c + ex, 
    "f_y": lambda x1, x2, x3, c1, c2, c3, ey: 2 * x1**2 + 2 * x2**2 + 2 * x3 \
                  - 0.3 * (x2 + x3) * (c1 + c2 + c3) + ey,
  },
  "test-3d": {
    "ex": std_normal_3d,
    "ey": std_normal_1d,
    "confounder": std_normal_3d,
    "f_z": std_normal_3d,
    "f_x": lambda z, c, ex:  z - 3 * c + ex, 
    "f_y": lambda x1, x2, x3, c1, c2, c3, ey: 3*x1 +  4*x2 - x3 - 2*(x1 + x2 + 2*x3) * (c1 + c2 + c3) + ey,
  },
}

structural_equations_fd = {
  "fd-lin2-2d": {
    "ex": std_normal_2d,
    "em": std_normal_2d,
    "ey": std_normal_1d,
    "confounder": std_normal_2d,
    "u_xy": std_normal_2d,
    "f_x": lambda u_xy, ex: u_xy + ex,
    "f_m": lambda x, c, em: 3 * x - em + c, 
    "f_y": lambda m1, m2, c1, c2, u1, u2, ey: 2 * m1 + m2 - 0.3 * (m1 + m2) * (c1 + c2 + u1 + u2)  + ey,
  },
  "fd-lin1-2d": {
    "ex": std_normal_2d,
    "em": std_normal_2d,
    "ey": std_normal_1d,
    "confounder": std_normal_2d,
    "u_xy": std_normal_2d,
    "f_x": lambda u_xy, ex: u_xy + ex,
    "f_m": lambda x, c, em: x - em + 3*c, 
    "f_y": lambda m1, m2, c1, c2, u1, u2, ey: 2 * m1 + m2 -  (m1 + m2) * (c1 + c2 + u1 + u2) + ey,
  },
   "fd-test-2d": {
    "ex": std_normal_2d,
    "em": std_normal_2d,
    "ey": std_normal_1d,
    "confounder": std_normal_2d,
    "u_xy": std_normal_2d,
    "f_x": lambda u_xy, ex: 5*u_xy + ex,
    "f_m": lambda x, c, em: x - em + 3*c, 
    "f_y": lambda m1, m2, c1, c2, u1, u2, ey: 2 * m1 + m2 -  (m1 + m2) * (c1 + c2 + u1 + u2) + ey,
  }
}


# =============================================================================
# DATA GENERATORS
# =============================================================================

def whiten(
  inputs: Dict[Text, np.ndarray]
) -> Dict[Text, Union[float, np.ndarray, None]]:
  """Whiten each input in the input dict."""
  res = {}
  for k, v in inputs.items():
    if v is not None:
      mu = np.mean(v, 0)
      std = np.maximum(np.std(v, 0), 1e-7)
      res[k + "_mu"] = mu
      res[k + "_std"] = std
      res[k] = (v - mu) / std
    else:
      res[k] = v
  return res


def whiten_with_mu_std(val: np.ndarray, mu: float, std: float) -> np.ndarray:
  return (val - mu) / std

def unwhiten_with_mu_std(val: np.ndarray, mu: float, std: float) -> np.ndarray:
  return (val*std + mu)

def get_nonconstant_axis(d):
  for ax in range(d.shape[1]):
    if(d[0,ax] != d[1, ax]): return d[:,ax]


def get_data(graph, dataset, equation, num_data, num_xstar, xstar_axis, seed_data, \
             path='../data', filename='lin2', save_data=False):
  """Get data for a given graph and dataset."""
  key_data = random.PRNGKey(int(seed_data))
  if(graph=='IV'):
    if dataset == "scalar":
      key_data, subkey_data = random.split(key_data)
      dat, data_xstar, data_ystar = get_synth_data(
      subkey_data, num_data,equation,
      disconnect_instrument=False)
      xstar_plotting = data_xstar
      xmin, xmax = np.min(dat['x']), np.max(dat['x'])
      xstar_grid = np.linspace(xmin, xmax, num_xstar + 1)
      xstar_grid = (xstar_grid[:-1] + xstar_grid[1:]) / 2
      xstar_grid_plotting = xstar_grid
      data_xstar = data_xstar[:,np.newaxis]
    elif dataset == "synth-2d":
      key_data, subkey_data = random.split(key_data)
      dat, data_xstar, data_ystar, xstar_grid,\
      xstar_grid_plotting, xstar_plotting = get_synth_data_2d(
      subkey_data, num_data, equation, xstar_axis, num_xstar)
    elif dataset == "synth-3d":
      key_data, subkey_data = random.split(key_data)
      dat, data_xstar, data_ystar, xstar_grid,\
      xstar_grid_plotting, xstar_plotting = get_synth_data_3d(
      subkey_data, num_data, equation, xstar_axis, num_xstar)
    elif dataset == "yeast":
      key_data, subkey_data = random.split(key_data)
      dat, data_xstar, data_ystar, xstar_grid,\
      xstar_grid_plotting, xstar_plotting = get_yeast_data(subkey_data)
    else:
        raise ValueError(f"Unknown dataset {dataset}")

    dat['x'], dat['y'], dat['z']= utils.jax_to_torch(dat['x']), utils.jax_to_torch(dat['y']), utils.jax_to_torch(dat['z'])
    dat['xstar_grid'] = xstar_grid
    if(save_data):
      with open(os.path.join(path, filename), 'wb') as handle:
        pickle.dump(dat, handle, protocol=pickle.HIGHEST_PROTOCOL)
    return dat
  elif(graph=='FD'):
    if dataset == "synth-2d":
      key_data, subkey_data = random.split(key_data)
      dat, data_xstar, data_ystar, xstar_grid,\
      xstar_grid_plotting, xstar_plotting = fd_get_synth_data_2d(
      subkey_data, num_data, equation,
      xstar_axis,num_xstar)
    else:
        raise ValueError(f"Unknown dataset {dataset}")
    dat['x'], dat['y'], dat['m']= utils.jax_to_torch(dat['x']), utils.jax_to_torch(dat['y']), utils.jax_to_torch(dat['m'])
    dat = utils.standardize_data_shapes(dat)


def get_synth_data(
  key: np.ndarray,
  num: int,
  equations: Text,
  num_xstar: int = 100,
  external_equations: Equations = None,
  disconnect_instrument: bool = False
) -> DataSynth:
  """Generate some scalar synthetic data.

    Args:
      key: A JAX random key.
      num: The number of examples to generate.
      equations: Which structural equations to choose for x and y. Default: 1
      num_xstar: Size of grid for interventions on x.
      external_equations: A dictionary that must contain the keys 'f_x' and
        'f_y' mapping to callables as values that take two np.ndarrays as
        arguments and produce another np.ndarray. These are the structural
        equations for X and Y in the graph Z -> X -> Y.
        If this argument is not provided, the `equation` argument selects
        structural equations from the pre-defined dict `structural_equations`.
      disconnect_instrument: Whether to regenerate random (standard Gaussian)
        values for the instrument after the data has been generated. This
        serves for diagnostic purposes, i.e., looking at the same x, y data,

    Returns:
      A 3-tuple (values, xstar, ystar) consisting a dictionary `values`
          containing values for x, y, z, confounder, ex, ey as well as two
          array xstar, ystar containing values for the true cause-effect.
  """
  if external_equations is not None:
    eqs = external_equations
  else:
    eqs = structural_equations[equations]

  key, subkey = random.split(key)
  ex, ey = eqs["noise"](subkey, num)
  key, subkey = random.split(key)
  confounder = eqs["confounder"](subkey, num)
  key, subkey = random.split(key)
  z = eqs["f_z"](subkey, num)
  x = eqs["f_x"](z, confounder, ex)
  y = eqs["f_y"](x, confounder, ey)

  values = whiten({'x': x, 'y': y, 'z': z, 'confounder': confounder,
                   'ex': ex, 'ey': ey})

  # Evaluate E[ Y | do(x^*)] empirically
  xmin, xmax = np.min(x), np.max(x)
  xstar = np.linspace(xmin, xmax, num_xstar)
  ystar = []
  for _ in range(500):
    key, subkey = random.split(key)
    tmpey = eqs["noise"](subkey, num_xstar)[1]
    key, subkey = random.split(key)
    tmpconf = eqs["confounder"](subkey, num_xstar)
    tmp_ystar = whiten_with_mu_std(
      eqs["f_y"](xstar, tmpconf, tmpey), values["y_mu"], values["y_std"])
    ystar.append(tmp_ystar)
  ystar = np.array(ystar)
  xstar = whiten_with_mu_std(xstar, values["x_mu"], values["x_std"])
  if disconnect_instrument:
    key, subkey = random.split(key)
    values['z'] = random.normal(subkey, shape=z.shape)
  return values, xstar, ystar


def get_synth_data_2d(
  key: np.ndarray,
  num: int,
  equations,
  axis,
  num_xstar_bound,
  num_xstar: int = 100,
  external_equations = None
):
    """Generate some 2d synthetic data for the IV setting.

    Args:
      key: A JAX random key.
      num: The number of samples to generate.
      equations: Which structural equations to choose for x and y. 
        Provided as a dict with keys 'f_x', 'f_y', 'f_z', 'confounder', 'ex', 'ey'
      axis: Which axis to vary to get x* values to calculate bounds at, keeping the
        others fixed.
      num_xstar_bound: Number of x* values chosen to calculate bounds at.
      num_xstar: Number of x* values chosen to calculate true E[Y|do(x*)] at.
      external_equations: The dictionary to use for the equations.

    Returns:
      values: A dictionary containing values for x, y, z, confounder, ex, ey.
      xstar: The x* values chosen to calculate true E[Y|do(x*)] at.
      ystar: The true E[Y|do(x*)] values corresponding to xstar.
      xstar_grid: The x* values chosen to calculate bounds at.
      xstar_grid_plotting: The values to plot on the x-axis to show the xstar_grid.
      xstar_plotting: The values to plot on the x-axis to show the xstar vs ystar curve.
    """
    if external_equations is not None:
        eqs = external_equations
    else:
        eqs = structural_equations[equations]
    
    # Get x_value for xstar_bound to make sure we have the same each time
    key_bound = random.PRNGKey(42)
    key_bound, subkey_bound = random.split(key_bound)
    _ex = eqs["ex"](key_bound, 2000)
    key_bound, subkey_bound = random.split(key_bound)
    _c = eqs["confounder"](key_bound, 2000)
    key_bound, subkey_bound = random.split(key_bound)
    _z = eqs["f_z"](key_bound, 2000)
    _x_bound = eqs["f_x"](np.array(_z), np.array(_c), np.array(_ex))
    _x_bound = _x_bound.T

    key, subkey = random.split(key)
    ex = eqs["ex"](subkey, num)
    key, subkey = random.split(key)
    ey = eqs["ey"](subkey, num)
    key, subkey = random.split(key)
    c = eqs["confounder"](subkey, num)
    key, subkey = random.split(key)
    z = eqs["f_z"](subkey, num)
    x = eqs["f_x"](np.array(z), np.array(c), np.array(ex))
    x1, x2 = x
    c1, c2 = c
    y = eqs["f_y"](x1, x2, c1, c2, ey)
    x = x.T
    z = np.array(z).T
    c = np.array(c).T
    ex = np.array(ex).T

    values = whiten({'x': x, 'y': y, 'z': z, 'confounder': c,
                   'ex': ex, 'ey': ey, 'x_temp': _x_bound})

  # Evaluate E[ Y | do(x^*)] empirically
    if(axis==0):
      xmin_b, xmax_b = np.min(_x_bound[:,axis]), np.max(_x_bound[:,axis])
      _diff = xmax_b - xmin_b
      xmin_b = xmin_b + _diff/10; xmax_b = xmax_b - _diff/10

      xmin, xmax = np.min(x[:,axis]), np.max(x[:,axis])
      x2_0 = np.mean(_x_bound[:,1])
      x1star = np.linspace(xmin, xmax, num_xstar)
      x2star = np.full_like(x1star, x2_0)
      x1star_bound = np.linspace(xmin_b, xmax_b, num_xstar_bound)
      x2star_bound = np.full_like(x1star_bound, x2_0)
      xstar_grid_plotting = x1star_bound
      xstar_plotting = x1star
    if(axis==1):
      xmin_b, xmax_b = np.min(_x_bound[:,axis]), np.max(_x_bound[:,axis])
      _diff = xmax_b - xmin_b
      xmin_b = xmin_b + _diff/10; xmax_b = xmax_b - _diff/10
      
      xmin, xmax = np.min(x[:,axis]), np.max(x[:,axis])
      x1_0 = np.mean(_x_bound[:,0])
      x2star = np.linspace(xmin, xmax, num_xstar)
      x1star = np.full_like(x2star, x1_0)
      x2star_bound = np.linspace(xmin_b, xmax_b, num_xstar_bound)
      x1star_bound = np.full_like(x2star_bound, x1_0)
      xstar_grid_plotting = x2star_bound
      xstar_plotting = x2star
    ystar = []
    for _ in range(500):
        key, subkey = random.split(key)
        tmpey = eqs["ey"](subkey, num_xstar)
        key, subkey = random.split(key)
        tmpconf = eqs["confounder"](subkey, num_xstar)
        tc1, tc2 = tmpconf
        tmp_ystar = whiten_with_mu_std(
          eqs["f_y"](x1star, x2star, tc1, tc2, tmpey), values["y_mu"], values["y_std"])
        ystar.append(tmp_ystar)
    ystar = np.array(ystar) 
    xstar = np.vstack((x1star, x2star)).T
    xstar_grid = np.vstack((x1star_bound, x2star_bound)).T
    xstar = whiten_with_mu_std(xstar, values["x_temp_mu"], values["x_temp_std"])
    xstar_plotting = get_nonconstant_axis(xstar)
    xstar_grid = whiten_with_mu_std(xstar_grid, values['x_temp_mu'], values['x_temp_std'])
    xstar_grid_plotting = get_nonconstant_axis(xstar_grid)
    for k in list(values.keys()):
      if(k.startswith("x_temp")): del values[k]

    return values, xstar, ystar, xstar_grid, xstar_grid_plotting, xstar_plotting

def get_synth_data_3d(
  key: np.ndarray,
  num: int,
  equations,
  axis,
  num_xstar_bound,
  num_xstar: int = 100,
  external_equations = None,
):
    """Generate some 3d synthetic data for the IV setting.

      key: A JAX random key.
      num: The number of samples to generate.
      equations: Which structural equations to choose for x and y. 
        Provided as a dict with keys 'f_x', 'f_y', 'f_z', 'confounder', 'ex', 'ey'
      axis: Which axis to vary to get x* values to calculate bounds at, keeping the
        others fixed.
      num_xstar_bound: Number of x* values chosen to calculate bounds at.
      num_xstar: Number of x* values chosen to calculate true E[Y|do(x*)] at.
      external_equations: The dictionary to use for the equations.

    Returns:
      values: A dictionary containing values for x, y, z, confounder, ex, ey.
      xstar: The x* values chosen to calculate true E[Y|do(x*)] at.
      ystar: The true E[Y|do(x*)] values corresponding to xstar.
      xstar_grid: The x* values chosen to calculate bounds at.
      xstar_grid_plotting: The values to plot on the x-axis to show the xstar_grid.
      xstar_plotting: The values to plot on the x-axis to show the xstar vs ystar curve.

    """
    if external_equations is not None:
        eqs = external_equations
    else:
        eqs = structural_equations[equations]

    # Get x_value for xstar_bound to make sure we have the same each time
    key_bound = random.PRNGKey(42)
    key_bound, subkey_bound = random.split(key_bound)
    _ex = eqs["ex"](subkey_bound, 2000)
    key_bound, subkey_bound = random.split(key_bound)
    _c = eqs["confounder"](subkey_bound, 2000)
    key_bound, subkey_bound = random.split(key_bound)
    _z = eqs["f_z"](subkey_bound, 2000)
    _x_bound = eqs["f_x"](np.array(_z), np.array(_c), np.array(_ex))
    _x_bound = _x_bound.T

    key, subkey = random.split(key)
    ex = eqs["ex"](subkey, num)
    key, subkey = random.split(key)
    ey = eqs["ey"](subkey, num)
    key, subkey = random.split(key)
    c = eqs["confounder"](subkey, num)
    key, subkey = random.split(key)
    z = eqs["f_z"](subkey, num)
    x = eqs["f_x"](np.array(z), np.array(c), np.array(ex))
    x1, x2, x3 = x
    c1, c2, c3 = c
    y = eqs["f_y"](x1, x2, x3, c1, c2, c3, ey)
    x = x.T
    z = np.array(z).T
    c = np.array(c).T
    ex = np.array(ex).T

    values = whiten({'x': x, 'y': y, 'z': z, 'confounder': c,
                   'ex': ex, 'ey': ey, 'x_temp': _x_bound})

  # Evaluate E[ Y | do(x^*)] empirically
    if(axis==0):
      xmin_b, xmax_b = np.min(_x_bound[:,axis]), np.max(_x_bound[:,axis])
      _diff = xmax_b - xmin_b
      xmin_b = xmin_b + _diff/10; xmax_b = xmax_b - _diff/10

      xmin, xmax = np.min(x[:,axis]), np.max(x[:,axis])
      x2_0 = np.mean(_x_bound[:,1])
      x3_0 = np.mean(_x_bound[:,2])
      x1star = np.linspace(xmin, xmax, num_xstar)
      x2star = np.full_like(x1star, x2_0)
      x3star = np.full_like(x1star, x3_0)
      x1star_bound = np.linspace(xmin_b, xmax_b, num_xstar_bound)
      x2star_bound = np.full_like(x1star_bound, x2_0)
      x3star_bound = np.full_like(x1star_bound, x3_0)
      xstar_grid_plotting = x1star_bound
      xstar_plotting = x1star
    if(axis==1):
      xmin_b, xmax_b = np.min(_x_bound[:,axis]), np.max(_x_bound[:,axis])
      _diff = xmax_b - xmin_b
      xmin_b = xmin_b + _diff/10; xmax_b = xmax_b - _diff/10

      xmin, xmax = np.min(x[:,axis]), np.max(x[:,axis])
      x1_0 = np.mean(_x_bound[:,0])
      x3_0 = np.mean(_x_bound[:,2])
      x2star = np.linspace(xmin, xmax, num_xstar)
      x1star = np.full_like(x2star, x1_0)
      x3star = np.full_like(x2star, x3_0)
      x2star_bound = np.linspace(xmin_b, xmax_b, num_xstar_bound)
      x1star_bound = np.full_like(x2star_bound, x1_0)
      x3star_bound = np.full_like(x2star_bound, x3_0)
      xstar_grid_plotting = x2star_bound
      xstar_plotting = x2star
    if(axis==2):
      xmin_b, xmax_b = np.min(_x_bound[:,axis]), np.max(_x_bound[:,axis])
      _diff = xmax_b - xmin_b
      xmin_b = xmin_b + _diff/10; xmax_b = xmax_b - _diff/10

      xmin, xmax = np.min(x[:,axis]), np.max(x[:,axis])
      x1_0 = np.mean(_x_bound[:,0])
      x2_0 = np.mean(_x_bound[:,1])
      x3star = np.linspace(xmin, xmax, num_xstar)
      x1star = np.full_like(x3star, x1_0)
      x2star = np.full_like(x3star, x2_0)
      x3star_bound = np.linspace(xmin_b, xmax_b, num_xstar_bound)
      x1star_bound = np.full_like(x3star_bound, x1_0)
      x2star_bound = np.full_like(x3star_bound, x2_0)
      xstar_grid_plotting = x3star_bound
      xstar_plotting = x3star
    ystar = []
    for _ in range(500):
        key, subkey = random.split(key)
        tmpey = eqs["ey"](subkey, num_xstar)
        key, subkey = random.split(key)
        tmpconf = eqs["confounder"](subkey, num_xstar)
        tc1, tc2, tc3 = tmpconf
        tmp_ystar = whiten_with_mu_std(
          eqs["f_y"](x1star, x2star, x3star, tc1, tc2, tc3, tmpey), values["y_mu"], values["y_std"])
        ystar.append(tmp_ystar)
    ystar = np.array(ystar) 
    xstar = np.vstack((x1star, x2star, x3star)).T
    xstar = whiten_with_mu_std(xstar, values["x_temp_mu"], values["x_temp_std"])
    xstar_plotting = get_nonconstant_axis(xstar)
    xstar_grid = np.vstack((x1star_bound, x2star_bound, x3star_bound)).T
    xstar_grid = whiten_with_mu_std(xstar_grid, values['x_temp_mu'], values['x_temp_std'])
    xstar_grid_plotting = get_nonconstant_axis(xstar_grid)
    for k in list(values.keys()):
      if(k.startswith("x_temp")): del values[k]

    return values, xstar, ystar, xstar_grid, xstar_grid_plotting, xstar_plotting


def get_yeast_data(key, num_xstar=5):
  """Generate yeast data and return the data and the true function"""
  key_alpha = random.PRNGKey(42)
  alpha = random.uniform(key_alpha, (15,))

  csv_path = '../data/mandelian.csv'
  df_yeast = pd.read_csv(csv_path)

  z = np.array(df_yeast.iloc[:, :5].values)
  x = np.array(df_yeast.iloc[:, 5:].values)
  key, subkey = random.split(key)
  c = random.multivariate_normal(subkey, np.zeros(5), np.eye(5), (112,))
  key, subkey = random.split(key)
  ey = random.multivariate_normal(subkey, np.zeros(1), np.eye(1), (112,))
  y = np.zeros(112)
  for i in range(15):
      if i < 5:
          y = y + alpha[i] * (x[:, i] * c[:, i])
      else:
          y = y + (alpha[i] * x[:, i])
  y = y + ey.squeeze()

  xstar_base = np.array(x[y.argmin()].copy())
  xstar = [i for i in range(num_xstar)]
  ystars = []
  for i in range(num_xstar):
    key, subkey = random.split(key)
    noise = random.multivariate_normal(subkey, np.zeros(1), np.eye(1), (3,)).squeeze()
    idx = np.array([j for j in range(3*i, 3*i + 3)])
    xstar[i] = np.array(xstar_base.copy())
    xstar[i] = xstar[i].at[idx].add(noise/5)
  xstar = np.array(xstar)

  for i in range(500):
    key, subkey = random.split(key)
    c_ = random.multivariate_normal(subkey, np.zeros(5), np.eye(5), (num_xstar,))
    key, subkey = random.split(key)
    ey_ = random.multivariate_normal(subkey, np.zeros(1), np.eye(1), (num_xstar,))   
    y_ = np.zeros(num_xstar)
    for i in range(15):
      if i < 5:
        y_ = y_ + alpha[i] * (xstar[:, i] * c_[:, i])
      else:
        y_ = y_ + (alpha[i] * xstar[:, i])
    y_ = y_ + ey_.squeeze()
    ystars.append(y_)
  ystar = np.array(ystars).mean(0)

  values = whiten({'x': x, 'y': y, 'z': z, 'confounder': c,'ey': ey})

  xstar = whiten_with_mu_std(xstar, values["x_mu"], values["x_std"])
  xstar_plotting = np.array([i for i in range(5)])
  xstar_grid = xstar
  xstar_grid_plotting = xstar_plotting

  return(values, xstar, ystar, xstar_grid, xstar_grid_plotting, xstar_plotting)


def fd_get_synth_data_2d(
  key: np.ndarray,
  num: int,
  equations,
  axis,
  num_xstar_bound,
  num_xstar: int = 100,
  external_equations = None
):
  """Generate some 2d synthetic data for the FD setting.

    key: A JAX random key.
    num: The number of samples to generate.
    equations: Which structural equations to choose for x and y. 
      Provided as a dict with keys 'f_x', 'f_y', 'f_z', 'confounder', 'ex', 'ey'
    axis: Which axis to vary to get x* values to calculate bounds at, keeping the
      others fixed.
    num_xstar_bound: Number of x* values chosen to calculate bounds at.
    num_xstar: Number of x* values chosen to calculate true E[Y|do(x*)] at.
    external_equations: The dictionary to use for the equations.

  Returns:
    values: A dictionary containing values for x, y, z, confounder, ex, ey.
    xstar: The x* values chosen to calculate true E[Y|do(x*)] at.
    ystar: The true E[Y|do(x*)] values corresponding to xstar.
    xstar_grid: The x* values chosen to calculate bounds at.
    xstar_grid_plotting: The values to plot on the x-axis to show the xstar_grid.
    xstar_plotting: The values to plot on the x-axis to show the xstar vs ystar curve.

  """
  if external_equations is not None:
      eqs = external_equations
  else:
      eqs = structural_equations_fd[equations]
  
  # Get x_value for xstar_bound to make sure we have the same each time
  
  key_bound = random.PRNGKey(40)
  key_bound, subkey_bound = random.split(key_bound)
  _ex = eqs["ex"](key_bound, 2000)
  key_bound, subkey_bound = random.split(key_bound)
  _u = eqs["u_xy"](key_bound, 2000)
  _x_bound = eqs["f_x"](np.array(_u), np.array(_ex))
  _x_bound = _x_bound.T
  #import pdb; pdb.set_trace()

  key, subkey = random.split(key)
  ex = eqs["ex"](subkey, num)
  key, subkey = random.split(key)
  em = eqs["em"](subkey, num)
  key, subkey = random.split(key)
  ey = eqs["ey"](subkey, num)
  key, subkey = random.split(key)
  c = eqs["confounder"](subkey, num)
  key, subkey = random.split(key)
  u = eqs["u_xy"](subkey, num)
  key, subkey = random.split(key)
  x = eqs["f_x"](np.array(u), np.array(ex))
  m = eqs["f_m"](np.array(x), np.array(c), np.array(em))
  c1, c2 = c
  m1, m2 = m
  u1, u2 = u
  y = eqs["f_y"](m1, m2, c1, c2, u1, u2, np.array(ey))
  x = x.T
  m = m.T
  c = np.array(c).T
  ex = np.array(ex).T

  values = whiten({'x': x, 'y': y, 'm': m, 'x_temp': _x_bound})

  # Evaluate E[ Y | do(x^*)] empirically
  if(axis==0):
      xmin_b, xmax_b = np.min(_x_bound[:,axis]), np.max(_x_bound[:,axis])
      _diff = xmax_b - xmin_b
      xmin_b = xmin_b + _diff/10; xmax_b = xmax_b - _diff/10

      xmin, xmax = np.min(x[:,axis]), np.max(x[:,axis])
      x2_0 = np.mean(_x_bound[:,1])
      x1star = np.linspace(xmin, xmax, num_xstar)
      x2star = np.full_like(x1star, x2_0)
      x1star_bound = np.linspace(xmin_b, xmax_b, num_xstar_bound)
      x2star_bound = np.full_like(x1star_bound, x2_0)
      xstar_grid_plotting = x1star_bound
      xstar_plotting = x1star
  if(axis==1):
      xmin_b, xmax_b = np.min(_x_bound[:,axis]), np.max(_x_bound[:,axis])
      _diff = xmax_b - xmin_b
      xmin_b = xmin_b + _diff/10; xmax_b = xmax_b - _diff/10

      xmin, xmax = np.min(x[:,axis]), np.max(x[:,axis])
      x1_0 = np.mean(_x_bound[:,0])
      x2star = np.linspace(xmin, xmax, num_xstar)
      x1star = np.full_like(x2star, x1_0)
      x2star_bound = np.linspace(xmin_b, xmax_b, num_xstar_bound)
      x1star_bound = np.full_like(x2star_bound, x1_0)
      xstar_grid_plotting = x2star_bound
      xstar_plotting = x2star
  ystar = []
  for _ in range(15000):
      key, subkey = random.split(key)
      tmpey = eqs["ey"](subkey, num_xstar)
      key, subkey = random.split(key)
      tmpem = eqs["em"](subkey, num_xstar)
      key, subkey = random.split(key)
      tmpconf = eqs["confounder"](subkey, num_xstar)
      key, subkey = random.split(key)
      tmp_u = eqs["u_xy"](subkey, num_xstar)
      tmp_m = eqs["f_m"](np.vstack((x1star, x2star)), np.array(tmpconf), np.array(tmpem))
      tc1, tc2 = tmpconf
      tm1, tm2 = tmp_m
      tu1, tu2 = tmp_u
      tmp_ystar = whiten_with_mu_std(
        eqs["f_y"](tm1, tm2, tc1, tc2, tu1, tu2, tmpey), values["y_mu"], values["y_std"])
      ystar.append(tmp_ystar)
  ystar = np.array(ystar) 
  xstar = np.vstack((x1star, x2star)).T
  xstar_grid = np.vstack((x1star_bound, x2star_bound)).T
  xstar = whiten_with_mu_std(xstar, values["x_temp_mu"], values["x_temp_std"])
  xstar_plotting = get_nonconstant_axis(xstar)
  xstar_grid = whiten_with_mu_std(xstar_grid, values['x_temp_mu'], values['x_temp_std'])
  xstar_grid_plotting = get_nonconstant_axis(xstar_grid)
  for k in list(values.keys()):
      if(k.startswith("x_temp")): del values[k]

  return values, xstar, ystar.mean(0), xstar_grid, xstar_grid_plotting, xstar_plotting


