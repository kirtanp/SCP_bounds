import yaml
import os

from datetime import datetime
import pickle

import numpy as np

import torch
from torch import optim
from torch.distributions.multivariate_normal import MultivariateNormal

import pyro.distributions as dist
import logging

from torch.utils.tensorboard import SummaryWriter

from utils import data
from utils import utils

from sacred import Experiment


ex = Experiment()

# Uncomment to run LOCALLY
ex.add_config("configs/fd_scp.yaml")

logger = logging.getLogger("mylogger")
logger.setLevel("INFO")

# # attach it to the experiment
ex.logger = logger


class Objective:
  """
  A class that represents the objective.
  We'll make it callable via the __call__ function.
  """

  def __init__(self, dim_m, psi_vector, x_transform, store_data, \
               sample_size_nx=200, sample_size_nm=50):
    self.distr = MultivariateNormal(torch.zeros(dim_m),
                                    torch.eye(dim_m))
    self.sample_size_nx = sample_size_nx
    self.sample_size_nm = sample_size_nm
    self.psi_vector = psi_vector
    self.x_transform = x_transform
    self.flow = dist.ConditionalTransformedDistribution(self.distr, [self.x_transform])
    self.store_data = store_data

  def __call__(self, model, x, xstar, results):
    # Sample n_x from data['x']
    #  sample flow(.|x*)
    #x = data['x']
    idx = torch.randint(0, len(x), (self.sample_size_nx, ))
    nx = x[idx]
    mu = torch.Tensor()
    psi_mstar = torch.Tensor()
    for nx_ in nx:
      nm_samples = self.distr.sample((self.sample_size_nm, ))
      m_samples = self.x_transform.condition(xstar)(nm_samples)
      nx_repeated_ = nx_.repeat(self.sample_size_nm, 1)
      psi_mstar_ = self.psi_vector(m_samples).detach()
      model_input = torch.cat((nx_repeated_, nm_samples), 1)
      mu_, _ = model(model_input)

      mu = torch.cat((mu, mu_), 0)
      psi_mstar = torch.cat((psi_mstar, psi_mstar_), 0)
      
    return torch.mean(torch.einsum("bi, bi -> b", psi_mstar, mu))

class Constraints:
  """
  A class that represents the constraints.
  We'll make it callable via the __call__ function.
  """

  def __init__(self, lhs, n_inferred, psi_vector, dim_theta, constr_indices, constr_type,\
               slack, save_rhs_every_step, store_data):
    """Initialize the Constraints class.

    Params:
      lhs: We can precompute the entire lhs and pass it in as
          a single 1D Tensor.
      n_inferred: We pre-compute the n = h^{-1}_{z_i}(x_i)
          for all datapoints and pass it in here.
      psi_vector: A callable that evaluates the basis functions
          at its argument.
      dim_theta: Dimension of theta.
      constr_indices: Indices of the data points that are used for the constraints.
      constr_type: Type of constraints. Either 'single' or 'multi'.
      slack: Slack variable for the constraints.
      save_rhs_every_step: Whether to save the rhs at every step.
      store_data: Whether to store the data.
    """
    self.lhs = lhs
    self.n_inferred = n_inferred
    self.psi_vector = psi_vector
    self.dim_theta = dim_theta
    self.constr_indices = constr_indices
    self.constr_type = constr_type
    self.n_constraints = 2 * len(constr_indices)
    self.slack = slack
    self.save_rhs_every_step = save_rhs_every_step
    self.store_data = store_data

  def __call__(self, model, results, opt_stp, data):
    """Constraints at current parameters and for given indices."""
    model_input = torch.cat((data['x'], self.n_inferred), 1)
    mu_theta, vec_theta = model(model_input[self.constr_indices])
    sigma_theta = utils.vec_to_sigma(vec_theta, self.dim_theta)

    psi_m = self.psi_vector(data['m'][self.constr_indices]).detach()
    # Double check whether dimensions are such that this really is the desired dot product.
    # If we can write it as matmul would be nicer
    rhs1 = torch.sum(psi_m * mu_theta, 1).squeeze()

    psi_m = psi_m.unsqueeze(1)
    mu_theta = mu_theta.unsqueeze(1)
    sig_adj = sigma_theta + torch.matmul(torch.transpose(mu_theta, 1, 2),
                                         mu_theta)
    rhs2 = torch.matmul(psi_m, sig_adj)
    rhs2 = torch.matmul(rhs2, torch.transpose(psi_m, 1, 2)).squeeze()
    rhs = torch.cat((rhs1, rhs2), 0)

    constr = self.lhs - rhs
    if(self.constr_type == 'single'):
      constr = constr[:len(constr)//2]
      constr = torch.mean((constr) ** 2)

    constrs = self.slack - torch.abs(constr)

    if(opt_stp == 0):
      results['rhs'].append(rhs.clone().detach().numpy())
    lhsmrhs = torch.abs(self.lhs - rhs.clone().detach())
    if(self.store_data):
      if(self.save_rhs_every_step == True):
        results['all_rhs'].append(rhs.clone().detach().numpy())
      satis_status = lhsmrhs <= self.slack
      results["lhsmrhs_min"].append(lhsmrhs.min().item())
      results["lhsmrhs_max"].append(lhsmrhs.max().item())
      results["lhsmrhs_norm"].append(torch.linalg.norm(lhsmrhs).item())
      results["satis_frac"].append(satis_status.sum().item()/(2*self.n_constraints))
      results["lhs"] = self.lhs
    results["lhsmrhs_mean"].append(lhsmrhs.mean().item())
    if(self.constr_type == 'single'):
      return constrs
    else:
      return constrs / len(self.constr_indices)


class AugmentedLagrangian:
  """
  Augmented Lagrangian method with inequality constraints
  for an objective that depends on data and with the option
  to subsample constraints.
  """

  def __init__(self,
               data,
               objective,
               constraints,
               constr_type,
               psi_vector,
               n_inferred,
               lhs_all,
               log_dir,
               n_constraints,
               tau_init=10,
               alpha=100):
    """Initialize the augmented Langrangian method.

    Params:
      data: Contains the data in some format that is accepted by the
          `objective` and `constraints` functions. For example, a
          dictionary with keys 'x', 'y', 'z', 'xstar' and tensors as values.
      objective: A callable taking the two positional arguments `data`,
          and `model` as well as potentially more keyword arguments and
          returning a single real number.
      constraints: A callable taking the two positional arguments `data`,
          `model`, and `indices` as well as potentially keyword arguments and
          returning a 1D tensor of values. These stand for the constraints
          that we are trying to get below the slack.
      constr_type: Type of constraints. Either 'single' or 'multi'.
      psi_vector: A callable that evaluates the basis functions
          at its argument.
      n_inferred: We pre-compute the n = h^{-1}_{z_i}(x_i)
          for all datapoints and pass it in here.
      lhs_all: The full left hand side of the constraints, not just the lhs for 
          the chosen indices. Mainly used for debugging and evaluation.
      log_dir: Directory to save the results.
      n_constraints: An integer indicating how many constraints to add to
          the Lagrangian in each round.
      tau_init: The initial value (real) of tau for the 'temperature' parameter
          of the augmented Lagrangian.
      alpha: The factor by which to increase `tau` when the constraints
          have not been reduced sufficiently.
    """
    self.data = data
    self.objective = objective
    self.constraints = constraints
    self.n_constraints = n_constraints
    self.psi_vector = psi_vector
    self.constr_type = constr_type
    self.n_inferred = n_inferred
    self.lhs_all = lhs_all
    self.log_dir = log_dir
    self.tau_init = tau_init
    self.alpha = alpha
    self.dim_x = data['x'].shape[1]
    self.dim_m = data['m'].shape[1]

  @ex.capture
  def optimize(self,
               bound,
               xstar,
               xstar_name,
               run_number,
               method_seed,
               _config,
               _log,
               _run,
               n_rounds=100,
               opt_steps=50,
               lr=0.01):
    """Run the augmented Lagrangian optimization."""
    # Initializing parameters

    out_dir = os.path.join(self.log_dir, f"{bound}-xstar_{xstar_name}")
    out_dir = os.path.join(out_dir, f"{run_number}")
    _log.info(f"Current run output directory: {out_dir}...")
    if not os.path.exists(out_dir):
      os.makedirs(out_dir)

    _log.info(f"Evaluate at xstar={xstar}...")
    _log.info(f"Evaluate {bound} bound...")

    _log.info(f"Initialize model and weights...")
    torch.manual_seed(method_seed)
    model = utils.Gaussian(self.dim_x + self.dim_m, _config['dim_theta'])
    n_inferred_w_nx = torch.cat((self.data['x'], self.n_inferred), 1) 
    model = utils.initialize_model_weights(self.data['m'], model, self.psi_vector, self.lhs_all,\
                                           n_inferred_w_nx, _config['dim_theta'])

    _log.info(f"Initialize dictionary for results...")
    results = {
      "rhs": [],
      "all_rhs": [],
      "objective": [],
      "objective_every_step": [],
      "lhsmrhs_mean": [],
      "lhsmrhs_max": [],
      "lhsmrhs_min": [],
      "lhsmrhs_norm": [],
      "satis_frac": []
    }
    _log.info(f"Store xstar value for easier cumulative analysis...")
    results['xstar'] = xstar
    results['xstar_name'] = xstar_name
    results['bound'] = bound
    
    sign = 1 if bound == "lower" else -1
    tau = self.tau_init
    alpha = self.alpha
    eta = 1 / tau ** 0.1
    lmbda = torch.ones(self.n_constraints)

    # Setup optimizer
    optimizer = optim.SGD(model.parameters(), lr=lr)

    writer = SummaryWriter(log_dir=out_dir)
    # Main optimization loop
    for rnd in range(n_rounds):
      #indices = self._get_indices()

      # Find approximate solution of subproblem at fixed lmbda
      for opt_stp in range(opt_steps):
        iter_idx = opt_steps * rnd + opt_stp

        # Compute augmented Lagrangian
        obj = self.objective(model, self.data['x'], xstar, results)
        constr = self.constraints(model, results, opt_stp, self.data)

        results['objective_every_step'].append(obj.item())

        case1 = - lmbda * constr + 0.5 * tau * constr**2
        case2 = - 0.5 * lmbda**2 / tau
        psi = torch.where(tau * constr <= lmbda, case1, case2)
        psisum = torch.sum(psi)
        lagrangian = sign*obj + psisum

        # Calculate gradients
        optimizer.zero_grad()
        lagrangian.backward()

        # Some tensorboard logging
        constr_norm = torch.linalg.norm(constr.clone().detach())
        writer.add_scalar("Optimization/objective", obj.item(), iter_idx)
        writer.add_scalar("Optimization/psisum", psisum.item(), iter_idx)
        writer.add_scalar("Optimization/lagrangian", lagrangian.item(), iter_idx)
        writer.add_scalar("Optimization/constraint_norm", constr_norm, iter_idx)
        writer.add_scalar("Optimization/tau", tau, iter_idx)
        writer.add_scalar("Optimization/eta", eta, iter_idx)
        for name, param in model.named_parameters():
          writer.add_scalar(f"GradNorm/{name}", param.grad.norm(), iter_idx)

        # Backprop
        optimizer.step()

      # Check current solution to subproblem
      results['objective'].append(obj.item())
      to_log = ['Round', rnd, ':', obj.item(), constr_norm.item(), lagrangian.item()]
      _log.info(' '.join(str(_) for _ in to_log))
      detached_constr = constr.clone().detach()
      if(torch.isnan(obj)):
        _log.info("Objective value is now NaN, stopping optimization...")
        break
      # lhsmrhs_normalized = |lhs-rhs|/2*n
      if(self.constr_type == 'single'):
        lhsmrhs_normalized = _config['slack'] - detached_constr
      else:
        lhsmrhs_normalized = _config['slack'] / len(detached_constr) - detached_constr 

      # Check if constraints are satisfied for inequality constraints
      cnorm = torch.linalg.norm(lhsmrhs_normalized)
      is_satisfied = np.all(lhsmrhs_normalized.numpy() >= 0)
      writer.add_scalar("Optimization/LHSmRHS_norm", cnorm.item(), iter_idx)
      if((cnorm < eta or is_satisfied)):
        # Global convergence check
        if(rnd>=1 and _config['global_conv_check']):
          if np.abs(results['objective'][-1] - results['objective'][-2]) <= 0.005:
            _log.info("Global convergence passed")
            break
        lmbda -= tau * detached_constr
        lmbda = torch.maximum(torch.tensor(0), lmbda)
        eta = torch.max(torch.tensor([eta / tau ** 0.5, _config['eta_min']]))
      else:
        # Increase penalty parameter, tighten tolerance
        tau = torch.min(torch.tensor([alpha * tau, _config['tau_max']]))
        eta = torch.max(torch.tensor([1 / tau ** 0.1, _config['eta_min']])) 

    _log.info(f"Finished optimization loop...")
    torch.save(model.state_dict(), os.path.join(out_dir, 'model_theta.ckpt'))

    _log.info(f"Convert all results to numpy arrays...")
    results = {k: np.array(v) for k, v in results.items()}
    
    # Retrospecitvely getting the best objective value
    fin_obj = utils.get_optimal_obj(bound, results['lhsmrhs_mean'], results['objective_every_step'])

    if _config['store_data']:
      _log.info(f"Save result data to...")
      result_path = os.path.join(out_dir, "results.npz")
      np.savez(result_path, **results)

    _log.info("Finished run.")
    
    return fin_obj

@ex.automain
def main(_config, _log):
  out_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + '_' + str(hash(tuple(_config.items())))
  out_dir = os.path.join(os.path.abspath(_config['output_dir']), out_name)
  _log.info(f"Save all output to {out_dir}...")
  if not os.path.exists(out_dir):
    os.makedirs(out_dir)

  # Save config
  with open(os.path.join(out_dir, 'config.yaml'), 'w') as fp:
    yaml.dump(_config, fp)

  _log.info(f"Get dataset...")
  if(_config['generate_data'] == True):
    dat = data.get_data('FD', _config['dataset'], _config['equations'], _config['num_data'],\
                        _config['num_xstar'], _config['xstar_axis'], _config['seed_data'])
  else:
    with open(_config['data_path'], 'rb') as f:
      dat = pickle.load(f)
    f.close()

  # Shapes
  # X=(num_data, dim_x), Z=(num_data, dim_z), Y=(num_data,)
  dat = utils.standardize_data_shapes(dat)
  xstar_grid = dat['xstar_grid']
  dim_x = dat['x'].shape[1]
  dim_m = dat['m'].shape[1]
  
  if('xstar_grid_plotting' in dat):
    xstar_grid_plotting = dat['xstar_grid_plotting']
  else:
    xstar_grid_plotting = ['x*' + str(i+1) for i in range(len(xstar_grid))]
  num_xstar = len(xstar_grid_plotting)

  _log.info(f"Get the basis function...")
  y_given_x =  utils.get_basis(dim_x, _config['dim_theta'], dat['x'], dat['y'])
  if(_config['response_type'] == 'mlp'):
    psi_vector =  utils.get_basis(dim_m, _config['dim_theta'], dat['m'], dat['y'], _config['dataset'], _config['equations'], out_dir).basis
  elif(_config['response_type'] == 'polynomial'):
    psi_vector = utils.psi_polyn(dim_m, _config['dim_theta'])
  _log.info("Get inferred N...")
  n_inferred, flow = utils.get_n_inferred(dim_x, dim_m, dat['x'], dat['m'], _config['model_mx'])
  indices_to_remove = utils.low_likelihood_indices(n_inferred, _config['dim_m'], _config['num_to_remove'])

  # We will be using the same set of indices for all xstars in the grid
  _log.info(f"Get the constraint indices to be used for subsampling...")
  constr_indices = utils.get_indices(dat, _config['num_constant_samples'])
  constr_indices = utils.tensor_difference(constr_indices, indices_to_remove)
  n_constraints = 2*len(constr_indices)

  _log.info("Get lhs...")
  xm = torch.cat((dat['x'], dat['m']), 1)
  lhs, lhs_all = utils.get_lhs(dim_x + dim_m, xm, dat['y'], constr_indices)
  
  objective = Objective(dim_x, psi_vector, flow, _config['store_data'])
  constraints = Constraints(lhs, n_inferred, psi_vector, _config['dim_theta'], constr_indices, _config['constr_type'],\
                _config['slack'], _config['save_rhs_every_step'], _config['store_data'])
  optim = AugmentedLagrangian(dat, objective, constraints, _config['constr_type'], psi_vector, n_inferred, lhs_all,\
                              out_dir, n_constraints, _config['tau_init'], _config['tau_factor'])
  
  results_global = {}
  if(_config['store_data']):
    results_global['lhs'] = lhs.clone().detach().numpy()
    results_global['n_inferred'] = n_inferred.clone().detach().numpy()
    results_global['constr_indices'] = constr_indices.numpy()

    result_path = os.path.join(out_dir, "results_global.npz")
    np.savez(result_path, **results_global)
  
  all_run_objs = np.zeros((num_xstar, 2, _config['num_runs_each_bound']))
  final_bounds = {'lower': [], 'upper': []}

  np.random.seed(_config['seed_method'])
  method_seeds = np.random.randint(0, 100000, size=_config['num_runs_each_bound'])
  # ---------------------------------------------------------------------------
  # Main loops over xstar and bounds
  # ---------------------------------------------------------------------------
  for i, xstar in enumerate(xstar_grid):
    xstar = utils.jax_to_torch(xstar).type(torch.FloatTensor)
    if(xstar.ndim == 0):
      xstar = torch.tensor([xstar])
    for j, bound in enumerate(["lower", "upper"]):
      # We take the min/max over multiple runs for each bound
      for k in range(_config['num_runs_each_bound']):
        num_runs = _config['num_runs_each_bound']
        _log.info(f"Run xstar={xstar_grid_plotting[i]}, bound={bound}, run={k+1}/{num_runs}...")
        vis = "=" * 10
        _log.info(f"{vis} {i * 2 * num_runs + k + 1}/{2 * num_xstar * num_runs} {vis}")
        fin_obj = optim.optimize(bound, xstar, xstar_grid_plotting[i], k, method_seeds[k], \
                                 n_rounds=_config['num_rounds'], opt_steps=_config['opt_steps'], lr=_config['lr'])
        all_run_objs[i, j, k] = fin_obj
      if(bound == 'lower'):
        final_bounds['lower'].append(all_run_objs[i, j, :].min())
        _log.info(f"Lower bound: {final_bounds['lower'][-1]}")
      else:
        final_bounds['upper'].append(all_run_objs[i, j, :].max())
        _log.info(f"Upper bound: {final_bounds['upper'][-1]}")

  bounds = {}
  bounds['all_runs'] = all_run_objs
  bounds['final_upper'] = np.array(final_bounds['upper'])
  bounds['final_lower'] = np.array(final_bounds['lower'])
  result_path = os.path.join(out_dir, "bounds.npz")
  np.savez(result_path, **bounds)
