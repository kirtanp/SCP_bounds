import numpy as np

import torch
from torch import nn

from torch.utils.data import DataLoader, TensorDataset, random_split
from torch.distributions.multivariate_normal import MultivariateNormal

import pyro.distributions as dist
import pyro.distributions.transforms as T

import pytorch_lightning as pl



def jax_to_torch(a):
    """Convert jax array to torch tensor"""
    a = np.asarray(a)
    b = torch.from_numpy(a)
    return b

def standardize_data_shapes(data):
    """ Make x, z and m batched"""
    keys_of_interest = set(['x', 'm', 'z'])
    all_keys = set(data.keys())
    for key in all_keys.intersection(keys_of_interest):
        if(data[key].ndim == 1):
            data[key] = data[key].unsqueeze(1)
    return(data)


def psi_poly(dim_theta):
    """The univariate polynomial response function."""
    def psi_vector(x):
        exp = torch.tensor([i for i in range(dim_theta)])
        return torch.pow(x, exp)
    return psi_vector

def psi_polyn(dim_x, dim_theta):
    """The multivariate polynomial response function."""
    if(dim_theta > dim_x*(dim_x + 3)/2 + 1):
        raise ValueError(f"dim_theta has to be dim_x*(dim_x + 3)/2 + 1 or less. Please set \
                           dim_theta to be {dim_x*(dim_x + 3)/2 + 1} or less. \
                           The current value is {dim_theta}")
    def psi_vector(x):
        if(x.ndim == 1):
            x = x.unsqueeze(0)
        x = x.unsqueeze(1)
        tril = torch.tril_indices(dim_x, dim_x)
        second_order = torch.einsum('bij,bjk->bik', torch.transpose(x, 1, 2), x)
        second_order = second_order[:, tril[0], tril[1]]
        first_order = torch.cat((torch.ones(x.shape[0], 1), x.squeeze(1)), 1)
        psi = torch.cat((first_order, second_order), 1)[:,:dim_theta]
        return psi.squeeze()
    
    return psi_vector


def is_psd(mat):
    """Check if matrix 'mat' is positive semidefinite"""
    #print(torch.linalg.eigvals(mat))
    return bool(torch.all(torch.linalg.eigvals(mat).real>=0))

def vec_to_sigma(vec, dim_out):
    """Converting vector to matrix while keeping grads intact"""
    omega = vec[:,:dim_out]
    omega = 1e-2 + torch.exp(omega)
    omega = torch.diag_embed(omega)

    tril_indices = torch.tril_indices(dim_out, dim_out, -1)
    chol = torch.zeros(len(vec), dim_out, dim_out)
    chol[:, tril_indices[0], tril_indices[1]] = vec[:, dim_out:]
    #print(vec.shape, omega.shape, chol.shape)
    chol = chol + omega
    chol_t = torch.transpose(chol, 1, 2)
    sigma = torch.matmul(chol, chol_t)
    return sigma


class Gaussian(pl.LightningModule):
    """Learning mean and covariance of target given input"""
    def __init__(self, dimension_in, dimension_out):
        super().__init__()
        self.dim_out = dimension_out
        self.dim_in = dimension_in
        if(self.dim_out == 1):
            self.dim_cov = 1
        else:
            self.dim_cov = (self.dim_out*(self.dim_out + 1))//2
        self.mean = nn.Sequential(
            nn.Linear(self.dim_in, 16),
            nn.ReLU(),
            nn.Linear(16, self.dim_out)
        )
        self.covariance = nn.Sequential(
            nn.Linear(self.dim_in, 32),
            nn.ReLU(),
            nn.Linear(32, self.dim_cov)
        )
    
    def _vec_to_chol(self, vec):
        """Converting vector to matrix while keeping it """
        omega = vec[:,:self.dim_out]
        omega = 1e-2 + torch.exp(omega)
        omega = torch.diag_embed(omega)
        
        tril_indices = torch.tril_indices(self.dim_out, self.dim_out, -1)
        chol = torch.zeros(len(vec), self.dim_out, self.dim_out)
        chol[:, tril_indices[0], tril_indices[1]] = vec[:, self.dim_out:]
        #print(vec.shape, omega.shape, chol.shape)
        chol = chol + omega
        return chol 

    def forward(self, x):
        # in lightning, forward defines the prediction/inference actions
        return self.mean(x), self.covariance(x)

    def training_step(self, batch, batch_idx):
        # training_step defined the train loop.
        # It is independent of forward
        z, x = batch
        mean, cov_vec = self.forward(z)
        chol = self._vec_to_chol(cov_vec)
        
        dist = MultivariateNormal(loc=mean, scale_tril=chol)
        loss = -torch.sum(dist.log_prob(x))
        
        # Logging to TensorBoard by default
        self.log('train_loss', loss)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-2)
        return optimizer


class FitRegressor(pl.LightningModule):
    """Learning a regression function"""
    def __init__(self, dimension_in, dimension_basis, lr=0.01):
        super().__init__()
        self.dim_basis = dimension_basis
        self.dim_in = dimension_in
        self.basis = nn.Sequential(
            nn.Linear(self.dim_in, 16),
            nn.ReLU(),
            nn.Linear(16, self.dim_basis)
        )
        self.final = nn.Sequential(
            nn.ReLU(),
            nn.Linear(self.dim_basis, 1)
        )
        self.lr = lr
        
    def forward(self, x):
        # in lightning, forward defines the prediction/inference actions
        return self.final(self.basis(x))

    def training_step(self, batch, batch_idx):
        # training_step defined the train loop.
        # It is independent of forward
        x, y = batch
        y_hat = self.forward(x)
        loss = nn.MSELoss()
        l = loss(y.squeeze(), y_hat.squeeze())
        return l

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer


class PhiDataModule(pl.LightningDataModule):
    def __init__(self, xz, y, batch_size):
        super().__init__()
        self.y = y
        self.batch_size = batch_size
        self.xz = xz
        self.dataset = TensorDataset(self.xz, self.y)
        #self.xz_train, self.xz_val = random_split(self.dataset, [4000, 1000])

    def train_dataloader(self):
        return DataLoader(self.dataset, batch_size=self.batch_size)
    
    #def val_dataloader(self):
    #    return DataLoader(self.xz_val, batch_size=self.batch_size)
    
class DataModule(pl.LightningDataModule):
    """z is input and x is output"""
    def __init__(self, z, x, batch_size):
        super().__init__()
        self.x = x
        self.z = z
        self.batch_size = batch_size
        self.dataset = TensorDataset(self.z, self.x)
        #self.xz_train, self.xz_val = random_split(self.dataset, [4000, 1000])

    def train_dataloader(self):
        return DataLoader(self.dataset, batch_size=self.batch_size)

def get_basis(dim_in, dim_theta, data_in, data_out):
    """Gtting the MLP basis"""
    basis = FitRegressor(dim_in, dim_theta)
    basis_dm = DataModule(data_in, data_out, 256)
    basis_trainer = pl.Trainer(max_epochs=100)
    basis_trainer.fit(basis, basis_dm)
    return basis


def get_indices(data, num_samples, data_seed):
    """Randomly sample indices. Returns 'num_samples' number of indices 
    among the total number of data points."""
    indices = torch.ones(len(data['x']))
    torch.random.manual_seed(data_seed)
    indices = torch.multinomial(indices, num_samples)
    return indices


def get_lhs(dim_in, data_in, y, constr_indices):
    """Return the lhs of the constraints. Run once up front"""
    phi1 = FitRegressor(dim_in, 16)

    phi1_dm = PhiDataModule(data_in, y, 256)
    phi1_trainer = pl.Trainer(max_epochs=250)
    phi1_trainer.fit(phi1, phi1_dm)

    phi2 = FitRegressor(dim_in, 16)

    phi2_dm = PhiDataModule(data_in , y**2, 256)
    phi2_trainer = pl.Trainer(max_epochs=250)
    phi2_trainer.fit(phi2, phi2_dm)

    lhs_all = torch.cat((phi1(data_in).squeeze(), phi2(data_in).squeeze()), 0).clone().detach()
    lhs1 = phi1(data_in).squeeze()[constr_indices]
    lhs2 = phi2(data_in).squeeze()[constr_indices]
    return torch.cat((lhs1, lhs2), 0).clone().detach(), lhs_all

def get_n_inferred(dim_in, dim_out, data_in, data_out, model_xz, return_flow=False):
    """Infer the N matrix from the data, to be run once up front"""
    if(model_xz == 'gaussian'):
        model_xz = Gaussian(dim_in, dim_out)
        xz_dm = DataModule(data_in, data_out, 256)
        xz_trainer = pl.Trainer(max_epochs=100)
        xz_trainer.fit(model_xz, xz_dm)

        mu, vecs = model_xz(data_in)
        sigma = vec_to_sigma(vecs.clone().detach(), dim_out)
        A, B = sigma, data_out - mu.clone().detach()

        if(data_out.ndim > 1):
            N = torch.linalg.solve(A, B)
        elif(data_out.ndim == 1):
            batch_size = len(A)
            A = A.view(batch_size, 1, 1)
            B = B.view(batch_size, 1, 1)
            N = torch.linalg.solve(A, B).squeeze()
        return N
    elif(model_xz == 'flow'):
        base_dist = dist.MultivariateNormal(torch.zeros(dim_out), torch.eye(dim_out))

        bound = torch.ceil(torch.max(torch.abs(data_out))).item()
        x_transform = T.conditional_spline(dim_out, context_dim=dim_in, bound=bound, order='quadratic')
        dist_x_given_z = dist.ConditionalTransformedDistribution(base_dist, [x_transform])

        modules = torch.nn.ModuleList([x_transform])
        optimizer = torch.optim.Adam(modules.parameters(), lr=3e-3)

        for step in range(2000):
            optimizer.zero_grad()
            ln_p_x_given_z = dist_x_given_z.condition(data_in).log_prob(data_out)
            loss = -(ln_p_x_given_z).mean()
            loss.backward()
            optimizer.step()
            dist_x_given_z.clear_cache()

        N = x_transform.condition(data_in).inv(data_out)
        if(return_flow):
            return N.clone().detach(), x_transform
        else: return N.clone().detach()

def low_likelihood_indices(n_inferred, dim_x, num_to_remove):
    """Find outliers from the dataset and return their indices"""
    dist = MultivariateNormal(loc = torch.zeros(dim_x), \
                              covariance_matrix = torch.eye(dim_x))
    log_likelihood = dist.log_prob(n_inferred)
    indices_remove = torch.sort(log_likelihood).indices[:num_to_remove]
    return(indices_remove)

def tensor_difference(t1, t2):
    "Returns elements in t1 but not in t2 (t1 - t2 as sets)"
    indices = torch.ones_like(t1, dtype = torch.bool)
    for elem in t2:
        indices = indices & (t1 != elem)  
    t1_without_t2 = t1[indices] 
    return t1_without_t2


def initialize_model_weights(x, model_theta, psi, lhs, N, dim_theta):
    """Initialize model weights by minimizing lhs-rhs"""
    def get_loss():
        mu_theta, vec_theta = model_theta(N)
        sigma_theta = vec_to_sigma(vec_theta, dim_theta)

        psi_x = psi(x).detach()
        rhs1 = torch.sum(psi_x * mu_theta, 1).squeeze()

        psi_x = psi_x.unsqueeze(1)
        mu_theta = mu_theta.unsqueeze(1)
        sig_adj = sigma_theta + torch.matmul(torch.transpose(mu_theta, 1, 2),
                                            mu_theta)
        rhs2 = torch.matmul(psi_x, sig_adj)
        rhs2 = torch.matmul(rhs2, torch.transpose(psi_x, 1, 2)).squeeze()
        rhs = torch.cat((rhs1, rhs2), 0)
        constr = lhs - rhs
        return torch.linalg.norm(constr)
    optimizer = torch.optim.Adam(model_theta.parameters(), lr=1e-3)
    for j in range(250):
        l = get_loss()
        optimizer.zero_grad()
        l.backward(retain_graph=True)
        optimizer.step()
    return model_theta


def get_optimal_obj(bound, diffmean, obj_step, sec_grad_limit=0.1):
    """Retrospectively get the best objective value"""
    # Get index to use for the objective
    sec_grad_abs = np.abs(np.gradient(diffmean, 2))
    where_high_grad = np.where(sec_grad_abs > sec_grad_limit)[0]
    if len(where_high_grad[np.where(where_high_grad > 5)]) == 0:
        if(bound == 'upper'):
            return obj_step.max()
        elif(bound == 'lower'):
            return obj_step.min()

    obj_idx = where_high_grad[np.where(where_high_grad > 5)][0] - 1
    obj = obj_step[obj_idx]
    return obj