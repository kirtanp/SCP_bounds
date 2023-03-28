# ReadMe
Code for the paper [Stochastic Causal Programming for Bounding Treatment Effects](https://arxiv.org/pdf/2202.10806.pdf).

# Code Overview
Requirements are mentioned in `requirements.txt`. Best to install those in a separate environment using conda, venv or some such tool.
## Data
Code for synthetic data generation is in `src/utils/data.py`.

## Scripts
`src/IV_SCP.py` implements the stochastic causal program as derieved for the Instrument Variable (IV) setting in Section 3 of the paper, meaning that the constraints are in closed form and the constrainted optimization formulation has been implemented as an augmented lagrangian with inequality constraints. Parameters for this can be set from `src/configs/iv_scp.yaml`.

`src/FD_SCP.py` implements the stochastic causal program as derieved for the Leaky Front Door (FD) setting in Section 3 of the paper, meaning that the constraints are in closed form and the constrainted optimization formulation has been implemented as an augmented lagrangian with inequality constraints. Parameters for this can be set from `src/configs/fd_scp.yaml`.


## Usage Tips
* It is recommened to change at least the values of `dim_theta` and `response_type` depending on the problem you have at hand and the dimensionality of your treatment space.
* The method is supposed to work best when you take minimum over 5 runs for each lower bound and the maximum over 5 runs for each upper bound.

## Running the code
To run the code, you simply need to set the parameters in the respective config files and then simply call `python IV_SCP.py` or `python FD_SCP.py`. It is possible to provide your own dataset in the form of a dictionary, with keys `x`, `y` `z` and `xstar_grid` for the IV setting and keys `x`, `y` `m` and `xstar_grid` for the FD setting. Here `xstar_grid` is the list of $x*$ values for which you want the bounds on $\mathbb{E}[y|do(x^*)]$.

Code hsa been changed significantly in structure (but not function or logic) from the paper so if something doesn't work feel free to reach out or raise an issue. Code for the general implementation will be added soon as well.

Note: The methods whitens the data, so the final results that come out have whitened values of interventions and effects. This will be changed soon.
