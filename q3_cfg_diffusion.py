# %%
import torch
import torch.utils.data
import torchvision
from torch import nn
from typing import Tuple, Optional
import torch.nn.functional as F
from tqdm import tqdm
from easydict import EasyDict
import matplotlib.pyplot as plt
from torch.amp import GradScaler, autocast
import os 

from cfg_utils.args import * 


class CFGDiffusion():
    def __init__(self, eps_model: nn.Module, n_steps: int, device: torch.device):
        super().__init__()
        self.eps_model = eps_model
        self.n_steps = n_steps
        
        self.lambda_min = -20
        self.lambda_max = 20



    ### UTILS
    def get_exp_ratio(self, l: torch.Tensor, l_prim: torch.Tensor):
        return torch.exp(l-l_prim)
    
    def get_lambda(self, t: torch.Tensor): 
        # TODO: Write function that returns lambda_t for a specific time t. Do not forget that in the paper, lambda is built using u in [0,1]
        # Note: lambda_t must be of shape (batch_size, 1, 1, 1)
        # Normalize t into u ∈ [0, 1]
        #u = t.float() / self.n_steps
        # Linear interpolation between lambda_min and lambda_max
        #lambda_t = self.lambda_min + u * (self.lambda_max - self.lambda_min)
        # Reshape to (B, 1, 1, 1)
        #lambda_t = lambda_t.view(-1, 1, 1, 1)
        #return lambda_t

        # Normalize t into u ∈ [0, 1]
        #u = t.float() / self.n_steps

        # Compute constants a and b using lambda_min and lambda_max
        #b = torch.arctan(torch.exp(-self.lambda_max / 2))
        #a = torch.arctan(torch.exp(-self.lambda_min / 2)) - b

        # λ(u) = -2 * log(tan(a * u + b))
        # = -2.0 * torch.log(torch.tan(a * u + b))

        # Reshape to (B, 1, 1, 1)
        #return lambda_t.view(-1, 1, 1, 1)

        u = t.float() / self.n_steps  # normalize

        # Ensure all constants are tensors on the correct device
        device = t.device
        lambda_min = torch.tensor(self.lambda_min, device=device)
        lambda_max = torch.tensor(self.lambda_max, device=device)

        b = torch.arctan(torch.exp(-lambda_max / 2))
        a = torch.arctan(torch.exp(-lambda_min / 2)) - b

        lambda_t = -2.0 * torch.log(torch.tan(a * u + b))

        return lambda_t.view(-1, 1, 1, 1)
        
    
    def alpha_lambda(self, lambda_t: torch.Tensor): 
        #TODO: Write function that returns Alpha(lambda_t) for a specific time t according to (1)
        var = 1.0 / (1.0 + torch.exp(-lambda_t))  # var = alpha_lambda^2
        return var.sqrt()  # alpha_lambda = sqrt(alpha_lambda^2)
    
    def sigma_lambda(self, lambda_t: torch.Tensor): 
        #TODO: Write function that returns Sigma(lambda_t) for a specific time t according to (1)
        alpha_squared = 1.0 / (1.0 + torch.exp(-lambda_t))  # compute alpha_lambda^2
        var = 1.0 - alpha_squared  # var = sigma_lambda^2
        return var.sqrt()  # sigma_lambda = sqrt(var)
    
    ## Forward sampling
    def q_sample(self, x: torch.Tensor, lambda_t: torch.Tensor, noise: torch.Tensor):
        #TODO: Write function that returns z_lambda of the forward process, for a specific: x, lambda l and N(0,1) noise  according to (1)
        alpha_lambda_t = self.alpha_lambda(lambda_t)  # compute alpha(lambda)
        sigma_lambda_t = self.sigma_lambda(lambda_t)  # compute sigma(lambda)
        z_lambda_t = alpha_lambda_t * x + sigma_lambda_t * noise  # sample z_lambda
        return z_lambda_t
               
    def sigma_q(self, lambda_t: torch.Tensor, lambda_t_prim: torch.Tensor):
        #TODO: Write function that returns variance of the forward process transition distribution q(•|z_l) according to (2)
        # alpha^2_lambda
        alpha_squared = 1.0 / (1.0 + torch.exp(-lambda_t))  # alpha_lambda^2
        sigma_squared = 1.0 - alpha_squared  # sigma_lambda^2

        # (1 - exp(lambda - lambda'))
        coef = 1.0 - torch.exp(lambda_t - lambda_t_prim)

        var_q = coef * sigma_squared  # variance of q(z_lambda | z_lambda')
        return var_q.sqrt()  # return standard deviation
    
    def sigma_q_x(self, lambda_t: torch.Tensor, lambda_t_prim: torch.Tensor):
        #TODO: Write function that returns variance of the forward process transition distribution q(•|z_l, x) according to (3)
        # alpha^2_lambda'
        alpha_squared_prim = 1.0 / (1.0 + torch.exp(-lambda_t_prim))  # alpha_lambda'^2
        sigma_squared_prim = 1.0 - alpha_squared_prim  # sigma_lambda'^2

        coef = 1.0 - torch.exp(lambda_t - lambda_t_prim)  # (1 - exp(lambda - lambda'))
    
        var_q_x = coef * sigma_squared_prim  # variance of q(z_lambda' | z_lambda, x)
        return var_q_x.sqrt()  # return standard deviation

    ### REVERSE SAMPLING
    def mu_p_theta(self, z_lambda_t: torch.Tensor, x: torch.Tensor, lambda_t: torch.Tensor, lambda_t_prim: torch.Tensor):
        #TODO: Write function that returns mean of the forward process transition distribution according to (4)
        alpha_lambda_t = self.alpha_lambda(lambda_t)  # alpha(lambda)
        alpha_lambda_t_prim = self.alpha_lambda(lambda_t_prim)  # alpha(lambda')

        # exp_coef = torch.exp(lambda_t_prim - lambda_t)  # exp(lambda' - lambda)
        exp_coef = torch.exp(lambda_t - lambda_t_prim)  # Correct: lambda - lambda'
        one_minus_exp_coef = 1.0 - exp_coef  # (1 - exp(lambda' - lambda))

        term1 = exp_coef * (alpha_lambda_t_prim / alpha_lambda_t) * z_lambda_t
        term2 = one_minus_exp_coef * alpha_lambda_t_prim * x

        mu = term1 + term2
        return mu

    def var_p_theta(self, lambda_t: torch.Tensor, lambda_t_prim: torch.Tensor, v: float=0.3):
        #TODO: Write function that returns var of the forward process transition distribution according to (4)
                # alpha^2_lambda and alpha^2_lambda'
        alpha_squared = 1.0 / (1.0 + torch.exp(-lambda_t))  # alpha_lambda^2
        alpha_squared_prim = 1.0 / (1.0 + torch.exp(-lambda_t_prim))  # alpha_lambda'^2

        # sigma^2_lambda and sigma^2_lambda'
        sigma_squared = 1.0 - alpha_squared
        sigma_squared_prim = 1.0 - alpha_squared_prim

        coef = 1.0 - torch.exp(lambda_t - lambda_t_prim)  # (1 - exp(lambda - lambda'))

        sigma_q_lambda = coef * sigma_squared      # sigma^2_lambda|lambda'
        sigma_q_lambda_prim = coef * sigma_squared_prim  # tilde_sigma^2_lambda'|lambda

        var = sigma_q_lambda_prim**(1.0 - v) * sigma_q_lambda**v  # interpolation

        return var
    
    def p_sample(self, z_lambda_t: torch.Tensor, lambda_t : torch.Tensor, lambda_t_prim: torch.Tensor,  x_t: torch.Tensor, set_seed=False):
        # TODO: Write a function that sample z_{lambda_t_prim} from p_theta(•|z_lambda_t) according to (4) 
        # Note that x_t correspond to x_theta(z_lambda_t)
        if set_seed:
            torch.manual_seed(42)
        mu = self.mu_p_theta(z_lambda_t, x_t, lambda_t, lambda_t_prim)  # mean
        var = self.var_p_theta(lambda_t, lambda_t_prim)  # variance
        noise = torch.randn_like(z_lambda_t)  # epsilon ~ N(0, I)

        sample = mu + var.sqrt() * noise  # sample from N(mu, var)

        return sample 

    ### LOSS
    def loss(self, x0: torch.Tensor, labels: torch.Tensor, noise: Optional[torch.Tensor] = None, set_seed=False):
        if set_seed:
            torch.manual_seed(42)
        batch_size = x0.shape[0]
        dim = list(range(1, x0.ndim))
        t = torch.randint(
            0, self.n_steps, (batch_size,), device=x0.device, dtype=torch.long
        )
        if noise is None:
            noise = torch.randn_like(x0)
        #TODO: q_sample z
        # Compute lambda(t)
        lambda_t = self.get_lambda(t)  # shape (B, 1, 1, 1)
        # Generate noisy latent z_lambda
        #z_lambda = self.q_sample(x0, lambda_t, noise)
        x_t = self.q_sample(x0, lambda_t, noise)

        # Predict noise using epsilon model
        # pred_noise = self.eps_model(z_lambda, lambda_t.squeeze(), labels)
        # pred_noise = self.eps_model(z_lambda, t)
        epsilon_theta = self.eps_model(x_t, labels)

        # Compute sigma_lambda
        #sigma_lambda_val = self.sigma_lambda(lambda_t)

        # Correct reshape: don't squeeze everything
        #sigma_lambda_val = sigma_lambda_val.view(batch_size)

        #TODO: compute loss
        # Compute loss as: (1 / (2 * sigma^2)) * || pred - true noise ||^2
        #mse = F.mse_loss(pred_noise, noise, reduction='none')  # shape (B, C, H, W)
        #mse = mse.view(batch_size, -1).mean(dim=1)  # average over all dimensions except batch

        #sigma_squared = sigma_lambda_val.squeeze() ** 2    # shape (B,)
        #sigma_squared = sigma_lambda_val ** 2

        #loss = 0.5 * (mse / sigma_squared).mean()
        loss = ((noise - epsilon_theta)**2).sum(dim=dim).mean()
    
        return loss



    