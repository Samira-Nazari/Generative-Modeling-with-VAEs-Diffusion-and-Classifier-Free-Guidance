"""
Solutions for Question 1 of hwk3.
@author: Shawn Tan and Jae Hyun Lim
"""
import math
import numpy as np
import torch

torch.manual_seed(42)

def log_likelihood_bernoulli(mu, target):
    """ 
    COMPLETE ME. DONT MODIFY THE PARAMETERS OF THE FUNCTION. Otherwise, tests might fail.

    *** note. ***

    :param mu: (FloatTensor) - shape: (batch_size x input_size) - The mean of Bernoulli random variables p(x=1).
    :param target: (FloatTensor) - shape: (batch_size x input_size) - Target samples (binary values).
    :return: (FloatTensor) - shape: (batch_size,) - log-likelihood of target samples on the Bernoulli random variables.
    """
    # init
    batch_size = mu.size(0)
    mu = mu.view(batch_size, -1)
    target = target.view(batch_size, -1)

    #TODO: compute log_likelihood_bernoulli
    eps = 1e-6
    mu = torch.clamp(mu, min=eps, max=1 - eps)

    ll_bernoulli = target * torch.log(mu) + (1 - target) * torch.log(1 - mu)
    ll_bernoulli = ll_bernoulli.sum(dim=1)
    
    return ll_bernoulli


def log_likelihood_normal(mu, logvar, z):
    """ 
    COMPLETE ME. DONT MODIFY THE PARAMETERS OF THE FUNCTION. Otherwise, tests might fail.

    *** note. ***

    :param mu: (FloatTensor) - shape: (batch_size x input_size) - The mean of Normal distributions.
    :param logvar: (FloatTensor) - shape: (batch_size x input_size) - The log variance of Normal distributions.
    :param z: (FloatTensor) - shape: (batch_size x input_size) - Target samples.
    :return: (FloatTensor) - shape: (batch_size,) - log probability of the sames on the given Normal distributions.
    """
    # init
    batch_size = mu.size(0)
    mu = mu.view(batch_size, -1)
    logvar = logvar.view(batch_size, -1)
    z = z.view(batch_size, -1)

    #TODO: compute log normal
    var = torch.exp(logvar)
    squared_term = (z - mu) ** 2 / var
    log_likelihood = -0.5 * (math.log(2 * math.pi) + logvar + squared_term)
    ll_normal = log_likelihood.sum(dim=1)
    
    return ll_normal


def log_mean_exp(y):
    """ 
    COMPLETE ME. DONT MODIFY THE PARAMETERS OF THE FUNCTION. Otherwise, tests might fail.

    *** note. ***

    :param y: (FloatTensor) - shape: (batch_size x sample_size) - Values to be evaluated for log_mean_exp. For example log proababilies
    :return: (FloatTensor) - shape: (batch_size,) - Output for log_mean_exp.
    """
    # init
    batch_size = y.size(0)
    sample_size = y.size(1)

    #TODO: compute log_mean_exp
    # max per row to use log-sum-exp trick
    max_result = torch.max(y, dim=1, keepdim=True)  # (values, indices)
    a_values = max_result[0]  # max values only
    a = a_values

    # stable log-mean-exp
    y_shifted = y - a 
    mean_exp = torch.mean(torch.exp(y_shifted), dim=1)  # mean of exponentials
    a_squeezed = a.squeeze(1)
    log_mean = torch.log(mean_exp)
    lme = log_mean + a_squeezed  

    return lme 


def kl_gaussian_gaussian_analytic(mu_q, logvar_q, mu_p, logvar_p):
    """ 
    COMPLETE ME. DONT MODIFY THE PARAMETERS OF THE FUNCTION. Otherwise, tests might fail.

    *** note. ***

    :param mu_q: (FloatTensor) - shape: (batch_size x input_size) - The mean of first distributions (Normal distributions).
    :param logvar_q: (FloatTensor) - shape: (batch_size x input_size) - The log variance of first distributions (Normal distributions).
    :param mu_p: (FloatTensor) - shape: (batch_size x input_size) - The mean of second distributions (Normal distributions).
    :param logvar_p: (FloatTensor) - shape: (batch_size x input_size) - The log variance of second distributions (Normal distributions).
    :return: (FloatTensor) - shape: (batch_size,) - kl-divergence of KL(q||p).
    """
    # init
    batch_size = mu_q.size(0)
    mu_q = mu_q.view(batch_size, -1)
    logvar_q = logvar_q.view(batch_size, -1)
    mu_p = mu_p.view(batch_size, -1)
    logvar_p = logvar_p.view(batch_size, -1)

    #TODO: compute kld
    var_q = torch.exp(logvar_q)
    var_p = torch.exp(logvar_p)
    
    term1 = logvar_p - logvar_q
    term2 = (var_q + (mu_q - mu_p) ** 2) / var_p
    kl_element = term1 + term2 - 1
    
    kl_gg = 0.5 * kl_element.sum(dim=1)

    return kl_gg


def kl_gaussian_gaussian_mc(mu_q, logvar_q, mu_p, logvar_p, num_samples=1):
    """ 
    COMPLETE ME. DONT MODIFY THE PARAMETERS OF THE FUNCTION. Otherwise, tests might fail.

    *** note. ***

    :param mu_q: (FloatTensor) - shape: (batch_size x input_size) - The mean of first distributions (Normal distributions).
    :param logvar_q: (FloatTensor) - shape: (batch_size x input_size) - The log variance of first distributions (Normal distributions).
    :param mu_p: (FloatTensor) - shape: (batch_size x input_size) - The mean of second distributions (Normal distributions).
    :param logvar_p: (FloatTensor) - shape: (batch_size x input_size) - The log variance of second distributions (Normal distributions).
    :param num_samples: (int) - shape: () - The number of sample for Monte Carlo estimate for KL-divergence
    :return: (FloatTensor) - shape: (batch_size,) - kl-divergence of KL(q||p).
    """
    # init
    batch_size = mu_q.size(0)
    input_size = np.prod(mu_q.size()[1:])
    mu_q = mu_q.view(batch_size, -1).unsqueeze(1).expand(batch_size, num_samples, input_size)
    logvar_q = logvar_q.view(batch_size, -1).unsqueeze(1).expand(batch_size, num_samples, input_size)
    mu_p = mu_p.view(batch_size, -1).unsqueeze(1).expand(batch_size, num_samples, input_size)
    logvar_p = logvar_p.view(batch_size, -1).unsqueeze(1).expand(batch_size, num_samples, input_size)

    #TODO: kld

    std_q = torch.exp(0.5 * logvar_q)
    eps = torch.randn(batch_size, num_samples, input_size, device=mu_q.device)
    z = mu_q + eps * std_q

    # log q(z)
    squared_term_q = ((z - mu_q) ** 2) / torch.exp(logvar_q)
    log_qz = -0.5 * (logvar_q + squared_term_q + math.log(2 * math.pi))
    log_qz = log_qz.sum(dim=2)

    # log p(z)
    squared_term_p = ((z - mu_p) ** 2) / torch.exp(logvar_p)
    log_pz = -0.5 * (logvar_p + squared_term_p + math.log(2 * math.pi))
    log_pz = log_pz.sum(dim=2)

    kl_mc = (log_qz - log_pz).mean(dim=1)  # mean over samples

    return kl_mc
