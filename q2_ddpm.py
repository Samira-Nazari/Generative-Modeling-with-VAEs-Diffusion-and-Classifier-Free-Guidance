import torch 
from torch import nn 
from typing import Optional, Tuple


class DenoiseDiffusion():
    def __init__(self, eps_model: nn.Module, n_steps: int, device: torch.device):
        super().__init__()
        self.eps_model = eps_model
        self.beta = torch.linspace(0.0001, 0.02, n_steps).to(device)
        self.alpha = 1.0 - self.beta
        self.alpha_bar = torch.cumprod(self.alpha, dim=0)
        self.n_steps = n_steps
        self.sigma2 = self.beta


    ### UTILS
    def gather(self, c: torch.Tensor, t: torch.Tensor):
        #c_ = c.gather(-1, t)
        #return c_.reshape(-1, 1, 1, 1)
        return c[t].reshape(-1, 1, 1, 1)

    ### FORWARD SAMPLING
    def q_xt_x0(self, x0: torch.Tensor, t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # TODO: return mean and variance of q(x_t|x_0)
        alpha_bar_t = self.gather(self.alpha_bar, t)
        mean = torch.sqrt(alpha_bar_t) * x0
        var = 1 - alpha_bar_t

        return mean, var

    def q_sample(self, x0: torch.Tensor, t: torch.Tensor, eps: Optional[torch.Tensor] = None):
        if eps is None:
            eps = torch.randn_like(x0)
        # TODO: return x_t sampled from q(•|x_0) according to (1)
        mean, var = self.q_xt_x0(x0, t)
        sample = mean + torch.sqrt(var) * eps

        return sample

    ### REVERSE SAMPLING
    def p_xt_prev_xt(self, xt: torch.Tensor, t: torch.Tensor):
        # TODO: return mean and variance of p_theta(x_{t-1} | x_t) according to (2)
        beta_t = self.gather(self.beta, t)
        alpha_t = self.gather(self.alpha, t)
        alpha_bar_t = self.gather(self.alpha_bar, t)

        eps_theta = self.eps_model(xt, t)

        mean = (1 / torch.sqrt(alpha_t)) * (xt - (1 - alpha_t) / torch.sqrt(1 - alpha_bar_t) * eps_theta)
        var = beta_t
        return mean, var

    # TODO: sample x_{t-1} from p_theta(•|x_t) according to (3)
    def p_sample(self, xt: torch.Tensor, t: torch.Tensor, set_seed=False):
        if set_seed:
            torch.manual_seed(42)
        mean, var = self.p_xt_prev_xt(xt, t)

        if (t == 0).all():
            return mean

        noise = torch.randn_like(xt)
        sample = mean + torch.sqrt(var) * noise

        return sample

    ### LOSS
    # TODO: compute loss according to (4)
    def loss(self, x0: torch.Tensor, noise: Optional[torch.Tensor] = None, set_seed=False):
        if set_seed:
            torch.manual_seed(42)
        batch_size = x0.shape[0]
        dim = list(range(1, x0.ndim))
        t = torch.randint(
            0, self.n_steps, (batch_size,), device=x0.device, dtype=torch.long
        )
        if noise is None:
            noise = torch.randn_like(x0)
        # TODO
        x_t = self.q_sample(x0, t, eps=noise)
        noise_pred = self.eps_model(x_t, t)


        #loss = (noise_pred - noise)**2
        #loss = loss.mean(dim=[1, 2, 3])
        #loss = loss.mean()
        loss = ((noise - noise_pred)**2).sum(dim=dim).mean()

        return loss
