"""
WGAN-GP Loss functions, Gradient Penalty calculation, and training step utilities.
Implements the Kantorovich-Rubinstein duality with exact 1-Lipschitz continuity penalty.
"""

import torch
import torch.nn as nn
from typing import List, Tuple, Union


def compute_gradient_penalty(
    critic: nn.Module,
    real_samples: torch.Tensor,
    fake_samples: torch.Tensor,
    device: torch.device
) -> torch.Tensor:
    """
    Computes the 1-Lipschitz Gradient Penalty for WGAN-GP.
    Supports both single-scale and multi-scale critics.
    
    L_GP = E_{\hat{x}} [ ( || \nabla_{\hat{x}} D(\hat{x}) ||_2 - 1 )^2 ]
    """
    batch_size = real_samples.size(0)
    
    # Random interpolation weight epsilon ~ U(0, 1)
    epsilon = torch.rand(batch_size, 1, 1, 1, device=device)
    interpolates = (epsilon * real_samples + ((1 - epsilon) * fake_samples)).requires_grad_(True)
    
    critic_interpolates = critic(interpolates)
    
    # If critic is multi-scale, it returns a list of tensors
    if isinstance(critic_interpolates, list):
        total_gp = torch.tensor(0.0, device=device)
        for d_out in critic_interpolates:
            grad_outputs = torch.ones_like(d_out, device=device, requires_grad=False)
            gradients = torch.autograd.grad(
                outputs=d_out,
                inputs=interpolates,
                grad_outputs=grad_outputs,
                create_graph=True,
                retain_graph=True,
                only_inputs=True
            )[0]
            gradients = gradients.view(batch_size, -1)
            grad_norm = gradients.norm(2, dim=1)
            gp = torch.mean((grad_norm - 1.0) ** 2)
            total_gp = total_gp + gp
        return total_gp / len(critic_interpolates)
    else:
        grad_outputs = torch.ones_like(critic_interpolates, device=device, requires_grad=False)
        gradients = torch.autograd.grad(
            outputs=critic_interpolates,
            inputs=interpolates,
            grad_outputs=grad_outputs,
            create_graph=True,
            retain_graph=True,
            only_inputs=True
        )[0]
        gradients = gradients.view(batch_size, -1)
        grad_norm = gradients.norm(2, dim=1)
        gp = torch.mean((grad_norm - 1.0) ** 2)
        return gp


def critic_loss(
    critic_real: Union[torch.Tensor, List[torch.Tensor]],
    critic_fake: Union[torch.Tensor, List[torch.Tensor]],
    drift_penalty: float = 0.001
) -> Tuple[torch.Tensor, float]:
    """
    Computes Wasserstein critic loss: E[D(fake)] - E[D(real)] + drift_penalty * E[D(real)^2]
    """
    if isinstance(critic_real, list):
        real_loss = sum(torch.mean(cr) for cr in critic_real) / len(critic_real)
        fake_loss = sum(torch.mean(cf) for cf in critic_fake) / len(critic_fake)
        drift = sum(torch.mean(cr ** 2) for cr in critic_real) / len(critic_real) * drift_penalty
    else:
        real_loss = torch.mean(critic_real)
        fake_loss = torch.mean(critic_fake)
        drift = torch.mean(critic_real ** 2) * drift_penalty
        
    loss = fake_loss - real_loss + drift
    wasserstein_distance = (real_loss - fake_loss).item()
    return loss, wasserstein_distance


def generator_loss(critic_fake: Union[torch.Tensor, List[torch.Tensor]]) -> torch.Tensor:
    """
    Computes generator loss: -E[D(fake)]
    """
    if isinstance(critic_fake, list):
        return -sum(torch.mean(cf) for cf in critic_fake) / len(critic_fake)
    else:
        return -torch.mean(critic_fake)
