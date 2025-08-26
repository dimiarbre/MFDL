import numpy as np
import torch
from torch.optim import Optimizer
from workloads_generator import compute_sensitivity


class MFDLSGD(Optimizer):
    def __init__(
        self,
        params,
        C,  # Correlation matrix, such that A = BC.
        participation_interval,
        C_sens=None,
        id=0,
        lr=1e-3,
        l2_norm_clip=1,
        noise_multiplier=1,
        batch_size=500,
        device: torch.device = torch.device("cpu"),
    ):
        # Initialize parameters
        defaults = dict(lr=lr)
        super(MFDLSGD, self).__init__(params, defaults)
        self.device = device
        self.id = id

        """ Differential privacy parameters """
        self.l2_norm_clip = l2_norm_clip
        # print(self.l2_norm_clip)
        self.noise_multiplier = noise_multiplier
        self.batch_size = batch_size

        """ Multi-Epoch participation"""
        self.participation_interval = participation_interval

        """ Matrix Factorization"""
        self.C = C
        if C_sens is None:
            self.C_sens = compute_sensitivity(
                self.C,
                participation_interval=self.participation_interval,
                num_epochs=len(C),  # TODO: Change this if C becomes between nodes
            )
        else:
            self.C_sens = C_sens
        C_tensor = torch.tensor(C).to(device=self.device)
        self.Cinv = torch.linalg.pinv(C_tensor)  # Used for correlation
        # Set very small values to zero
        self.Cinv[torch.abs(self.Cinv) < 1e-15] = 0
        # print(self.Cinv)

        # Calculate the total number of trainable parameters
        self.num_trainable_params = sum(
            p.numel()
            for group in self.param_groups
            for p in group["params"]
            if p.requires_grad
        )

        # Generate all the independent noises for each trainable parameter at the beginning.
        # TODO: optimize this in terms of memory, but you'd need a banded assumption or something along those line.

        self.noise_variance = (
            self.l2_norm_clip * self.C_sens * self.noise_multiplier / self.batch_size
        )  # TODO: Check the value for the noise variance

        self.noises = torch.normal(
            0,
            std=self.noise_variance,
            size=(self.Cinv.shape[-1], self.num_trainable_params),
            device=device,
        )
        if self.id == 0:
            print(f"Noise variance: {self.noise_variance}")

        param_dtype = next(
            p.dtype
            for group in self.param_groups
            for p in group["params"]
            if p.requires_grad
        )
        self.noises = self.noises.to(param_dtype)
        self.Cinv = self.Cinv.to(param_dtype)
        self.noises = torch.matmul(
            self.Cinv, self.noises
        )  # the actual correlated noises to add
        self.noise_index = 0

        for group in self.param_groups:
            group["accum_grads"] = [
                (
                    torch.zeros_like(param.data, device=device)
                    if param.requires_grad
                    else None
                )
                for param in group["params"]
            ]

    def generate_noise(self):
        """Add noise"""

        noises_to_return = self.noises[self.noise_index]
        self.noise_index += 1
        return noises_to_return

    def zero_microbatch_grad(self):
        for group in self.param_groups:
            for param in group["params"]:
                if param.grad is not None:
                    param.grad.zero_()

                param.grad_sample = None

    def microbatch_step(self):
        total_norm = None

        # Calculate the total L2 norm of gradients
        for group in self.param_groups:
            for param in group["params"]:
                if param.requires_grad and param.grad is not None:
                    grad_samples = param.grad_sample

                    batch_size = grad_samples.shape[0]

                    if total_norm is None:
                        total_norm = torch.zeros(batch_size, device=self.device)

                    # Compute per-sample norm
                    total_norm += (
                        torch.norm(grad_samples.view(batch_size, -1), dim=1) ** 2
                    )

        assert total_norm is not None
        total_norm = total_norm**0.5
        clip_coef = (self.l2_norm_clip / (total_norm + 1e-6)).clamp(max=1.0)

        for group in self.param_groups:
            for param, accum_grad in zip(group["params"], group["accum_grads"]):
                if not param.requires_grad:
                    continue

                grad_samples = param.grad_sample
                clipped_grads = grad_samples * clip_coef.view(
                    -1, *([1] * (grad_samples.dim() - 1))
                )

                # Average across samples and accumulate
                accum_grad.add_(clipped_grads.sum(dim=0))

    @torch.no_grad()
    def step(self, *args, **kwargs):
        """Update parameters based on accumulated gradients with added noise for privacy."""

        current_step_noise = self.generate_noise()
        noises_used = 0

        for group in self.param_groups:
            lr = group["lr"]

            for ind, param in enumerate(group["params"]):
                if param.requires_grad:
                    # Get state for this parameter
                    accum_grads = (
                        group["accum_grads"][ind].to(param.device) / self.batch_size
                    )

                    # Flatten accum_grads to 1D to match the linear noise sampling
                    num_params = accum_grads.numel()
                    noise1 = current_step_noise[
                        noises_used : noises_used + num_params
                    ].to(param.device)
                    noises_used += num_params

                    # Reshape noise to match accum_grads shape
                    noise1 = noise1.view_as(accum_grads)

                    # noisy_grad = accum_grads
                    noisy_grad = accum_grads + noise1

                    # Update parameters without weight decay or momentum
                    param.data = param.data - noisy_grad * lr

                    # Zero out the accumulated gradients after each minibatch step
                    group["accum_grads"][ind].zero_()
        assert noises_used == self.num_trainable_params  # Ensure we used ALL the noises
