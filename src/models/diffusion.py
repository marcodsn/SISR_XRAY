import math

import torch
import torch.nn as nn


# https://arxiv.org/abs/2104.07636
class ConditionalUNet(nn.Module):
    def __init__(self, in_channels=2, out_channels=1, base_dim=128):
        # in_channels=2 because: 1 channel (noisy input) + 1 channel (LR condition)
        super().__init__()

        # Time embedding
        self.time_mlp = nn.Sequential(
            nn.Linear(1, base_dim),
            nn.SiLU(),
            nn.Linear(base_dim, base_dim),
        )

        # Initial Conv
        self.inc = nn.Sequential(
            nn.Conv2d(in_channels, base_dim, 3, padding=1),
            nn.SiLU(),
        )

        # Downsampling
        self.down1 = nn.Sequential(
            nn.Conv2d(base_dim, base_dim * 2, 4, 2, 1), nn.SiLU()
        )
        self.down2 = nn.Sequential(
            nn.Conv2d(base_dim * 2, base_dim * 4, 4, 2, 1), nn.SiLU()
        )

        # Middle
        self.mid = nn.Sequential(
            nn.Conv2d(base_dim * 4, base_dim * 4, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(base_dim * 4, base_dim * 4, 3, padding=1),
            nn.SiLU(),
        )

        # Upsampling
        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(base_dim * 4, base_dim * 2, 4, 2, 1), nn.SiLU()
        )
        self.up2 = nn.Sequential(
            nn.ConvTranspose2d(base_dim * 4, base_dim, 4, 2, 1), nn.SiLU()
        )

        # Output
        self.outc = nn.Conv2d(base_dim * 2, out_channels, 3, padding=1)

    def forward(self, x_t, y_0, t):
        # Time embedding
        t_emb = self.time_mlp(t.float().view(-1, 1))
        t_emb = t_emb.view(-1, t_emb.shape[1], 1, 1)

        # Conditioning: Concatenate Noisy Input (x_t) with Low Res (y_0)
        x = torch.cat([x_t, y_0], dim=1)

        x1 = self.inc(x)
        x2 = self.down1(x1 + t_emb)
        x3 = self.down2(x2)

        x_mid = self.mid(x3)

        x_up1 = self.up1(x_mid)
        # Skip connection 1
        x_up1 = torch.cat([x_up1, x2], dim=1)

        x_up2 = self.up2(x_up1)
        # Skip connection 2
        x_up2 = torch.cat([x_up2, x1], dim=1)

        out = self.outc(x_up2)
        return out


# https://arxiv.org/abs/2307.12348
class ResShiftSchedule:
    def __init__(self, num_steps=15, kappa=2.0, p=0.3, device="cpu"):
        self.T = num_steps
        self.kappa = kappa
        self.p = p
        self.device = device

        # Calculate eta values (shifting sequence)
        eta_1 = (0.001) ** 2
        eta_T = 0.999
        b0 = math.exp((1.0 / (2 * (self.T - 1))) * math.log(eta_T / eta_1))

        self.sqrt_eta = torch.zeros(self.T + 1)
        self.sqrt_eta[0] = 0
        self.sqrt_eta[1] = math.sqrt(eta_1)

        for t in range(2, self.T):
            term = ((t - 1) / (self.T - 1)) ** self.p
            beta_t = term * (self.T - 1)
            self.sqrt_eta[t] = self.sqrt_eta[1] * (b0**beta_t)

        self.sqrt_eta[self.T] = math.sqrt(eta_T)
        self.eta = self.sqrt_eta**2

        # Calculate alpha values
        self.alpha = torch.zeros(self.T + 1)
        self.alpha[1] = self.eta[1]
        for t in range(2, self.T + 1):
            self.alpha[t] = self.eta[t] - self.eta[t - 1]

        # Move to device
        self.eta = self.eta.to(self.device)
        self.alpha = self.alpha.to(self.device)
        self.sqrt_eta = self.sqrt_eta.to(self.device)

    def get_alpha(self, t):
        return self.alpha[t].view(-1, 1, 1, 1)

    def get_eta(self, t):
        return self.eta[t].view(-1, 1, 1, 1)


@torch.no_grad()
def p_sample_loop_resshift(model, y_0, schedule):
    """Inference: Generates HR image from LR image y_0 in T steps."""
    model.eval()
    device = y_0.device  # Detect device from input
    batch_size = y_0.shape[0]

    # Start from x_T (Approx LR image with noise)
    x_t = y_0.clone()

    for t in reversed(range(1, schedule.T + 1)):
        t_tensor = torch.full((batch_size,), t, device=device, dtype=torch.long)

        # Predict clean x_0
        predicted_x0 = model(x_t, y_0, t_tensor)

        # Calculate posterior parameters
        eta_t = schedule.get_eta(t)
        alpha_t = schedule.get_alpha(t)
        eta_prev = schedule.get_eta(t - 1) if t > 1 else torch.zeros_like(eta_t)

        if t == 1:
            mu = predicted_x0
            var = torch.zeros_like(mu)
        else:
            term1 = (eta_prev / eta_t) * x_t
            term2 = (alpha_t / eta_t) * predicted_x0
            mu = term1 + term2
            var = (schedule.kappa**2) * (eta_prev / eta_t) * alpha_t

        noise = torch.randn_like(x_t)
        x_t = mu + torch.sqrt(var) * noise

    return x_t
