"""models/conv_autoencoder.py

Convolutional Autoencoder for log-mel spectrograms.

Input shape : [B, 1, n_mels, T]  →  default (B, 1, 96, 55)
Output shape: same as input.

To swap in a different architecture, create a new file under models/,
expose a class with the same encode / decode / forward interface, and
update the import in inference/detector.py.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvAutoencoder(nn.Module):
    """
    Design (tailored to 96 × 55 mel spectrograms):
      Encoder  – 3 stride-2 conv blocks:  [96,55]→[48,28]→[24,14]→[12,7]
      Bottleneck – flat FC latent vector of size `latent_dim`.
      Decoder  – 3 transposed-conv blocks: [12,7]→[24,14]→[48,28]→[96,56]
               – center-crop to restore exact [96,55].
    No output activation (log-mel values can be negative).
    """

    def __init__(self, latent_dim: int = 128):
        super().__init__()
        self.latent_dim = latent_dim
        self.target_h = 96
        self.target_w = 55

        # ── Encoder ──────────────────────────────────────────
        self.enc = nn.Sequential(
            # [B, 1, 96, 55] → [B, 16, 48, 28]
            nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            # → [B, 32, 24, 14]
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            # → [B, 64, 12, 7]
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            # extra capacity at same spatial res
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

        self.flatten = nn.Flatten()
        self.fc_enc  = nn.Linear(64 * 12 * 7, latent_dim)

        # ── Decoder ──────────────────────────────────────────
        self.fc_dec = nn.Sequential(
            nn.Linear(latent_dim, 64 * 12 * 7),
            nn.ReLU(inplace=True),
        )
        self.dec = nn.Sequential(
            nn.Unflatten(1, (64, 12, 7)),
            # [B,64,12,7] → [B,32,24,14]
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            # → [B,16,48,28]
            nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            # → [B,1,96,56]
            nn.ConvTranspose2d(16, 1, kernel_size=3, stride=2, padding=1, output_padding=1),
        )

    # ── spatial helpers ──────────────────────────────────────

    @staticmethod
    def _crop_or_pad(x: torch.Tensor, h: int, w: int) -> torch.Tensor:
        """Center-crop or zero-pad *x* to (h, w)."""
        _, _, cur_h, cur_w = x.shape

        # height
        if cur_h > h:
            top = (cur_h - h) // 2
            x = x[:, :, top : top + h, :]
        elif cur_h < h:
            pt = (h - cur_h) // 2
            pb = h - cur_h - pt
            x = F.pad(x, (0, 0, pt, pb))

        # width
        _, _, _, cur_w = x.shape
        if cur_w > w:
            left = (cur_w - w) // 2
            x = x[:, :, :, left : left + w]
        elif cur_w < w:
            pl = (w - cur_w) // 2
            pr = w - cur_w - pl
            x = F.pad(x, (pl, pr, 0, 0))

        return x

    # ── forward pass ─────────────────────────────────────────

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        h = self.enc(x)           # [B, 64, 12, 7]
        h = self.flatten(h)       # [B, 64*12*7]
        return self.fc_enc(h)     # [B, latent_dim]

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        h     = self.fc_dec(z)    # [B, 64*12*7]
        x_hat = self.dec(h)       # [B, 1, 96, 56]
        return self._crop_or_pad(x_hat, self.target_h, self.target_w)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decode(self.encode(x))
