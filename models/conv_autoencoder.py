# src/models/conv_autoencoder.py
import torch
import torch.nn as nn


class ConvAutoencoder(nn.Module):
    """
    Convolutional Autoencoder for log-mel spectrograms.

    Your data (from your pipeline):
      - .npy shape: (n_mels, T) = (64, 61)  (with sr=16000, win=2s, n_fft=1024, hop=512, center=False)
      - Dataset returns: [B, 1, 64, 61]

    Design choices:
      - No Sigmoid at the output (because your spectrograms are per-sample normalized and contain negatives).
      - Uses a compact fully-connected bottleneck (Flatten + Linear) to capture global TF patterns.
      - Decoder reconstructs, then we center-crop/pad to exactly (64, 61) to avoid shape issues.
    """

    def __init__(self, latent_dim: int = 32):
        super().__init__()
        self.latent_dim = latent_dim

        # We will reconstruct to a "safe" size and then crop to (64, 61).
        # With the strides below, the decoder naturally gives width 64, so we'll crop width -> 61.
        self.target_h = 64
        self.target_w = 61

        # -------- Encoder --------
        # Input: [B, 1, 64, 61]
        self.enc = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1),   # -> [B,16,32,31]
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),

            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),  # -> [B,32,16,16]
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),  # -> [B,64,8,8]
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

        # Bottleneck
        self.flatten = nn.Flatten()
        self.fc_enc = nn.Linear(64 * 8 * 8, latent_dim)

        # -------- Decoder --------
        self.fc_dec = nn.Sequential(
            nn.Linear(latent_dim, 64 * 8 * 8),
            nn.ReLU(inplace=True),
        )

        self.dec = nn.Sequential(
            nn.Unflatten(1, (64, 8, 8)),

            nn.ConvTranspose2d(
                64, 32, kernel_size=3, stride=2, padding=1, output_padding=1
            ),  # -> [B,32,16,16]
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(
                32, 16, kernel_size=3, stride=2, padding=1, output_padding=1
            ),  # -> [B,16,32,32]
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(
                16, 1, kernel_size=3, stride=2, padding=1, output_padding=1
            ),  # -> [B,1,64,64]
            # NO activation here (linear output)
        )

    @staticmethod
    def _center_crop_or_pad(x: torch.Tensor, target_h: int, target_w: int) -> torch.Tensor:
        """
        x: [B, C, H, W]
        Center-crop if too big, pad with zeros if too small.
        """
        b, c, h, w = x.shape

        # --- Crop or pad H ---
        if h > target_h:
            top = (h - target_h) // 2
            x = x[:, :, top:top + target_h, :]
        elif h < target_h:
            pad_top = (target_h - h) // 2
            pad_bottom = target_h - h - pad_top
            x = torch.nn.functional.pad(x, (0, 0, pad_top, pad_bottom))

        # --- Crop or pad W ---
        b, c, h, w = x.shape
        if w > target_w:
            left = (w - target_w) // 2
            x = x[:, :, :, left:left + target_w]
        elif w < target_w:
            pad_left = (target_w - w) // 2
            pad_right = target_w - w - pad_left
            x = torch.nn.functional.pad(x, (pad_left, pad_right, 0, 0))

        return x

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        h = self.enc(x)
        h = self.flatten(h)
        z = self.fc_enc(h)
        return z

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        h = self.fc_dec(z)
        x_hat = self.dec(h)
        x_hat = self._center_crop_or_pad(x_hat, self.target_h, self.target_w)
        return x_hat

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encode(x)
        x_hat = self.decode(z)
        return x_hat
