import torch.nn as nn
import torch


class Autoencoder(nn.Module):
    def __init__(self, num_channels=1, task=""):
        super().__init__()
        self.num_channels = num_channels
        self.task = task

        self.encoder = nn.Sequential(
            nn.Conv2d(num_channels, 16, 3, stride=2, padding=1),  # 28x28 -> 14x14
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.2),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),  # 14x14 -> 7x7
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2),
            nn.Conv2d(32, 64, 7, stride=1, padding=0),  # 7x7 -> 1x1
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            nn.Flatten(),
            nn.Linear(1*1*64, 32),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(0.2),
            nn.Linear(32, 16),
        )


        self.decoder = nn.Sequential(
            nn.Linear(16, 32),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(0.2),
            nn.Linear(32, 1*1*64),
            nn.BatchNorm1d(1*1*64),
            nn.LeakyReLU(0.2),
            nn.Unflatten(1, (64, 1, 1)),
            nn.ConvTranspose2d(64, 32, 7),  # 1x1 -> 7x7
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1, output_padding=1),  # 7x7 -> 14x14
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(16, num_channels, 3, stride=2, padding=1, output_padding=1),  # 14x14 -> 28x28
            nn.Sigmoid(),
        )

    # def forward(self, x):
    #     encoded = self.encoder(x)
    #     decoded = self.decoder(encoded)
    #     return decoded

    def forward(self, x):
        if x.shape[1] > 1:
            # if color, run each channel through separately
            # then stack them into rgb
            encoded_list = []
            decoded_list = []

            for channel in range(x.shape[1]):
                gray_im = x[:, channel, :, :].reshape(-1, 1, 28, 28)

                encoded = self.encoder(gray_im)
                decoded = self.decoder(encoded)

                encoded_list.append(encoded)
                decoded_list.append(decoded)

            encoded = torch.stack(encoded_list, dim=-1)
            decoded = torch.stack(decoded_list, dim=-1)
            # print(f"encoded shape, {encoded.shape}")

            return decoded


        else: # monochrome
            encoded = self.encoder(x)
            decoded = self.decoder(encoded)
            return decoded

    def generate(self, x):
        pass