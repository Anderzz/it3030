import torch
import torch.nn as nn

latent_dim = 16 # latent dimension for sampling

class VAE(nn.Module):
    def __init__(self):
        super().__init__()

            # encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=8, kernel_size=3, stride=2, padding=1),   # 28 -> 14
            nn.ReLU(),
            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, stride=2, padding=1),  # 14 -> 7 
            nn.ReLU(),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=2, padding=1), # 7  -> 4
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=2, padding=1), # 4  -> 2
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64*2*2, 128)
        )
        

        # learn the representations
        # self.fc1 = nn.Linear(64, 128)
        self.fc_mu = nn.Linear(128, latent_dim)
        self.fc_log_var = nn.Linear(128, latent_dim)
        # self.fc2 = nn.Linear(latent_dim, 64)


        # decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Unflatten(1, (64, 2, 2)),
            nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=3, stride=2, padding=1, output_padding=1), # 3 -> 5
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=32, out_channels=16, kernel_size=3, stride=2, padding=1), # 5 -> 9
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=16, out_channels=8,  kernel_size=3, stride=2, padding=1, output_padding=1), # 9 -> 17
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=8, out_channels=1,  kernel_size=3, stride=2, padding=1, output_padding=1), # 17 -> 28
            nn.Sigmoid(),
        )


    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5*log_var)
        eps = torch.randn_like(std)
        sample = mu + (eps * std)
        return sample
    
    def forward(self, x):
        # encode
        x = self.encoder(x)

        mu = self.fc_mu(x)
        log_var = self.fc_log_var(x)


        # reparam to get latent vector
        z = self.reparameterize(mu, log_var)

        reconstruction = self.decoder(z)
        
        return reconstruction, mu, log_var