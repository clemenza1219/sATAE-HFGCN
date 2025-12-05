import torch.nn as nn

class Autoencoder(nn.Module):
    def __init__(self, dim, dim_coef):
        super(Autoencoder, self).__init__()
        # self.dim = dim
        self.encoder = nn.Sequential(
            nn.Linear(dim, dim // dim_coef),
            nn.Tanh(),
            nn.Linear(dim // dim_coef, dim // (dim_coef ** int(2))),
            nn.Tanh(),
            nn.Linear(dim // (dim_coef ** int(2)), dim // (dim_coef ** int(3))),
            nn.Tanh(),
            nn.Linear(dim // (dim_coef ** int(3)), dim // (dim_coef ** int(4))),
            nn.Tanh(),
            nn.Linear(dim // (dim_coef ** int(4)), dim // (dim_coef ** int(5))),
            nn.Tanh(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(dim // (dim_coef ** int(5)), dim // (dim_coef ** int(4))),
            nn.Tanh(),
            nn.Linear(dim // (dim_coef ** int(4)), dim // (dim_coef ** int(3))),
            nn.Tanh(),
            nn.Linear(dim // (dim_coef ** int(3)), dim // (dim_coef ** int(2))),
            nn.Tanh(),
            nn.Linear( dim // (dim_coef ** int(2)), dim // dim_coef),
            nn.Tanh(),
            nn.Linear( dim // dim_coef, dim),
            nn.Tanh(),

        )
    def forward(self, x):
        # print(x.shape)
        site, state, bands, time = x.shape
        encoder_out = self.encoder(x)
        encoded_output = encoder_out  

        decoder_out = self.decoder(encoder_out)
        return decoder_out, encoded_output


class Att_Autoencoder(nn.Module):
    def __init__(self, dim, dim_coef):
        super(Att_Autoencoder, self).__init__()
        # self.dim = dim
        self.en_part1 = nn.Sequential(
            nn.Linear(dim, dim // dim_coef),
            nn.Tanh(),
            nn.Linear(dim // dim_coef, dim // (dim_coef ** int(2))),
            nn.Tanh()
        )
        self.en_att1 = nn.Sequential(
            nn.AvgPool1d(dim_coef ** int(2)),
            nn.Linear(dim // (dim_coef ** int(2)), dim // (dim_coef ** int(2))),
            nn.Tanh()
        )

        self.en_part2 = nn.Sequential(
            nn.Linear(dim // (dim_coef ** int(2)), dim // (dim_coef ** int(3))),
            nn.Tanh(),
            nn.Linear(dim // (dim_coef ** int(3)), dim // (dim_coef ** int(4))),
            nn.Tanh()
        )

        self.en_att2 = nn.Sequential(
            nn.AvgPool1d(dim_coef ** int(2)),
            nn.Linear(dim // (dim_coef ** int(4)), dim // (dim_coef ** int(4))),
            nn.Tanh()
        )
        self.mid_linear = nn.Linear(dim // (dim_coef ** int(4)), 2 ** int(3))

        self.decoder = nn.Sequential(

            nn.Linear(2 ** int(3), dim // (dim_coef ** int(4))),
            nn.Tanh(),
            nn.Linear(dim // (dim_coef ** int(4)), dim // (dim_coef ** int(3))),
            nn.Tanh(),
            nn.Linear(dim // (dim_coef ** int(3)), dim // (dim_coef ** int(2))),
            nn.Tanh(),
            nn.Linear(dim // (dim_coef ** int(2)), dim // dim_coef),
            nn.Tanh(),
            nn.Linear(dim // dim_coef, dim),
            nn.Tanh(),
        )
    def forward(self, x):
        # print(x.shape)

        num_site, state, bands, time = x.shape  # torch.Size([16, 3, 6, 30000])
        # block 1
        en_part1_out = self.en_part1(x) # torch.Size([16, 3, 6, 1200])
        pool_input = x.view(num_site, -1, time)
        pool_att1 = self.en_att1(pool_input)
        pool_out = pool_att1.view(num_site, state, bands, -1)
        att1 = en_part1_out * pool_out # torch.Size([16, 3, 6, 1200])

        # block 2
        en_part2_out = self.en_part2(att1) # torch.Size([16, 3, 6, 48])
        pool_input2 = att1.view(num_site, state * bands, -1)
        pool_att2 = self.en_att2(pool_input2)
        pool_out2 = pool_att2.view(num_site, state, bands, -1)
        att2 = en_part2_out * pool_out2  # torch.Size([16, 3, 6, 48])

        encoder_out = self.mid_linear(att2)  # torch.Size([16, 3, 6, 16])
        encoded_output = encoder_out
        # print(encoded_output.shape)
        decoder_out = self.decoder(encoder_out)  # torch.Size([16, 3, 6, 30000])
        return decoder_out, encoded_output