import torch.nn as nn
import torch.nn.init as init

class Ae(nn.Module):
    def __init__(self, n_input, n_z):
        super(Ae, self).__init__()
        # 编码器
        self.encoder = nn.Sequential()
        self.encoder.add_module("cov01", nn.Conv2d(in_channels=n_input, out_channels=64, kernel_size=[1, 1], stride=1,
                                                   padding='same'))
        self.encoder.add_module("bn01", nn.BatchNorm2d(64))
        self.encoder.add_module("relu01", nn.LeakyReLU(negative_slope=0.01))


        self.encoder.add_module("cov02", nn.Conv2d(in_channels=64, out_channels=64, kernel_size=[3, 3], stride=1,
                                                   padding='same'))
        self.encoder.add_module("bn02", nn.BatchNorm2d(64))
        self.encoder.add_module("relu02", nn.LeakyReLU(negative_slope=0.01))

        self.encoder.add_module("cov03", nn.Conv2d(in_channels=64, out_channels=n_z, kernel_size=[3, 3], stride=1,
                                                   padding='same'))
        self.encoder.add_module("bn03", nn.BatchNorm2d(n_z))
        self.encoder.add_module("relu03", nn.LeakyReLU(negative_slope=0.01))

        # 解码器
        self.decoder = nn.Sequential()

        self.decoder.add_module("tr01",
                                nn.ConvTranspose2d(in_channels=n_z, out_channels=64, kernel_size=[3, 3], stride=1,
                                                   padding=1))
        self.decoder.add_module('rbn1', nn.BatchNorm2d(64))
        self.decoder.add_module("rre01", nn.LeakyReLU(negative_slope=0.01))

        self.decoder.add_module("tr02",
                                nn.ConvTranspose2d(in_channels=64, out_channels=64, kernel_size=[3, 3], stride=1,
                                                   padding=1))
        self.decoder.add_module('rbn2', nn.BatchNorm2d(64))
        self.decoder.add_module("rre02", nn.LeakyReLU(negative_slope=0.01))

        self.decoder.add_module("tr03",
                                nn.ConvTranspose2d(in_channels=64, out_channels=n_input, kernel_size=[1, 1], stride=1,
                                                   padding=0))



    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)


    def forward(self, x):
        h = self.encoder(x)
        x_rescon = self.decoder(h)
        return x_rescon, h



