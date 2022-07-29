
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from utils import gabor_kernel
from tqdm import tqdm
from torch.distributions import uniform
from torch.autograd import Variable

#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device='cpu'

class Identity(nn.Module):
    def forward(self, x):
        return x


class GFB(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=2):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = (kernel_size, kernel_size)
        self.stride = (stride, stride)
        self.padding = (kernel_size // 2, kernel_size // 2)


        self.thetaA = torch.zeros(self.out_channels)
        self.ratioA = torch.zeros(self.out_channels)
        self.sfA = torch.zeros(self.out_channels)
        self.sigxA = torch.zeros(self.out_channels)
        self.sigyA = torch.zeros(self.out_channels)
        self.weight = torch.zeros((out_channels, in_channels, kernel_size, kernel_size))

    def forward(self, x):
        return F.conv2d(x, self.weight, None, self.stride, self.padding)

    def initialize(self, sf, theta, sigx, sigy, phase, color_channels):
        #Generate 512 random ints between 0 and 2.
        #Each channel has the same weights?????
        #Within each channel each neuron have the same weights?
        #Gabor function?


        #random_channel = torch.randint(0, self.in_channels, (self.out_channels,)) #Rand from 0 to 2, 512 different
        for i in range(self.out_channels):
            #Gives the weights for the same unit per each channel
            self.weight[i, :] = gabor_kernel(frequency=sf[i], sigma_x=sigx[i], sigma_y=sigy[i],
                                                             theta=theta[i], offset=phase[i], ks=self.kernel_size[0])
            #self.orientations[i] = theta[i]
            self.sfA[i] = sf[i]
            self.thetaA[i] = theta[i]
            self.sigxA[i] = sigx[i]
            self.sigyA[i] = sigy[i]
            self.ratioA[i] = sigx[i]/sigy[i]

            #self.generalparam = CHOOSEPARAM[i]
        self.thetaA = nn.Parameter(self.thetaA, requires_grad=False)
        self.sfA = nn.Parameter(self.sfA, requires_grad=False)
        self.sigxA = nn.Parameter(self.sigxA, requires_grad=False)
        self.sigyA = nn.Parameter(self.sigyA, requires_grad=False)
        self.ratioA = nn.Parameter(self.ratioA, requires_grad=False)
        self.weight = nn.Parameter(self.weight, requires_grad=False)
        #self.generalparam = nn.Parameter(self.generalparam, requires_grad=False)

class VOneBlock(nn.Module):
    def __init__(self, sf, theta, sigx, sigy, phase,
                 k_exc=25, noise_mode=None, noise_scale=1, noise_level=1,
                 simple_channels=128, complex_channels=128, ksize=25, stride=2, input_size=64, color_channels = 1):
        super().__init__()
        self.in_channels = color_channels

        self.simple_channels = simple_channels
        self.complex_channels = complex_channels
        self.out_channels = simple_channels + complex_channels
        self.stride = stride
        self.input_size = input_size
        print(f'Stride is {self.stride}')
        print(f'In channels is {self.in_channels}')
        self.sf = sf
        self.theta = theta
        self.sigx = sigx
        self.sigy = sigy
        self.phase = phase
        self.k_exc = k_exc

        self.set_noise_mode(noise_mode, noise_scale, noise_level)
        self.fixed_noise = None
        self.color_channels = torch.randint(0, self.in_channels, (self.out_channels,))
        self.simple_conv_q0 = GFB(self.in_channels, self.out_channels, ksize, stride)
        self.simple_conv_q1 = GFB(self.in_channels, self.out_channels, ksize, stride)
        self.simple_conv_q0.initialize(sf=self.sf, theta=self.theta, sigx=self.sigx, sigy=self.sigy,
                                       phase=self.phase, color_channels=self.color_channels)
        self.simple_conv_q1.initialize(sf=self.sf, theta=self.theta, sigx=self.sigx, sigy=self.sigy,
                                       phase=self.phase + np.pi / 2, color_channels=self.color_channels)
        self.thetaA = self.simple_conv_q0.thetaA
        self.thetaA = nn.Parameter(self.thetaA, requires_grad=False)
        self.ratioA = self.simple_conv_q0.ratioA
        self.ratioA = nn.Parameter(self.ratioA, requires_grad=False)
        self.sfA = self.simple_conv_q0.sfA
        self.sfA = nn.Parameter(self.sfA, requires_grad=False)
        self.sigxA = self.simple_conv_q0.sigxA
        self.sigxA = nn.Parameter(self.sigxA, requires_grad=False)
        self.sigyA = self.simple_conv_q0.sigyA
        self.sigyA = nn.Parameter(self.sigyA, requires_grad=False)

        #self.generalparam = self.simple_conv_q0.generalparam
        #self.generalparam = nn.Parameter(self.generalparam, requires_grad=False)
        self.simple = nn.ReLU(inplace=True)
        self.complex = Identity()
        self.gabors = Identity()
        self.noise = nn.ReLU(inplace=True)
        self.output = Identity()

    def forward(self, x):
        # Gabor activations [Batch, out_channels, H/stride, W/stride]
        x = self.gabors_f(x)
        # Noise [Batch, out_channels, H/stride, W/stride]
        x = self.noise_f(x)
        # V1 Block output: (Batch, out_channels, H/stride, W/stride)
        x = self.output(x)
        return x

    def gabors_f(self, x):
        s_q0 = self.simple_conv_q0(x)
        s_q1 = self.simple_conv_q1(x)
        c = self.complex(torch.sqrt(s_q0[:, self.simple_channels:, :, :] ** 2 +
                                    s_q1[:, self.simple_channels:, :, :] ** 2) / np.sqrt(2))
        s = self.simple(s_q0[:, 0:self.simple_channels, :, :])
        return self.gabors(self.k_exc * torch.cat((s, c), 1))

    def noise_f(self, x):
        if self.noise_mode == 'neuronal':
            eps = 10e-5
            x *= self.noise_scale
            x += self.noise_level
            if self.fixed_noise is not None:
                x += self.fixed_noise * torch.sqrt(F.relu(x.clone()) + eps)
            else:
                x += torch.distributions.normal.Normal(torch.zeros_like(x), scale=1).rsample() * \
                     torch.sqrt(F.relu(x.clone()) + eps)
            x -= self.noise_level
            x /= self.noise_scale
        if self.noise_mode == 'gaussian':
            if self.fixed_noise is not None:
                x += self.fixed_noise * self.noise_scale
            else:
                x += torch.distributions.normal.Normal(torch.zeros_like(x), scale=1).rsample() * self.noise_scale
        return self.noise(x)

    def set_noise_mode(self, noise_mode=None, noise_scale=1, noise_level=1):
        self.noise_mode = noise_mode
        self.noise_scale = noise_scale
        self.noise_level = noise_level

    def fix_noise(self, batch_size=256, seed=None):
        noise_mean = torch.zeros(batch_size, self.out_channels, int(self.input_size/self.stride),
                                 int(self.input_size/self.stride))
        if seed:
            torch.manual_seed(seed)
        if self.noise_mode:
            self.fixed_noise = torch.distributions.normal.Normal(noise_mean, scale=1).rsample().to(device)

    def unfix_noise(self):
        self.fixed_noise = None

import os

class DivisiveNormBlock(nn.Module):

    def __init__(self, channel_num = 512, size = 32, ksizeDN = 12, use_full_image_net = 0, restore_path=None, map_location=None):
        super().__init__()

        #Basic model parameters
        self.channel_num = channel_num
        self.size = size
        self.ksizeDN = ksizeDN
        self.use_full_image_net = use_full_image_net

        #If using full image net, gaussian parameters are loaded from epoch file
        if use_full_image_net == 1:

            #Not all that important, just loads model parameters and shortens their names
            state = torch.load(os.path.join(restore_path),
                               map_location=map_location)

            from collections import OrderedDict
            new_state_dict = OrderedDict()

            for k, v in state['net'].items():
                    name = k[7:]  # remove `module.`
                    new_state_dict[name] = v

            #Load all model parameters, calculate gaussian
            self.thetaD = torch.nn.Parameter(new_state_dict['dn_block.thetaD'], requires_grad=False)
            self.p = torch.nn.Parameter(new_state_dict['dn_block.p'], requires_grad=False)
            self.sig = torch.nn.Parameter(new_state_dict['dn_block.sig'], requires_grad=False)
            self.a = torch.nn.Parameter(new_state_dict['dn_block.a'], requires_grad=False)
            self.x = torch.linspace(-self.ksizeDN, self.ksizeDN, self.ksizeDN * 2 + 1)
            self.y = torch.linspace(-self.ksizeDN, self.ksizeDN, self.ksizeDN * 2 + 1)
            self.xv, self.yv = torch.meshgrid(self.x, self.y)
            self.xv = nn.Parameter(torch.tensor(self.xv.expand(self.channel_num, self.channel_num, self.ksizeDN * 2 + 1,
                                                  self.ksizeDN * 2 + 1), device=self.thetaD.device).clone(), requires_grad=False)
            self.yv = nn.Parameter(torch.tensor(self.yv.expand(self.channel_num, self.channel_num, self.ksizeDN * 2 + 1,
                                                  self.ksizeDN * 2 + 1), device=self.thetaD.device).clone(), requires_grad=False)

            xrot = self.xv * torch.cos(self.thetaD) + self.yv * torch.sin(self.thetaD)
            yrot = -self.xv * torch.sin(self.thetaD) + self.yv * torch.cos(self.thetaD)
            gaussian_bank = (abs(self.a) /
                             (2 * torch.pi * self.p * self.sig)) * \
                            torch.exp(-0.5 * ((((xrot) ** 2) / self.p ** 2) +
                                              (((yrot) ** 2) / self.sig ** 2)))
            self.gaussian_bank = torch.relu(gaussian_bank)

        else:
            #Initialize model parameters. Gaussian is calculated during each forward of batches
            grad = True
            self.thetaD = torch.nn.Parameter(uniform.Uniform(0, 3.1415926535).sample([self.channel_num, self.channel_num,1,1]),
                                    requires_grad=grad)
            self.p = torch.nn.Parameter(uniform.Uniform(2, 6).sample([self.channel_num, self.channel_num,1,1]),
                                    requires_grad=grad)
            self.sig = torch.nn.Parameter(uniform.Uniform(2, 6).sample([self.channel_num, self.channel_num,1,1]),
                                      requires_grad=grad)
            self.a = torch.nn.Parameter(
            torch.abs(torch.randn(self.channel_num, self.channel_num,1,1, requires_grad=grad)))

            self.x = torch.linspace(-self.ksizeDN, self.ksizeDN, self.ksizeDN * 2 + 1)
            self.y = torch.linspace(-self.ksizeDN, self.ksizeDN, self.ksizeDN * 2 + 1)
            self.xv, self.yv = torch.meshgrid(self.x, self.y)
            self.xv = nn.Parameter(self.xv.clone().expand(self.channel_num, self.channel_num, self.ksizeDN * 2 + 1,
                                              self.ksizeDN * 2 + 1).clone(), requires_grad=False)
            self.yv = nn.Parameter(self.yv.clone().expand(self.channel_num, self.channel_num, self.ksizeDN * 2 + 1,
                                              self.ksizeDN * 2 + 1).clone(), requires_grad=False)
    def forward(self, x):

        x = self.dn_f(x)

        return x

    def dn_f(self, x):

        #Gaussian will be recalculated each time to correspond to learned parameters when using Tiny IN
        if self.use_full_image_net != 1:
            xrot = self.xv * torch.cos(self.thetaD) + self.yv * torch.sin(self.thetaD)
            yrot = -self.xv * torch.sin(self.thetaD) + self.yv * torch.cos(self.thetaD)
            gaussian_bank = (abs(self.a) /
                         (2 * torch.pi * self.p * self.sig)) * \
                        torch.exp(-0.5 * ((((xrot) ** 2) / self.p ** 2) +
                                          (((yrot) ** 2) / self.sig ** 2)))
            gaussian_bank = torch.relu(gaussian_bank)

        #Gaussian will just be picked from initially calculated gaussian for full IN
        else:
            gaussian_bank = torch.tensor(self.gaussian_bank, device=x.device)

        batch_size = x.shape[0]
        bias = torch.ones(1, self.channel_num, 1, 1, device=x.device)
        normalized_channels = torch.zeros((batch_size, self.channel_num, self.size, self.size), device=x.device)
        for b in range(batch_size):
            x_conv = torch.reshape(x[b].clone(), (1, self.channel_num, self.size, self.size))
            p = int((self.ksizeDN * 2) / 2)
            under_sum = F.conv2d(x_conv, gaussian_bank, stride=1, padding=p)
            normalized_channels[b] = x[b] / (bias + under_sum)
        return normalized_channels



'''
    def conv_gauss(self, x_conv, gauss_conv):
        x_conv = torch.reshape(x_conv, (1, self.channel_num, self.size, self.size))
        p = int((self.ksizeDN * 2) / 2)
        output = F.conv2d(x_conv, gauss_conv, stride=1, padding=p)
        return output

    def get_gaussian(self, cc, oc):

        xrot = (self.xv * torch.cos(self.thetaD[cc, oc]) + self.yv * torch.sin(self.thetaD[cc, oc]))
        yrot = (-self.xv * torch.sin(self.thetaD[cc, oc]) + self.yv * torch.cos(self.thetaD[cc, oc]))
        g_kernel = torch.tensor((abs(self.a[cc, oc]) /
                                 (2 * torch.pi * self.p[cc, oc] * self.sig[cc, oc])) * \
                                torch.exp(-0.5 * ((((xrot) ** 2) / self.p[cc, oc] ** 2) +
                                                  (((yrot) ** 2) / self.sig[cc, oc] ** 2))))

        return g_kernel


        xrot = (self.xv * torch.cos(self.thetaD[cc, oc]) + self.yv * torch.sin(self.thetaD[cc, oc]))
        yrot = (-self.xv * torch.sin(self.thetaD[cc, oc]) + self.yv * torch.cos(self.thetaD[cc, oc]))
        g_kernel = torch.tensor((abs(self.a[cc, oc]) /
                                 (2 * torch.pi * self.p[cc, oc] * self.sig[cc, oc])) * \
                                torch.exp(-0.5 * ((((xrot) ** 2) / self.p[cc, oc] ** 2) +
                                                  (((yrot) ** 2) / self.sig[cc, oc] ** 2))))

'''