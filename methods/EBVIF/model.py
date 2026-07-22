import torch
import torch.nn as nn
import numpy
import random

def transform(data, msfa_size=4, spatial_ratio=2):
              
    if numpy.random.uniform() < 0.5:
        rn = random.randint(1, 3)
        data = torch.rot90(data, rn, [2, 3])

          
    if numpy.random.uniform() < 0.5:
        r = numpy.random.uniform()
        if r < 0.25:
            data = torch.flip(data, [2])
        elif r < 0.5:
            data = torch.flip(data, [3])
        elif r < 0.75:
            data = torch.flip(data, [2])
            data = torch.flip(data, [3])
        else:
            data = torch.flip(data, [3])
            data = torch.flip(data, [2])

           
    if numpy.random.uniform() < 0.5:
        i = random.randint(0, msfa_size * spatial_ratio - 1)
        j = random.randint(0, msfa_size * spatial_ratio - 1)
        data = torch.roll(data, (i, j), (2, 3))

    return data

class CNNs_v1(nn.Module):
    def __init__(self, Cin, Cout):
        super(CNNs_v1, self).__init__()
        self.conv1 = nn.Conv2d(Cin, 64, 3, 1, 1)
        self.conv2 = nn.Conv2d(64, 64, 3, 1, 1)
        self.conv3 = nn.Conv2d(64, Cout, 3, 1, 1)
        self.relu = nn.ReLU()

    def forward(self, mosaic, pan):
        mosaic_reshape = torch.nn.functional.pixel_unshuffle(mosaic, downscale_factor=4)
        mosaic_interpolate = torch.nn.functional.interpolate(mosaic_reshape, scale_factor=8, mode="bicubic")
        x = torch.cat((mosaic_interpolate, pan), 1)
        y = self.conv3(self.relu(self.conv2(self.relu(self.conv1(x)))))
        return y

class Degrade_SRF(nn.Module):
    def __init__(self, Cin, Cout):
        super(Degrade_SRF, self).__init__()
        self.spec_res = nn.Conv2d(Cin, Cout, 1, 1, 0, bias=False)
        self.spec_res.weight.data = torch.ones_like(self.spec_res.weight)
        self.spec_res.weight.data /= self.spec_res.weight.data.sum()

    def forward(self, x):
        y = self.spec_res(x)
        return y

class Degrade_BDM(nn.Module):
    def __init__(self, ksize=13):
        super(Degrade_BDM, self).__init__()

    def forward(self, x, msfa_kernel):
        z = torch.nn.functional.conv2d(
            x, msfa_kernel, bias=None, stride=msfa_kernel.shape[2], groups=x.shape[1]
        )
        return z

class FuseNet(nn.Module):
    def __init__(self, args):
        super(FuseNet, self).__init__()
        self.mu_sigma_f_and_g = CNNs_v1(args.num_bands + 1, args.num_bands)

        self.degrade_srf = Degrade_SRF(args.num_bands, 1)
        self.degrade_bdm = Degrade_BDM(ksize=13)

    def forward_for_inference(self, z, y, msfa_kernel):
                    
        est_f = self.mu_sigma_f_and_g(z, y)

                    
        degrade_srf = self.degrade_srf(est_f)
        sparsity_y = y - degrade_srf

        degrade_bdm = self.degrade_bdm(est_f, msfa_kernel)
        degrade_bdm = torch.nn.functional.pixel_shuffle(degrade_bdm, 4)
        sparsity_z = z - degrade_bdm

        est_g = self.mu_sigma_f_and_g(sparsity_z, sparsity_y)

        return est_f, est_g

    def forward_for_train(self, z, y, msfa_kernel):
                    
        est_f = self.mu_sigma_f_and_g(z, y)

                    
        degrade_srf = self.degrade_srf(est_f)
        sparsity_y = y - degrade_srf

        degrade_bdm = self.degrade_bdm(est_f, msfa_kernel)
        degrade_bdm = torch.nn.functional.pixel_shuffle(degrade_bdm, 4)
        sparsity_z = z - degrade_bdm

        est_g = self.mu_sigma_f_and_g(sparsity_z, sparsity_y)

                 
        degrade_srf = self.degrade_srf(est_f + est_g)
        degrade_bdm = torch.nn.functional.pixel_shuffle(
            self.degrade_bdm(est_f + est_g, msfa_kernel), 4
        )

                          
        loss_y = 0.5 * ((y - degrade_srf) ** 2).mean()
        loss_z = 0.5 * ((z - degrade_bdm) ** 2).mean()
        loss = loss_y + loss_z

            
        fused = transform(est_f + est_g, msfa_size=4, spatial_ratio=2).detach()
        degrade_y1 = self.degrade_srf(fused)
        degrade_z1 = self.degrade_bdm(fused, msfa_kernel)
        degrade_z1 = torch.nn.functional.pixel_shuffle(degrade_z1, 4)

                     
        est_f = self.mu_sigma_f_and_g(degrade_z1, degrade_y1)

                     
        degrade_srf = self.degrade_srf(est_f)
        sparsity_y = degrade_y1 - degrade_srf

        degrade_bdm = self.degrade_bdm(est_f, msfa_kernel)
        degrade_bdm = torch.nn.functional.pixel_shuffle(degrade_bdm, 4)
        sparsity_z = degrade_z1 - degrade_bdm

        est_g = self.mu_sigma_f_and_g(sparsity_z, sparsity_y)

        loss += 0.5 * ((fused - est_f - est_g) ** 2).mean()

        return loss

    def forward(self, z, y, msfa_kernel, inference_flag=False):
        if inference_flag:
            return self.forward_for_inference(z, y, msfa_kernel)
        else:
            return self.forward_for_train(z, y, msfa_kernel)
