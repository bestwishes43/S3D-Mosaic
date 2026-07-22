import sys, os
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')); sys.path.insert(0, _ROOT)

import torch
torch.manual_seed(42)
from torch.utils.data import Subset
import argparse

from model import Network
from project_utils.quality_index import calc_psnr, calc_ssim, calc_sam, calc_ergas

from project_utils.dataloader import CAVEDatasetDC


def main():
    parser = argparse.ArgumentParser(description='Test')
    parser.add_argument('--num_bands', type=int, default=16, help='Number of bands of a MS image.')
    args = parser.parse_args()

    device_default = 'cuda' if torch.cuda.is_available() else 'cpu'
    dtype_default = torch.float32
    
    ckpt_path = "./checkpoint/best_958.pth"
    dataset = CAVEDatasetDC(os.path.join(_ROOT, 'datasets/CAVE/HSI')); test_set = Subset(dataset, range(20, 32))
    dataloader = torch.utils.data.DataLoader(test_set, batch_size=1, shuffle=False)

    fuse_net = Network(args)
    fuse_net.load_state_dict(torch.load(ckpt_path, map_location='cpu')["net"], strict=False)
    fuse_net = fuse_net.eval().to(device_default)

    psnr_list = []
    sam_list = []
    ssim_list = []
    ergas_list = []
    for idx, (raw, pan, msi_gt) in enumerate(dataloader):
        
        raw = raw.to(device_default, dtype=dtype_default)
        pan = pan.to(device_default, dtype=dtype_default)
        msi_gt = msi_gt.to(device_default, dtype=dtype_default)

        with torch.no_grad():
            X = fuse_net(raw, pan).detach()

        msi_gt = msi_gt.cpu()
        X = X.cpu()

        psnr = calc_psnr(msi_gt, X).item()
        sam = calc_sam(msi_gt, X).item()
        ssim = calc_ssim(msi_gt, X).item()
        ergas = calc_ergas(msi_gt, X).item()
        
        print(f"|{idx}|{psnr:.2f}|{sam:.3f}|{ssim:.3f}|{ergas:.3f}|")
        psnr_list.append(psnr)
        sam_list.append(sam)
        ssim_list.append(ssim)
        ergas_list.append(ergas)

    print(torch.tensor(psnr_list).mean(), torch.tensor(sam_list).mean(), torch.tensor(ssim_list).mean(), torch.tensor(ergas_list).mean())
    
if __name__ == '__main__':
    main()