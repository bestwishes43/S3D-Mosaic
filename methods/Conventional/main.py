import time
import torch
torch.manual_seed(42)
from torch.utils.data import Subset

from project_utils.dataloader import CAVE, NTIRE
from project_utils.metrics import calculate_psnr, calculate_sam, SSIM
from algorithms import *


def main():
    device_default = 'cuda' if torch.cuda.is_available() else 'cpu'
    dtype_default = torch.float32

    calculate_ssim = SSIM().to(device=device_default)
    pattern_size = [5, 5]; dataset = CAVE('../../datasets/CAVE/HSI'); test_set = Subset(dataset, [0, 1, 2, 3, 4, 5, 8, 16, 17, 21])
    dataloader = torch.utils.data.DataLoader(test_set, shuffle=False)

    C = dataset.data_shape[0]
    msfa = dataset.msfa.unsqueeze(0).to(device_default)
    mask = torch.concat([msfa == ch for ch in range(C)], dim=1).float()
    
    wb = WeightedBilinear2D(pattern_size).to(device_default).eval()
    itsd = ItSD(pattern_size).to(device_default).eval()
    ppid = PPID(pattern_size).to(device_default).eval()

    for method in ['wb', 'itsd', 'ppid']:
        model = eval(method)
        psnr_list = []
        ssim_list = []
        sam_list = []
        for idx, (raw, msi_gt) in enumerate(dataloader):

            raw = raw.to(device_default, dtype=dtype_default)
            msi_gt = msi_gt.to(device_default, dtype=dtype_default)
            with torch.no_grad():
                start_time = time.time()
                X = model(raw, mask)
                end_time = time.time()
            
            psnr = calculate_psnr(msi_gt, X).item()
            ssim = calculate_ssim(msi_gt, X).item()
            sam = calculate_sam(msi_gt, X).item()
            print(f"|{idx}|{psnr:.2f}|{sam:.3f}|{ssim:.3f}|{end_time - start_time:.2f}|")
            
            psnr_list.append(psnr)
            sam_list.append(sam)
            ssim_list.append(ssim)
        
        print(f"{method}:", torch.tensor(psnr_list).mean(), torch.tensor(ssim_list).mean(), torch.tensor(sam_list).mean())


if __name__ == '__main__':
    main()
