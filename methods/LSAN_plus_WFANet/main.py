import torch
torch.manual_seed(42)
from torch.utils.data import Subset
import argparse

from model import Mpattern_opt, HWViT
import utils
from project_utils.quality_index import calc_psnr, calc_ssim, calc_sam, calc_ergas

from project_utils.dataloader import CAVEDatasetDC


def load_checkpoint_compat(path, map_location="cpu"):
    # Prefer safer loading on new PyTorch, but keep compatibility with old versions.
    try:
        return torch.load(path, map_location=map_location, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=map_location)

def input_matrix_wpn(inH, inW, msfa_size):

    h_offset_coord = torch.zeros(inH, inW, 1)
    w_offset_coord = torch.zeros(inH, inW, 1)
    for i in range(0,msfa_size):
        h_offset_coord[i::msfa_size, :, 0] = (i+1)/msfa_size
        w_offset_coord[:, i::msfa_size, 0] = (i+1)/msfa_size

    pos_mat = torch.cat((h_offset_coord, w_offset_coord), 2)
    pos_mat = pos_mat.contiguous().view(1, -1, 2)
    return pos_mat

def main():
    parser = argparse.ArgumentParser(description='Test')
    parser.add_argument('--spatial_ratio', type=int, default=2, help='Ratio of spatial resolutions between MS and PAN')
    parser.add_argument('--msfa_size', type=int, default=4, help='Size of MSFA')
    args = parser.parse_args()

    device_default = 'cuda' if torch.cuda.is_available() else 'cpu'
    dtype_default = torch.float32
    
    demosaic_ckpt_path = "./checkpoint/best_994.pth"
    pan_shap_ckpt_path = "./checkpoint/best_46.pth"

    dataset = CAVEDatasetDC('../../datasets/CAVE/HSI'); test_set = Subset(dataset, range(20, 32))
    dataloader = torch.utils.data.DataLoader(test_set, batch_size=1, shuffle=False)

    MSFA = torch.tensor([[0, 1, 2, 3],
                        [4, 5, 6, 7],
                        [8, 9, 10, 11],
                        [12, 13, 14, 15]])
    
    demosaic_net, ps_net = Mpattern_opt(args), HWViT(16, 1, 32, 32, 8, 0.085)
    demosaic_net.load_state_dict(load_checkpoint_compat(demosaic_ckpt_path, map_location="cpu"), strict=False)
    demosaic_net.eval().to(device_default)
    
    ps_net.load_state_dict(load_checkpoint_compat(pan_shap_ckpt_path, map_location="cpu"), strict=False)
    ps_net.eval().to(device_default)
    

    psnr_list = []
    sam_list = []
    ssim_list = []
    ergas_list = []
    for idx, (raw, pan, msi_gt) in enumerate(dataloader):
        
        raw = raw.to(device_default, dtype=dtype_default)
        pan = pan.to(device_default, dtype=dtype_default)
        msi_gt = msi_gt.to(device_default, dtype=dtype_default)

        with torch.no_grad():
            scale_coord_map = input_matrix_wpn(raw.shape[2], raw.shape[3], MSFA.shape[0]).to(device_default)
            mosaic_up = torch.zeros(raw.shape[0], MSFA.shape[0]*MSFA.shape[1], raw.shape[2], raw.shape[3]).to(device_default)
            for i in range(MSFA.shape[0]):
                for j in range(MSFA.shape[1]):
                    mosaic_up[:, i*MSFA.shape[1]+j, i::MSFA.shape[0], j::MSFA.shape[1]] = raw[:, 0, i::MSFA.shape[0], j::MSFA.shape[1]]
            demosaic_tensor = demosaic_net([mosaic_up, raw], scale_coord_map)
            X = utils.generate_patch(demosaic_tensor, pan, ps_net, size=64, recon_size=32, ratio=2).detach()

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