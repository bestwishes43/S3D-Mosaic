import torch
torch.manual_seed(42)
from torch.utils.data import Subset

from project_utils.dataloader import CAVE, NTIRE
from project_utils.metrics import calculate_psnr, calculate_sam, SSIM

def input_matrix_wpn(inH, inW, msfa_size):
    h_offset_coord = torch.zeros(inH, inW, 1)
    w_offset_coord = torch.zeros(inH, inW, 1)
    for i in range(0,msfa_size):
        h_offset_coord[i::msfa_size, :, 0] = (i+1)/msfa_size
        w_offset_coord[:, i::msfa_size, 0] = (i+1)/msfa_size

    pos_mat = torch.cat((h_offset_coord, w_offset_coord), 2)
    pos_mat = pos_mat.contiguous().view(1, -1, 2)
    return pos_mat

def main(msfa_size=5):
    device_default = 'cuda' if torch.cuda.is_available() else 'cpu'
    dtype_default = torch.float32
    calculate_ssim = SSIM().to(device_default)

    if msfa_size == 5:
        ckpt_path = "./checkpoint/CAVE_best_10860.pth"
        dataset = CAVE('../../datasets/CAVE/HSI'); test_set = Subset(dataset, [0, 1, 2, 3, 4, 5, 8, 16, 17, 21])
    else:
        ckpt_path = "./checkpoint/ARAD_best_297.pth"
        dataset = NTIRE('../../datasets/NTIRE/test_spectral_16'); test_set = dataset

    C, H, W = dataset.data_shape
    H = H // msfa_size * msfa_size
    W = W // msfa_size * msfa_size

    mask = torch.stack([
        dataset.msfa == ch for ch in range(C)], dim=1
    ).to(device_default, dtype=dtype_default)[..., :H, :W]
    
    dataloader = torch.utils.data.DataLoader(test_set, batch_size=1, shuffle=False)
    
    model = torch.load(ckpt_path, weights_only=False)["model"]
    model.eval().to(device_default)

    psnr_list = []
    sam_list = []
    ssim_list = []
    for idx, (raw, msi_gt) in enumerate(dataloader):
        raw = raw.to(device_default, dtype=dtype_default)[..., :H, :W]
        msi_gt = msi_gt.to(device_default, dtype=dtype_default)[..., :H, :W]
        
        im_input = (mask * raw)[:, dataset.inverse_channel_sequence]
        
        scale_coord_map = input_matrix_wpn(H, W, msfa_size).to(device_default)
        with torch.no_grad():
            X = model([im_input, raw], scale_coord_map) 
        X = X[:, dataset.channel_sequence]

        psnr = calculate_psnr(msi_gt, X).item()
        sam = calculate_sam(msi_gt, X).item()
        ssim = calculate_ssim(msi_gt, X).item()
        print(f"|{idx}|{psnr:.3f}|{sam:.4f}|{ssim:.4f}|")
        psnr_list.append(psnr)
        sam_list.append(sam)
        ssim_list.append(ssim)
    print(torch.tensor(psnr_list).mean(), torch.tensor(sam_list).mean(), torch.tensor(ssim_list).mean())


if __name__ == "__main__":
    
    main()