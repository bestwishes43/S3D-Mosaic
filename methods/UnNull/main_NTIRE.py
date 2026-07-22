import sys, os
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')); sys.path.insert(0, _ROOT)
os.environ["KMP_DUPLICATE_LIB_OK"]  =  "TRUE"
import torch
torch.manual_seed(42)

# Yurong Chen's Code
from optim_code.model_MSFA     import UnNull_ARAD_MSFA, tensor_weight_conv

# My Code, for comparison
from project_utils.dataloader import NTIRE
from project_utils.metrics import calculate_psnr, calculate_sam, SSIM


def main_MSFA_NTIRE():
    device_default = 'cuda' if torch.cuda.is_available() else 'cpu'
    dtype_default = torch.float32
    calculate_ssim = SSIM().to(device_default)
    #----------------------- Data Configuration -----------------------#
    dataset = NTIRE(os.path.join(_ROOT, 'datasets/NTIRE/Valid_spectral_16')); test_set = dataset
    dataloader = torch.utils.data.DataLoader(test_set, batch_size=1, shuffle=False)

    h, w, nC = 480, 512, 16
    msfa = dataset.msfa.to(device_default).squeeze()
    msfa = msfa[:h, :w]
    Phi = torch.stack([msfa == ch for ch in range(nC)], dim=-1).float()
    

    psnr_list = []
    ssim_list = []
    sam_list = []
    time_list = []
    for idx, (raw, msi_gt) in enumerate(dataloader):
        msi_gt = msi_gt[..., :h, :w]
        raw = raw.to(device_default, dtype=dtype_default)
        truth_tensor = msi_gt.to(device_default, dtype=dtype_default)
        data_truth = truth_tensor[0].permute(1, 2, 0)
        
        meas = torch.sum(data_truth * Phi, 2)

        
        meas_3d = data_truth * Phi
        LRHSI_new = torch.zeros_like(Phi)
        for i in range(nC):
            LRHSI_new[:, :, i] = tensor_weight_conv(meas_3d[:, :, i] * Phi[:, :, i], nC).to(device_default)
            
        LRHSI = LRHSI_new.unsqueeze(0).permute(0, 3, 1, 2)
        PSNR = calculate_psnr(truth_tensor, LRHSI)

        # print('                                                        ')
        # print('--------- Start :')
        # print('--------- LRHSI PSNR:', PSNR)
        # print('                                                        ')
        
        
        #Sparse_noise = np.random.choice((0, 1, 2), size=(meas.shape[0], meas.shape[1]), p=[0.99, 0.01/2., 0.01/2.])
        #Gauss_noise = np.random.normal(loc=0.5, scale=0.5, size=(meas.shape[0], meas.shape[1]))
        #meas = meas + 0.1*torch.from_numpy(Gauss_noise).float().to(device)
        #meas[Sparse_noise == 1] = torch.max(meas)
        #meas[Sparse_noise == 2] = 0
        #meas = meas.float()


        #------------------------- Training Model -------------------------#
        recon, running_time = UnNull_ARAD_MSFA(meas, Phi, LRHSI, truth_tensor)
        
        #------------------------- Evaluation -------------------------#
        psnr = calculate_psnr(truth_tensor, recon.unsqueeze(0).permute(0, 3, 1, 2)).item()
        sam = calculate_sam(truth_tensor, recon.unsqueeze(0).permute(0, 3, 1, 2)).item()
        ssim = calculate_ssim(truth_tensor, recon.unsqueeze(0).permute(0, 3, 1, 2)).item()
        psnr_list.append(psnr)
        sam_list.append(sam)
        ssim_list.append(ssim)
        time_list.append(running_time)
        print(
            f"|{idx}|{psnr:.2f}|{sam:.3f}|{ssim:.3f}|{running_time:.2f}|"  
        )
    print(torch.tensor(psnr_list).mean(), torch.tensor(sam_list).mean(), torch.tensor(ssim_list).mean(), torch.tensor(time_list).mean())

if __name__ == '__main__':
    main_MSFA_ARAD()