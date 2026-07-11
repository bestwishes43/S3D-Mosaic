import os
os.environ["KMP_DUPLICATE_LIB_OK"]  =  "TRUE"
import torch
torch.manual_seed(42)
import scipy.io as scio

# Yurong Chen's Code
from optim_code.model_MSFA     import UnNull_Real_MSFA, tensor_weight_conv

# My Code, for comparison
from project_utils.dataloader import Specvision


def main():
    device_default = 'cuda' if torch.cuda.is_available() else 'cpu'
    dtype_default = torch.float32
    #----------------------- Data Configuration -----------------------#
    dataset = Specvision('../../datasets/Specvision'); test_set = dataset
    dataloader = torch.utils.data.DataLoader(test_set, batch_size=1, shuffle=False)

    h, w, nC = 1020, 1020, 9
    msfa = dataset.msfa.to(device_default).squeeze()
    Phi = torch.stack([msfa == ch for ch in range(nC)], dim=-1).float()
    
    for idx, (raw, msi_gt) in enumerate(dataloader):
        raw = raw.to(device_default, dtype=dtype_default)
        truth_tensor = msi_gt.to(device_default, dtype=dtype_default)
        
        meas = raw.squeeze()

        LRHSI_new = torch.zeros_like(Phi)
        for i in range(nC):
            LRHSI_new[:, :, i] = tensor_weight_conv(meas * Phi[:, :, i], nC).to(device_default)
        

        LRHSI = LRHSI_new.unsqueeze(0).permute(0, 3, 1, 2)

        #------------------------- Training Model -------------------------#
        recon = UnNull_Real_MSFA(meas, Phi, LRHSI, truth_tensor)
        
        X_numpy = recon.cpu().numpy()
        meas_3D = (meas.unsqueeze(-1) * Phi).cpu().numpy()
        scio.savemat(f"UnNull_data_{idx}.mat", {"UnNull": X_numpy, "meas_3D":meas_3D})
        

if __name__ == '__main__':
    main()
