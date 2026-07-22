import torch 
from torch.utils import data
import torch.nn.functional as F
import numpy as np
from pathlib import Path
import h5py
import tifffile as tiff

from project_utils.common import im2single, repeat_to

__all__ = ["CAVE", "NTIRE", "CAVEDatasetDC"]

class RealDataset(data.Dataset):
    def __init__(
        self, root: str, **kwargs
    ) -> None:
        super().__init__()
        self.root = Path(root)
        
        self.data_files = []
        self.data_shape = []
        self.wavelengths: torch.Tensor = None # The 2D array of wavelengths

        self.init_dataset_info()
        self.msfa = self._init_msfa()
    
    def __len__(self) -> int:
        return len(self.data_files)
    
    def __getitem__(self, index: int):
        raw = self.load_data(index)
        return raw, torch.zeros(self.data_shape)
    
    def _init_msfa(self) -> torch.Tensor:
        sorted_unique = torch.unique(self.wavelengths, sorted=True)
        msfa_pattern = torch.searchsorted(sorted_unique, self.wavelengths)
        msfa = repeat_to(msfa_pattern, self.data_shape[1:])
        return msfa.unsqueeze(0)

    def init_dataset_info(self):
        raise NotImplementedError("子类必须实现数据集专属信息初始化方法！")
    
    def load_data(self, index: int):
        raise NotImplementedError("子类必须实现数据加载方法！")

class Specvision(RealDataset):
    def __init__(self, root, camera_id=0):
        self.camera_id = camera_id
        super().__init__(root)

    def init_dataset_info(self):
        self.data_files = list(self.root.glob('*.raw'))
        self.data_shape = [9, 1020, 1020] 
        self.wavelengths = torch.tensor([
            [[514, 468, 556],
             [496, 454, 540],
             [530, 485, 570]],
            [[698, 757, 638],
             [719, 776, 657],
             [675, 735, 620]]
        ])[self.camera_id]

    def load_data(self, index: int) -> torch.Tensor:
        raw_file = self.data_files[index]
        data = np.fromfile(
            raw_file, dtype=np.uint16
        ).reshape(2, 2040, 2040)
        
        data_tensor = im2single(
            torch.from_numpy(data[self.camera_id]), bit_depth=12
        )[None]
        
        data_tensor = F.avg_pool2d(data_tensor, kernel_size=2, stride=2) # Quad 合并
        return data_tensor


# ------------------------------- Simulation dataset ------------------------------- #

class CAVEDatasetDC(torch.utils.data.Dataset):
    def __init__(self, root, transform=None, channel_range=[12, 28]):
        self.root = Path(root)
        self.data_files = list(self.root.glob('*.tiff'))
        
        self.down_scale = 2
        self.wavelengths = 400. + torch.arange(channel_range[0], channel_range[1]) * 10.
        self.channel_range = channel_range
        self.data_shape = [channel_range[1] - channel_range[0], 512, 512]
        self.transform = transform
        
        self.__init_msfa()
        self.__init_pan()
    
    def __init_pan(self):
        spe_res = torch.tensor([1., 1., 2., 4., 
                                8., 9., 10., 12., 
                                16., 12., 10., 9., 
                                7., 3., 2., 1.]).reshape(self.data_shape[0], 1, 1)
        self.spe_res = spe_res / spe_res.sum()
        

    def __init_msfa(self):
        msfa_array = torch.argsort(self.wavelengths).reshape(4, 4)
        msfa = repeat_to(msfa_array, [self.data_shape[1]//self.down_scale, self.data_shape[2]//self.down_scale])
        self.msfa = msfa.unsqueeze(0)  # shape=[1, H, W]
        

    def __len__(self):
        return len(self.data_files)

    def __getitem__(self, index):
        hsi = tiff.imread(self.data_files[index])
        hsi = torch.from_numpy(hsi)
        hsi = im2single(hsi)  # shape=[C, H, W], dtype=float32

        hsi = hsi / hsi.max()
        
        if self.transform is not None:
            hsi = self.transform(hsi)
        
        msi = hsi[self.channel_range[0]:self.channel_range[1], :self.data_shape[1], :self.data_shape[2]]

        pan = torch.sum(msi * self.spe_res, dim=0, keepdim=True)

        blured_msi = torch.nn.functional.avg_pool2d(msi, 2, 2)
        raw = torch.gather(blured_msi, dim=0, index=self.msfa)
        return raw, pan, msi


class MosaicDataset(data.Dataset):
    def __init__(
        self, root: str, transform = None, **kwargs
    ) -> None:
        super().__init__()
        self.root = Path(root)
        self.transform = transform
        
        self.data_files = []
        self.data_shape = []
        self.wavelengths: torch.Tensor = None # The 2D array of wavelengths

        self.init_dataset_info()
        self.msfa = self._init_msfa()
    
    def __len__(self) -> int:
        return len(self.data_files)
    
    def __getitem__(self, index: int):
        msi = self.load_data(index)

        if self.transform is not None:
            msi = self.transform(msi)
        
        raw = torch.gather(msi, dim=0, index=self.msfa)
        return raw, msi
    
    def _init_msfa(self) -> torch.Tensor:
        sorted_unique = torch.unique(self.wavelengths, sorted=True)
        msfa_pattern = torch.searchsorted(sorted_unique, self.wavelengths)
        msfa = repeat_to(msfa_pattern, self.data_shape[1:])
        return msfa.unsqueeze(0)

    def init_dataset_info(self):
        raise NotImplementedError("子类必须实现数据集专属信息初始化方法！")
    
    def load_data(self, index: int):
        raise NotImplementedError("子类必须实现数据加载方法！")

class CAVE(MosaicDataset):
    def init_dataset_info(self):
        self.data_files = list(self.root.glob('*.tiff'))
        self.data_shape = [25, 512, 512] 
        self.wavelengths = torch.tensor([
            [520., 570., 620., 580., 550.], 
            [480., 530., 400., 460., 630.], 
            [430., 510., 610., 470., 600.], 
            [640., 420., 500., 490., 540.],
            [410., 440., 590., 450., 560.],
        ])
        self.channel_sequence = self.wavelengths.flatten().argsort()
        self.inverse_channel_sequence = torch.argsort(self.channel_sequence)

    def load_data(self, index: int):
        data = tiff.imread(self.data_files[index])[:self.data_shape[0]]
        data = im2single(torch.from_numpy(data))
        return data

class NTIRE(MosaicDataset):
    def init_dataset_info(self):
        self.data_files = list(self.root.glob('*.mat'))
        self.data_shape = [16, 480, 512] 
        self.wavelengths = torch.tensor([
            [650., 705., 765., 855.], 
            [630., 680., 745., 835.], 
            [415., 465., 530., 580.], 
            [400., 435., 500., 555.]
        ])
        # Bands in the NTIRE 2022 Spectral Demosaic dataset are not ordered by wavelength.
        # Reorder spectral bands to arrange the MSI cube according to the wavelength sequence.
        self.channel_sequence = self.wavelengths.flatten().argsort()
        self.inverse_channel_sequence = torch.argsort(self.channel_sequence)

    def load_data(self, index: int) -> torch.Tensor:
        with h5py.File(self.data_files[index], 'r') as h5:
            data = torch.from_numpy(h5['cube'][()]).mT 
        return data[self.channel_sequence] # [C, H, W]