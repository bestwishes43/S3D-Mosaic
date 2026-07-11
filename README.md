# S3D-Mosaic

This repository provides the official PyTorch implementation for the paper **"S3D-Mosaic: Cross-Band Self-Structure Similarity Spectral Demosaicing via Subgradient Guidance"**.

---

## 📦 Dataset Download

We evaluate our method on the following spectral demosaicing datasets:

| Dataset                                    | Bands | Source                                                                                    | Download Link                                                                             |
| ------------------------------------------ | ----- | ----------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------- |
| **CAVE**                             | 25    | [Columbia Multispectral Repository](https://cave.cs.columbia.edu/repository/Multispectral) | [ZIP](https://cave.cs.columbia.edu/old/databases/multispectral/zip/complete_ms_data.zip)   |
| **NTIRE 2022**                       | 16    | [NTIRE Spectral Demosaicing Challenge](https://codalab.lisn.upsaclay.fr/competitions/722)  | [Baidu Pan](https://pan.baidu.com/s/1REEnuPON7pcok4H0nBWtFg?pwd=mig2) (password: `mig2`) |
| **Real‑world UAV** (9‑band mosaic) | 9     | Captured by a UAV‑mounted camera (our release)                                           | [Baidu Pan](https://pan.baidu.com/s/1feZLKEJomqxPR1MJhH0eTQ?pwd=hpmb) (password: `hpmb`) |

> **Note:** For the CAVE dataset, the original data is spectrally sampled and converted to 25 bands automatically by our data loader.

---

## 🚀 Evaluation

Run the main script inside each method folder to perform evaluation:

```bash
cd ./methods/<method_name>
python main.py
```

- For **compared methods**, we only provide the evaluation code (pre‑trained models or inference scripts are included where applicable).
- **📢 The full pipeline of S3D-Mosaic will go live here as soon as our paper gets accepted.**

---

## ⚙️ On‑the‑fly Compilation of DSE‑DConv

Our method uses `torch.utils.cpp_extension.load_inline` to compile the **DSE‑DConv** operator at runtime. Therefore, a working C++ compiler and CUDA development environment are **required**.

- **Windows**:

  - Install [Microsoft Visual Studio Build Tools](https://visualstudio.microsoft.com/downloads/#build-tools-for-visual-studio-2022) (or Visual Studio with C++ support).
  - Before running the script, open a command prompt and execute:

    ```cmd
    "C:\Program Files\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
    ```

    (Adjust the path according to your installation.)
  - Then run the Python evaluation script in the same terminal.

---

## 🤝 Acknowledgements

We thank the providers of the CAVE and NTIRE datasets. The real‑world UAV dataset was captured and released by our team.
We also thank the authors of [**USD**](https://ieeexplore.ieee.org/document/10443845), [**UnNull**](https://ieeexplore.ieee.org/document/10970444), [**EFN**](https://ieeexplore.ieee.org/document/11367377), and [**EBVIF**](https://ieeexplore.ieee.org/document/11480451) for making their source code publicly available.
