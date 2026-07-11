from imageio.v2 import imread
import numpy as np
from pathlib import Path
import shutil
import tifffile as tiff

if __name__ == '__main__':
    # Original CAVE dataset, which can be downloaded from https://cave.cs.columbia.edu/old/databases/multispectral/zip/complete_ms_data.zip
    raw_dir = Path("complete_ms_data")
    out_dir = Path("./")
    
    for sub in ["HSI", "RGB"]:
        (out_dir / sub).mkdir(parents=True, exist_ok=True)

    wavelengths = list(range(400, 701, 10))

    for scene_root in raw_dir.iterdir():
        if not scene_root.is_dir():
            continue
        scene_name = scene_root.stem
        img_dir = scene_root / scene_name

        img_stack = []
        for png_path in img_dir.glob("*_ms_*.png"):
            img = imread(png_path, format='PNG-PIL', pilmode='I;16')
            img_stack.append(img)
        img_stack = np.asarray(img_stack)

        if scene_name == "watercolors_ms":
            img_stack = img_stack.astype(np.uint8)

        tiff.imwrite(
            out_dir / "HSI" / f"{scene_name}.tiff",
            img_stack,
            imagej=True,
            compression='DEFLATE',
            predictor=2,
            metadata={'wavelengths': wavelengths}
        )
        
        rgb_name = scene_name.replace('_ms', '_RGB') + ".bmp"
        shutil.copy(img_dir / rgb_name, out_dir / "RGB" / rgb_name)
        print(f"✅ Scene {scene_name} completed.")
    print("\n🎉 All scenes completed!")
        
