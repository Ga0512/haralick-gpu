import torch
from haralick_torch.io import read_tiff_as_tensor
from haralick_torch.tiling import process_in_tiles
from haralick_torch.utils import save_as_tif

device = "cuda" if torch.cuda.is_available() else "cpu"

img, ref = read_tiff_as_tensor("image.tif", device)
textures = process_in_tiles(img, tile_size=256, window_size=15, levels=32)

for name, tensor in textures.items():
    save_as_tif(tensor.numpy(), ref, f"{name}.tif")
