import torch
from haralick_torch.io import read_nir_as_tensor, write_haralick_geotiff
from haralick_torch.haralick import compute_haralick
from haralick_torch.tiling import process_in_tiles

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ---- INPUT
img = "image.tif"
out = "haralick.tif"
nir_band = 4

# ---- IO
nir, ref_ds = read_nir_as_tensor(img, nir_band, DEVICE)

# ---- HARALICK
textures = process_in_tiles(
    nir,
    compute_haralick,
    window=11,
    stride=1
)

# ---- SAVE
write_haralick_geotiff(
    out,
    textures,
    list(textures.keys()),
    ref_ds
)
