import numpy as np
import nibabel as nib

dyn = nib.load("/home/joao-oliveira/Documents/mapa-vasos/dwi_corrected.nii.gz")
b0 = nib.load("/home/joao-oliveira/Documents/mapa-vasos/b0_final.nii.gz")
data = dyn.get_fdata()

dwi = data[:, :, :, 0:60]
limiar = 1e-10 

indices_b0 = np.arange(60, 120, 1)
indices_dwi = np.arange(0, 60, 1)
diff_volumes = dwi[..., indices_dwi]
limiar = 1e-10 
b0_mean_all = np.mean(data[..., indices_b0], axis=3)
b0_expanded = np.repeat(b0_mean_all[..., np.newaxis], diff_volumes.shape[3], axis=3)
mask = (b0_expanded > limiar) & (diff_volumes > limiar)
adc = np.zeros_like(diff_volumes)
adc[mask] = -np.log(diff_volumes[mask] / b0_expanded[mask]) / 150


amplitude = np.max(adc, axis=3) - np.min(adc, axis=3)
amplitude = np.nan_to_num(amplitude)

amp_img = nib.Nifti1Image(
    amplitude.astype(np.float32),
    affine=b0.affine,
    header=b0.header
)

out_path = "/home/joao-oliveira/Documents/mapa-vasos/amplitude_map.nii.gz"
nib.save(amp_img, out_path)

print(f"Mapa salvo em:\n{out_path}")