import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
import os
from scipy import ndimage

def calculate_adc_all(masked_dynDWI, bvals, b0_threshold=50):
    """
    Calcula o Mapa ADC para volumes DWI (b>0),
    utilizando a média dos volumes b=0 como referência.
    masked_dynDWI: Volume 4D [x, y, z, tempo]
    bvals: Vetor com os valores de b para cada volume.
    """
    indices_b0 = np.where(bvals <= b0_threshold)[0]
    indices_dwi = np.where(bvals > b0_threshold)[0]

    if len(indices_b0) == 0:
        raise ValueError("Nenhum volume b0 encontrado no arquivo bvals.")

    diff_volumes = masked_dynDWI[..., indices_dwi]
    b0_mean = np.mean(masked_dynDWI[..., indices_b0], axis=3)
    limiar = 1e-10
    adc = np.full_like(diff_volumes, np.nan)

    with np.errstate(divide='ignore', invalid='ignore'):
        for i, b in enumerate(bvals[indices_dwi]):
            diff = diff_volumes[..., i]
            ratio = diff / b0_mean
            valid = (b0_mean > limiar) & (diff > limiar) & (ratio > 0) & (ratio <= 1.0)
            adc[..., i][valid] = -np.log(ratio[valid]) / b
    #adc[adc < 0] = np.nan
    return adc

def calculate_adc_paired(masked_dynDWI, bvals, b0_threshold=50):
    """
    Calcula o Mapa ADC para volumes DWI,
    utilizando o volume b0 pareado correspondente (1º DWI com 1º b0...).
    """
    indices_b0 = np.where(bvals <= b0_threshold)[0]
    indices_dwi = np.where(bvals > b0_threshold)[0]

    if len(indices_b0) == 0:
        raise ValueError("Nenhum volume b0 encontrado no arquivo bvals.")

    if len(indices_b0) != len(indices_dwi):
        raise ValueError(
            f"Número de b0 ({len(indices_b0)}) difere do número de DWI ({len(indices_dwi)}). "
            "Não é possível realizar o pareamento um-a-um."
        )

    diff_volumes = masked_dynDWI[..., indices_dwi]
    b0_volumes = masked_dynDWI[..., indices_b0]
    limiar = 1e-10
    adc = np.full_like(diff_volumes, np.nan)

    with np.errstate(divide='ignore', invalid='ignore'):
        for i, b in enumerate(bvals[indices_dwi]):
            diff = diff_volumes[..., i]
            b0 = b0_volumes[..., i]
            ratio = diff / b0
            valid = (b0 > limiar) & (diff > limiar) & (ratio > 0) & (ratio <= 1.0)
            adc[..., i][valid] = -np.log(ratio[valid]) / b
    return adc

def processar_mapa_adc_b0_mean(dir_base, return_adc=False, save_adc=True):
    path_dynDWI = os.path.join(dir_base, "dwi_corrected.nii.gz")
    path_bvals = os.path.join(dir_base, "bvals.bval")
    path_b0_mask = os.path.join(dir_base, "b0_brain_mask.nii.gz")

    dwi_img = nib.load(path_dynDWI)
    dynDWI = dwi_img.get_fdata()
    affine = dwi_img.affine
    header = dwi_img.header

    if not os.path.exists(path_bvals):
        raise FileNotFoundError(f"Arquivo bvals não encontrado: {path_bvals}")
    bvals = np.loadtxt(path_bvals)

    if dynDWI.shape[3] != len(bvals):
        raise ValueError("Número de volumes do DWI não corresponde ao número de bvals.")

    if os.path.exists(path_b0_mask):
        print("Aplicando b0_brain_mask.nii.gz")
        b0_mask = nib.load(path_b0_mask).get_fdata()
        dynDWI[b0_mask == 0, :] = np.nan
    else:
        print("b0_brain_mask.nii.gz não encontrado — prosseguindo sem máscara.")

    adc = calculate_adc_all(masked_dynDWI=dynDWI, bvals=bvals)
    if save_adc:
        new_header = header.copy()
        new_header.set_data_shape(adc.shape)
        save_path = os.path.join(dir_base, "Analysis")
        os.makedirs(save_path, exist_ok=True)
        output_file = os.path.join(save_path, "adc.nii.gz")
        nib.save(nib.Nifti1Image(adc, affine=affine, header=new_header), output_file)
        print("ADC (média b0) salvo em:", output_file)

    if return_adc:
        return adc

def processar_mapa_adc_b0_paired(dir_base, return_adc=False, save_adc=True):
    path_dynDWI = os.path.join(dir_base, "dwi_corrected.nii.gz")
    path_bvals = os.path.join(dir_base, "bvals.bval")
    path_b0_mask = os.path.join(dir_base, "b0_brain_mask.nii.gz")

    dwi_img = nib.load(path_dynDWI)
    dynDWI = dwi_img.get_fdata()
    affine = dwi_img.affine
    header = dwi_img.header

    if not os.path.exists(path_bvals):
        raise FileNotFoundError(f"Arquivo bvals não encontrado: {path_bvals}")
    bvals = np.loadtxt(path_bvals)

    if dynDWI.shape[3] != len(bvals):
        raise ValueError("Número de volumes do DWI não corresponde ao número de bvals.")

    if os.path.exists(path_b0_mask):
        print("Aplicando b0_brain_mask.nii.gz")
        b0_mask = nib.load(path_b0_mask).get_fdata()
        dynDWI[b0_mask == 0, :] = np.nan
    else:
        print("b0_brain_mask.nii.gz não encontrado — prosseguindo sem máscara.")

    adc_paired = calculate_adc_paired(masked_dynDWI=dynDWI, bvals=bvals)
    if save_adc:
        new_header = header.copy()
        new_header.set_data_shape(adc_paired.shape)
        save_path = os.path.join(dir_base, "Analysis")
        os.makedirs(save_path, exist_ok=True)
        output_file = os.path.join(save_path, "adc_paried.nii.gz")
        nib.save(nib.Nifti1Image(adc_paired, affine=affine, header=new_header), output_file)
        print("ADC (b0 pareado) salvo em:", output_file)

    if return_adc:
        return adc_paired

def completar_mapa_nan(mapa_4d, min_vizinhos_sinal=5):
    """
    Para cada fatia axial (z) e cada instante de tempo (t), utiliza uma vizinhança 3x3
    (8 vizinhos) no plano x,y. O preenchimento só ocorre se o número de vizinhos
    com sinal for >= min_vizinhos_sinal.
    """
    mapa_saida = mapa_4d.copy()
    kernel = np.ones((3, 3), dtype=np.uint8)
    kernel[1, 1] = 0

    for t in range(mapa_4d.shape[3]):
        for z in range(mapa_4d.shape[2]):
            slice_2d = mapa_saida[..., z, t]
            nan_mask = np.isnan(slice_2d)
            slice_soma = np.nan_to_num(slice_2d, nan=0.0)
            slice_contagem = (~nan_mask).astype(float)
            soma_vizinhos = ndimage.convolve(slice_soma, kernel, mode='constant', cval=0.0)
            contagem_vizinhos = ndimage.convolve(slice_contagem, kernel, mode='constant', cval=0.0)
            media_vizinhos = np.divide(soma_vizinhos, contagem_vizinhos, out=np.full_like(slice_soma, np.nan), where=contagem_vizinhos != 0)
            preencher_mask = nan_mask & (contagem_vizinhos >= min_vizinhos_sinal)
            slice_2d[preencher_mask] = media_vizinhos[preencher_mask]
            mapa_saida[..., z, t] = slice_2d
    return mapa_saida

def visualizar_imagem_3D(adc, fatias_axiais=[8, 14, 20], voxel_size=(2.0, 2.0, 4.0)):
    """
    Visualiza um volume 3D de ADC considerando o tamanho não-isotrópico dos voxels
    e usando posicionamento radiológico correto sem rotação
    """
    
    if adc.ndim == 4:
        volume = adc[:, :, :, 0]
    else:
        volume = adc
    
    dim_x, dim_y, dim_z = volume.shape
    idx_sagital = dim_x // 2
    idx_coronal = dim_y // 2
    vmin, vmax = 0, 0.5*1e-2 # Define os limites do mapa de cores
    
    aspect_axial = voxel_size[0] / voxel_size[1]  # x/y
    aspect_sagital = voxel_size[2] / voxel_size[1]  # z/y 
    aspect_coronal = voxel_size[2] / voxel_size[0]  # z/x
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10), constrained_layout=True)
    
    # Planos axial, sagital e coronal
    axial = np.rot90(volume[:, :, dim_z//2], k=-1)
    im0 = axes[0, 0].imshow(axial, cmap='jet', aspect=aspect_axial, vmin=vmin, vmax=vmax, origin='lower')
    axes[0, 0].set_title(f'Plano Axial (z = {dim_z//2})\nVoxel: {voxel_size[0]}x{voxel_size[1]}mm')
    axes[0, 0].set_xlabel('Y (Esquerda → Direita)')
    axes[0, 0].set_ylabel('X (Posterior → Anterior)')
    plt.colorbar(im0, ax=axes[0, 0])
    
    sagital = volume[idx_sagital, :, :].T
    im1 = axes[0, 1].imshow(sagital, cmap='jet', aspect=aspect_sagital,vmin=vmin, vmax=vmax, origin='lower')
    axes[0, 1].set_title(f'Plano Sagital (x = {idx_sagital})\nVoxel: {voxel_size[2]}x{voxel_size[1]}mm')
    axes[0, 1].set_xlabel('Y (Posterior → Anterior)')
    axes[0, 1].set_ylabel('Z (Inferior → Superior)')
    plt.colorbar(im1, ax=axes[0, 1])
    
    coronal = volume[:, idx_coronal, :].T
    im2 = axes[0, 2].imshow(coronal, cmap='jet', aspect=aspect_coronal,vmin=vmin, vmax=vmax, origin='lower')
    axes[0, 2].set_title(f'Plano Coronal (y = {idx_coronal})\nVoxel: {voxel_size[2]}x{voxel_size[0]}mm')
    axes[0, 2].set_xlabel('X (Esquerda → Direita)')
    axes[0, 2].set_ylabel('Z (Inferior → Superior)')
    plt.colorbar(im2, ax=axes[0, 2])
    
    for i, fatia in enumerate(fatias_axiais):
        fatia = min(max(0, fatia), dim_z - 1)
        axial_fatia = np.rot90(volume[:, :, fatia], k=-1)
        im = axes[1, i].imshow(axial_fatia, cmap='jet', aspect=aspect_axial, vmin=vmin, vmax=vmax, origin='lower')
        axes[1, i].set_title(f'Fatia Axial {i+1} (z = {fatia})\nVoxel: {voxel_size[0]}x{voxel_size[1]}mm')
        axes[1, i].set_xlabel('Y (Esquerda → Direita)')
        axes[1, i].set_ylabel('X (Posterior → Anterior)')
        plt.colorbar(im, ax=axes[1, i])
    
    fig.suptitle(f'Visualização ADC - Resolução: {voxel_size[0]}x{voxel_size[1]}x{voxel_size[2]}mm', fontsize=14, fontweight='bold')

    for ax in axes.flat:
        ax.set_xticks([])
        ax.set_yticks([])
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)
    
    plt.show()

def visualizar_grid_temporal_axial(volume_4d, numero_fatia,indices_temporais=None, vmin=0, vmax=0.5e-2):
    """
    Visualiza um grid 3x3 de fatias axiais de um volume 4D,
    comparando diferentes índices de tempo (4ª dimensão).
    Esta versão assume que os voxels são isotrópicos no plano axial (x/y),
    portanto 'aspect=1.0' é usado.
    indices_temporais: lista de até 9 índices da 4ª dimensão.
    vmin, vmax: limites do colormap.
    """
        
    max_t = volume_4d.shape[3] - 1
    if indices_temporais is None:
        indices_para_plotar = list(range(9))
    else:
        indices_para_plotar = indices_temporais[:9]

    fig, axes = plt.subplots(3, 3, figsize=(12, 12), constrained_layout=True)
    for i, ax in enumerate(axes.flat):
        if i < len(indices_para_plotar):
            idx_t = indices_para_plotar[i]
            if idx_t > max_t:
                ax.set_title(f'Tempo (t) = {idx_t}\n(Índice Inválido)')
                ax.axis('off')
                continue
            volume_3d_tempo = volume_4d[:, :, :, idx_t]
            fatia = volume_3d_tempo[:, :, numero_fatia]
            fatia_plot = np.rot90(fatia, k=-1)
            ax.imshow(fatia_plot, cmap='jet', aspect=1.0, 
                      vmin=vmin, vmax=vmax, origin='lower')
            ax.set_title(f'Tempo (t) = {idx_t}')
            ax.axis('off') 
        else:
            ax.axis('off')
    fig.suptitle(f'Comparação Temporal - Fatia Axial (z) = {numero_fatia}',fontsize=16, fontweight='bold')
    plt.show()

def calcular_curva_media(adc_4d, roi_3d):
    return np.nanmean(adc_4d[roi_3d == 1], axis=0)
