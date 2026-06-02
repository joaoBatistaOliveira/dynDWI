import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import nibabel as nib
import os
import json
import matplotlib.colors as mcolors
import matplotlib.gridspec as gridspec
from scipy.ndimage import gaussian_filter
import scipy.signal as ss
from scipy.signal import find_peaks
from scipy.signal import savgol_filter
from scipy import ndimage



def calculate_adc_all_original(masked_dynDWI, indice = None):
    """
    Calcula o Mapa de Coeficiente de Difusão Aparente (ADC) 
    para volumes DWI (b>0), utilizando a média dos volumes b=0 como referência.
    Parâmetros:
        masked_dynDWI (np.array): Array 4D contendo as imagens de DWI. 
                                  Assume-se que os volumes DWI (b>0) estão nos índices 0-59 
                                  e os volumes b=0 estão nos índices 60-119 (ao longo do eixo 3).
    
    Retorna:
        np.array: O Mapa de ADC (com dimensões X, Y, Z, 60).
    """
    if indice is None:
        indices_b0 = np.arange(60, 120, 1)
        indices_dwi = np.arange(0, 60, 1)
    else:
        indices_b0 = np.arange(indice, indice*2, 1)
        indices_dwi = np.arange(0, indice, 1)
    diff_volumes = masked_dynDWI[..., indices_dwi]
    limiar = 1e-10 
    b0_mean_all = np.mean(masked_dynDWI[..., indices_b0], axis=3)
    b0_expanded = np.repeat(b0_mean_all[..., np.newaxis], diff_volumes.shape[3], axis=3)
    mask = (b0_expanded > limiar) & (diff_volumes > limiar)
    adc = np.zeros_like(diff_volumes)
    adc[mask] = -np.log(diff_volumes[mask] / b0_expanded[mask]) / 150
    
    return adc

def calculate_adc_all(masked_dynDWI, indice=None):
    """
    Calcula o Mapa de Coeficiente de Difusão Aparente (ADC) 
    para volumes DWI (b>0), utilizando a média dos volumes b=0 como referência.
    
    ADC é forçado a ser não-negativo por restrição física.
    """
    if indice is None:
        indices_b0 = np.arange(60, 120, 1)
        indices_dwi = np.arange(0, 60, 1)
    else:
        indices_b0 = np.arange(indice, indice*2, 1)
        indices_dwi = np.arange(0, indice, 1)
    
    diff_volumes = masked_dynDWI[..., indices_dwi]
    limiar = 1e-10 
    

    b0_mean_all = np.mean(masked_dynDWI[..., indices_b0], axis=3)
    b0_expanded = np.repeat(b0_mean_all[..., np.newaxis], diff_volumes.shape[3], axis=3)
    
    # Máscara para voxels válidos
    mask = (b0_expanded > limiar) & (diff_volumes > limiar)
    
    adc = np.zeros_like(diff_volumes)

    with np.errstate(divide='ignore', invalid='ignore'):
        ratio = diff_volumes / b0_expanded
        valid_adc_mask = mask & (ratio <= 1.0) & (ratio > 0)
        adc[valid_adc_mask] = -np.log(ratio[valid_adc_mask]) / 150
    #preencher ADCs negativos/inválidos NaN
    adc[~valid_adc_mask] = np.nan 
    
    return adc

def calculate_adc_all(masked_dynDWI, bvals, b0_threshold=50):
    indices_b0 = np.where(bvals <= b0_threshold)[0]
    indices_dwi = np.where(bvals > b0_threshold)[0]

    if len(indices_b0) == 0:
        raise ValueError("Nenhum volume b0 encontrado no arquivo bvals.bval.")

    diff_volumes = masked_dynDWI[..., indices_dwi]
    b0_mean = np.mean(masked_dynDWI[..., indices_b0], axis=3)

    limiar = 1e-10
    adc = np.full_like(diff_volumes, np.nan)

    with np.errstate(divide='ignore', invalid='ignore'):
        for i, b in enumerate(bvals[indices_dwi]):
            ratio = diff_volumes[..., i] / b0_mean
            valid = (b0_mean > limiar) & (diff_volumes[..., i] > limiar) & (ratio > 0) & (ratio <= 1.0)
            adc[..., i][valid] = -np.log(ratio[valid]) / b

    return adc


def calculate_adc_paired(masked_dynDWI, indice=None):
    """
    Calcula o Mapa de Coeficiente de Difusão Aparente (ADC) 
    para volumes DWI (b>0), utilizando o volume b=0 pareado correspondente.
    Parâmetros:
    masked_dynDWI (np.array): Volume 4D com todos os dados.
    indice (int, opcional): O índice que separa os volumes DWI dos volumes b0.
                            Se None, assume 60 volumes DWI (0-59) e 60 b0 (60-119).
    Retorna:
    np.array: Um mapa 4D de ADC com as mesmas dimensões [x, y, z, tempo]
              que os volumes de difusão de entrada.
    """
    if indice is None:
        indices_b0 = np.arange(60, 120, 1)
        indices_dwi = np.arange(0, 60, 1)
    else:
        indices_b0 = np.arange(indice, indice*2, 1)
        indices_dwi = np.arange(0, indice, 1)
    
    diff_volumes = masked_dynDWI[..., indices_dwi]
    limiar = 1e-10 

    b0_paired_volumes = masked_dynDWI[..., indices_b0]
    mask = (b0_paired_volumes > limiar) & (diff_volumes > limiar)
    adc = np.zeros_like(diff_volumes)

    with np.errstate(divide='ignore', invalid='ignore'):
        ratio = diff_volumes / b0_paired_volumes
        valid_adc_mask = mask & (ratio <= 1.0) & (ratio > 0)
        adc[valid_adc_mask] = -np.log(ratio[valid_adc_mask]) / 150
        
    adc[~valid_adc_mask] = np.nan 
    return adc

def suavizar_volumes_3d(imagem_4d, sigma=1.0):
    """
    Aplica filtro gaussiano nas dimensões espaciais (últimas 3 dimensões)
    mantendo a dimensão temporal intacta.
    """
    return gaussian_filter(imagem_4d, sigma=(sigma, sigma, sigma, 0))

def processar_mapa_adc_b0_mean(dir_base, return_adc = False):
    path_dynDWI = os.path.join(dir_base, "dwi_corrected.nii.gz")
    Affine = nib.load(path_dynDWI).affine
    header = nib.load(path_dynDWI).header
    dynDWI = nib.load(path_dynDWI).get_fdata()
    path_b0 = os.path.join(dir_base, "b0_brain_mask.nii.gz")
    b0 = nib.load(path_b0).get_fdata()
    dynDWI[b0 == 0, :] = np.nan
    adc= calculate_adc_all(masked_dynDWI=dynDWI, indice = dynDWI.shape[3]//2)
    new_header = header.copy()
    new_header.set_data_shape(adc.shape)
    save_path = os.path.join(dir_base, 'Analysis')
    os.makedirs(save_path, exist_ok=True)
    nib.save(nib.Nifti1Image(adc, affine=Affine, header=new_header), os.path.join(save_path, "adc.nii.gz"))
    print("ADC salvo em: ", save_path)
    if return_adc:
        return adc


def processar_mapa_adc_b0_mean(dir_base, return_adc=False):
    path_dynDWI = os.path.join(dir_base, "dynDWI.nii.gz")
    path_bvals = os.path.join(dir_base, "bvals.bval")
    path_b0_mask = os.path.join(dir_base, "b0_brain_mask.nii.gz")

    # Carregar DWI
    dwi_img = nib.load(path_dynDWI)
    dynDWI = dwi_img.get_fdata()
    affine = dwi_img.affine
    header = dwi_img.header

    # Carregar bvals
    if not os.path.exists(path_bvals):
        raise FileNotFoundError(f"Arquivo bvals não encontrado: {path_bvals}")

    bvals = np.loadtxt(path_bvals)

    if dynDWI.shape[3] != len(bvals):
        raise ValueError("Número de volumes do DWI não corresponde ao número de bvals.")

    # Aplicar máscara se existir
    if os.path.exists(path_b0_mask):
        print("Aplicando b0_brain_mask.nii.gz")
        b0_mask = nib.load(path_b0_mask).get_fdata()
        dynDWI[b0_mask == 0, :] = np.nan
    else:
        print("b0_brain_mask.nii.gz não encontrado — prosseguindo sem máscara.")

    # Calcular ADC
    adc = calculate_adc_all(
        masked_dynDWI=dynDWI,
        bvals=bvals
    )

    # Atualizar header
    new_header = header.copy()
    new_header.set_data_shape(adc.shape)

    # Salvar
    save_path = os.path.join(dir_base, "Analysis")
    os.makedirs(save_path, exist_ok=True)

    adc_path = os.path.join(save_path, "adc.nii.gz")
    nib.save(nib.Nifti1Image(adc, affine=affine, header=new_header), adc_path)

    print("ADC salvo em:", adc_path)

    if return_adc:
        return adc


def processar_mapa_adc_b0_paired(dir_base, return_adc = False):
    path_dynDWI = os.path.join(dir_base, "dwi_corrected.nii.gz")
    Affine = nib.load(path_dynDWI).affine
    header = nib.load(path_dynDWI).header
    dynDWI = nib.load(path_dynDWI).get_fdata()
    path_b0 = os.path.join(dir_base, "b0_brain_mask.nii.gz")
    b0 = nib.load(path_b0).get_fdata()
    dynDWI[b0 == 0, :] = np.nan
    adc = calculate_adc_paired(masked_dynDWI=dynDWI, indice = dynDWI.shape[3]//2)
    new_header = header.copy()
    new_header.set_data_shape(adc.shape)
    save_path = os.path.join(dir_base, 'Analysis')
    os.makedirs(save_path, exist_ok=True)
    nib.save(nib.Nifti1Image(adc, affine=Affine, header=new_header), os.path.join(save_path, "adc.nii.gz"))
    print("ADC salvo em: ", save_path)
    if return_adc:
        return adc

def visualizar_imagem_3D(adc, fatias_axiais=[8, 14, 20], voxel_size=(2.0, 2.0, 4.0)):
    """
    Visualiza um volume 3D de ADC considerando o tamanho não-isotrópico dos voxels
    e usando posicionamento radiológico correto sem rotação
    
    Parâmetros:
    adc: array 4D com dimensões [x, y, z, tempo] ou 3D [x, y, z]
    fatias_axiais: lista com os índices das 3 fatias axiais para mostrar
    voxel_size: tupla com o tamanho dos voxels em mm (x, y, z)
    """
    
    if adc.ndim == 4:
        volume = adc[:, :, :, 0]
    else:
        volume = adc
    
 
    dim_x, dim_y, dim_z = volume.shape
    idx_sagital = dim_x // 2
    idx_coronal = dim_y // 2
    
    # Define os limites do mapa de cores
    vmin, vmax = 0, 1e-2
    
    # Calcula os aspectos ratio baseado no tamanho dos voxels
    aspect_axial = voxel_size[0] / voxel_size[1]  # x/y
    aspect_sagital = voxel_size[2] / voxel_size[1]  # z/y 
    aspect_coronal = voxel_size[2] / voxel_size[0]  # z/x
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10), constrained_layout=True)
    
    # Planos axial, sagital e coronal
    axial = np.rot90(volume[:, :, dim_z//2], k=-1)
    im0 = axes[0, 0].imshow(axial, cmap='jet', aspect=aspect_axial, 
                           vmin=vmin, vmax=vmax, origin='lower')
    axes[0, 0].set_title(f'Plano Axial (z = {dim_z//2})\nVoxel: {voxel_size[0]}x{voxel_size[1]}mm')
    axes[0, 0].set_xlabel('Y (Esquerda → Direita)')
    axes[0, 0].set_ylabel('X (Posterior → Anterior)')
    plt.colorbar(im0, ax=axes[0, 0])
    
    sagital = volume[idx_sagital, :, :].T
    im1 = axes[0, 1].imshow(sagital, cmap='jet', aspect=aspect_sagital, 
                           vmin=vmin, vmax=vmax, origin='lower')
    axes[0, 1].set_title(f'Plano Sagital (x = {idx_sagital})\nVoxel: {voxel_size[2]}x{voxel_size[1]}mm')
    axes[0, 1].set_xlabel('Y (Posterior → Anterior)')
    axes[0, 1].set_ylabel('Z (Inferior → Superior)')
    plt.colorbar(im1, ax=axes[0, 1])
    
    coronal = volume[:, idx_coronal, :].T
    im2 = axes[0, 2].imshow(coronal, cmap='jet', aspect=aspect_coronal, 
                           vmin=vmin, vmax=vmax, origin='lower')
    axes[0, 2].set_title(f'Plano Coronal (y = {idx_coronal})\nVoxel: {voxel_size[2]}x{voxel_size[0]}mm')
    axes[0, 2].set_xlabel('X (Esquerda → Direita)')
    axes[0, 2].set_ylabel('Z (Inferior → Superior)')
    plt.colorbar(im2, ax=axes[0, 2])
    
    # 3 fatias axiais
    for i, fatia in enumerate(fatias_axiais):
        fatia = min(max(0, fatia), dim_z - 1)
        axial_fatia = np.rot90(volume[:, :, fatia], k=-1)
        
        im = axes[1, i].imshow(axial_fatia, cmap='jet', aspect=aspect_axial, 
                              vmin=vmin, vmax=vmax, origin='lower')
        axes[1, i].set_title(f'Fatia Axial {i+1} (z = {fatia})\nVoxel: {voxel_size[0]}x{voxel_size[1]}mm')
        axes[1, i].set_xlabel('Y (Esquerda → Direita)')
        axes[1, i].set_ylabel('X (Posterior → Anterior)')
        plt.colorbar(im, ax=axes[1, i])
    
    fig.suptitle(f'Visualização ADC - Resolução: {voxel_size[0]}x{voxel_size[1]}x{voxel_size[2]}mm', 
                 fontsize=14, fontweight='bold')

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
    
    Parâmetros:
    volume_4d: array 4D [x, y, z, tempo]
    numero_fatia: int, o índice 'z' da fatia axial a ser mostrada.
    indices_temporais: lista de até 9 índices da 4ª dimensão (tempo).
                       Se None, usa [0, 1, 2, 3, 4, 5, 6, 7, 8].
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

    fig.suptitle(f'Comparação Temporal - Fatia Axial (z) = {numero_fatia}', 
                 fontsize=16, fontweight='bold')
    plt.show()

def completar_mapa_nan(mapa_4d, min_vizinhos_sinal=23):
    """
    Preenche valores NaN em um mapa 4D baseado na média dos vizinhos,
    apenas se o número de vizinhos com sinal atingir um limiar mínimo.

    Parâmetros:
    mapa_4d (np.array): O mapa 4D [x, y, z, t] contendo NaNs.
    min_vizinhos_sinal (int): O número mínimo de vizinhos não-NaN (de 26)
                              necessário para preencher um voxel NaN.
                              Se for muito baixo (ex: 1), preencherá agressivamente.
                              Se for muito alto (ex: 20), será muito conservador.

    Retorna:
    np.array: Um novo mapa 4D com os NaNs indicados preenchidos.
    """
    
    if mapa_4d.ndim != 4:
        raise ValueError(f"O mapa de entrada deve ser 4D. Dimensões fornecidas: {mapa_4d.ndim}")

    mapa_saida = mapa_4d.copy()
    kernel_vizinhos = np.ones((3, 3, 3), dtype=np.uint8)
    kernel_vizinhos[1, 1, 1] = 0  # Ignora o voxel central
    for t in range(mapa_4d.shape[3]):
        volume_3d = mapa_saida[..., t]
        mascara_nan_original = np.isnan(volume_3d)
        
        #Prepara os mapas para convolução
        #substitui NaN por 0 para que não afetem a soma
        mapa_soma = np.nan_to_num(volume_3d, nan=0.0)

        # Mapa para CONTAGEM: 1 onde há sinal, 0 onde há NaN
        mapa_contagem = (~mascara_nan_original).astype(float)
        
        #Aplica a convolução
        # Calcula a SOMA dos vizinhos não-NaN para cada voxel
        soma_vizinhos = ndimage.convolve(mapa_soma, kernel_vizinhos, 
                                         mode='constant', cval=0.0)
        
        # Calcula a CONTAGEM de vizinhos não-NaN para cada voxel
        contagem_vizinhos = ndimage.convolve(mapa_contagem, kernel_vizinhos, 
                                             mode='constant', cval=0.0)
        
        # Calcula a Média dos vizinhos
        # onde a contagem é 0, o resultado será NaN
        media_vizinhos = np.divide(soma_vizinhos, contagem_vizinhos, 
                                   out=np.full_like(soma_vizinhos, np.nan), 
                                   where=contagem_vizinhos != 0)

        # Identifica os voxels que devem ser preenchidos
        # Regra 1: Era NaN originalmente
        # Regra 2: contagem >= limiar
        voxels_para_preencher = (mascara_nan_original) & \
                                (contagem_vizinhos >= min_vizinhos_sinal)
        
        # Preenche os locais indicados com a média calculada dos seus vizinhos
        mapa_saida[..., t][voxels_para_preencher] = media_vizinhos[voxels_para_preencher]
    return mapa_saida