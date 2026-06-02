import numpy as np
import nibabel as nib

import pandas as pd
import os
def ordenar_imagens_por_fase(df, dwi_data):
    dwi_data = dwi_data[...,0:60]
    # 1. Selecionar apenas os volumes onde difusao == 1 para obter as fases válidas
    df_difusao = df[df["difusao"] == 1]
    fases_validas = df_difusao['fase_flag'].values
    # fases_validas = 1 - fases_validas

    # 2. Verificar se o número de fases válidas corresponde ao número de volumes.
    print(len(fases_validas), dwi_data.shape[3])
    if len(fases_validas) != dwi_data.shape[3]:
        
        raise ValueError(f"O número de fases válidas ({len(fases_validas)}) não corresponde ao número de volumes de imagem ({dwi_data.shape[3]}). Verifique o alinhamento dos seus dados.")

    # 3. Usar np.argsort para obter a ordem dos índices que classificariam as fases de forma crescente.
    # Esses índices (0, 1, 2, ...) correspondem à ordem original dos volumes no DWI_M.
    indices_ordenacao = np.argsort(fases_validas)

    # 4. Usar esses índices de ordenação para reordenar a última dimensão do array DWI_M.
    # Isso reordena os volumes do DWI_M de acordo com a ordem crescente das fases.
    dwi_ordenado = dwi_data[:, :, :, indices_ordenacao]

    return dwi_ordenado, fases_validas[indices_ordenacao], indices_ordenacao
df = pd.read_csv('/home/joao-oliveira/Documents/mapa-vasos/physiological_marked.csv')
data = nib.load("/home/joao-oliveira/Documents/mapa-vasos/dwi_corrected.nii.gz")
dwi_data = data.get_fdata()

dwi, _, _= ordenar_imagens_por_fase(df, dwi_data)

nib.save(nib.Nifti1Image(dwi, affine=data.affine, header=data.header), "/home/joao-oliveira/Documents/mapa-vasos/dwi.nii.gz")
import numpy as np
import nibabel as nib


def refine_spvs_mask(
    initial_mask_path,
    dwi_4d_path,
    output_mask_path="pvs_refined.nii.gz",
    corr_threshold=0.6,
    convergence_threshold=1.0,
    max_iterations=20,
    verbose=True,
):
    """
    Refina iterativamente uma máscara sPVS usando correlação temporal.

    Parâmetros
    ----------
    initial_mask_path : str
        Caminho para a máscara inicial 3D (ex: pvs01.nii.gz)

    dwi_4d_path : str
        Caminho para a série DWI 4D (ex: dwi_ordenada.nii.gz)

    output_mask_path : str
        Nome da máscara refinada de saída

    corr_threshold : float
        Limiar mínimo de correlação temporal

    convergence_threshold : float
        Critério de convergência em porcentagem (%)

    max_iterations : int
        Número máximo de iterações

    verbose : bool
        Exibe informações das iterações

    Retorna
    -------
    refined_mask : np.ndarray
        Máscara refinada binária 3D
    """

    # =========================
    # Carregamento dos dados
    # =========================
    mask_img = nib.load(initial_mask_path)
    dwi_img = nib.load(dwi_4d_path)

    mask = mask_img.get_fdata()
    dwi = dwi_img.get_fdata()

    # Garantir máscara binária
    mask = (mask > 0).astype(np.uint8)

    if dwi.ndim != 4:
        raise ValueError("A imagem DWI precisa ser 4D.")

    if mask.shape != dwi.shape[:3]:
        raise ValueError("Dimensões da máscara e DWI não coincidem.")

    previous_volume = np.sum(mask)

    if verbose:
        print(f"Volume inicial: {previous_volume} voxels")

    # ==================================================
    # Pré-processamento dos sinais temporais da DWI
    # ==================================================
    X, Y, Z, T = dwi.shape

    # Flatten espacial
    dwi_2d = dwi.reshape(-1, T)

    # Máscara flatten
    mask_flat = mask.flatten()

    # ==================================================
    # Loop iterativo
    # ==================================================
    for iteration in range(max_iterations):

        if verbose:
            print(f"\nIteração {iteration + 1}")

        # Voxels atuais da máscara
        current_voxels = dwi_2d[mask_flat > 0]

        if current_voxels.shape[0] == 0:
            print("Máscara vazia.")
            break

        # ==============================
        # Sinal médio global da máscara
        # ==============================
        global_signal = np.mean(current_voxels, axis=0)

        # Normalização do sinal global
        global_signal = (
            global_signal - np.mean(global_signal)
        ) / (np.std(global_signal) + 1e-8)

        # =========================================
        # Correlação voxel a voxel
        # =========================================
        voxel_signals = dwi_2d.copy()

        # Normalização temporal dos voxels
        voxel_means = np.mean(voxel_signals, axis=1, keepdims=True)
        voxel_stds = np.std(voxel_signals, axis=1, keepdims=True)

        voxel_signals_norm = (
            voxel_signals - voxel_means
        ) / (voxel_stds + 1e-8)

        # Correlação equivalente ao Pearson
        correlations = np.mean(
            voxel_signals_norm * global_signal,
            axis=1
        )

        # =========================================
        # Atualização da máscara
        # =========================================
        new_mask_flat = (correlations > corr_threshold).astype(np.uint8)

        # Opcional:
        # restringe crescimento apenas dentro da máscara inicial
        new_mask_flat *= (mask.flatten() > 0)

        # Reconstrói máscara 3D
        new_mask = new_mask_flat.reshape(X, Y, Z)

        # =========================================
        # Verifica convergência
        # =========================================
        current_volume = np.sum(new_mask)

        volume_change = (
            abs(current_volume - previous_volume)
            / (previous_volume + 1e-8)
        ) * 100

        if verbose:
            print(f"Volume atual: {current_volume} voxels")
            print(f"Variação percentual: {volume_change:.3f}%")

        # Critério de convergência
        if volume_change < convergence_threshold:
            if verbose:
                print("\nConvergência atingida.")
            mask = new_mask
            break

        # Atualiza para próxima iteração
        mask = new_mask
        mask_flat = mask.flatten()
        previous_volume = current_volume

    # =========================
    # Salvar resultado
    # =========================
    output_img = nib.Nifti1Image(
        mask.astype(np.uint8),
        affine=mask_img.affine,
        header=mask_img.header
    )

    nib.save(output_img, output_mask_path)

    if verbose:
        print(f"\nMáscara refinada salva em:")
        print(output_mask_path)

    return mask
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt


def refine_spvs_mask(
    initial_mask_path,
    dwi_4d_path,
    output_mask_path="pvs_refined.nii.gz",
    corr_threshold=0.6,
    convergence_threshold=1.0,
    max_iterations=20,
    plot_curves=True,
    verbose=True,
):
    """
    Refina iterativamente uma máscara sPVS usando correlação temporal
    e plota a curva média do sinal no início de cada iteração.
    """

    # =========================
    # Carregamento
    # =========================
    mask_img = nib.load(initial_mask_path)
    dwi_img = nib.load(dwi_4d_path)

    mask = mask_img.get_fdata()
    dwi = dwi_img.get_fdata()

    mask = (mask > 0).astype(np.uint8)

    if dwi.ndim != 4:
        raise ValueError("A DWI deve ser 4D.")

    if mask.shape != dwi.shape[:3]:
        raise ValueError("Dimensões incompatíveis.")

    X, Y, Z, T = dwi.shape

    dwi_2d = dwi.reshape(-1, T)
    mask_flat = mask.flatten()

    previous_volume = np.sum(mask)

    if verbose:
        print(f"Volume inicial: {previous_volume} voxels")

    # =========================================
    # Figura para acompanhar as iterações
    # =========================================
    if plot_curves:
        plt.figure(figsize=(10, 6))

    # =========================================
    # Loop iterativo
    # =========================================
    for iteration in range(max_iterations):

        if verbose:
            print(f"\nIteração {iteration + 1}")

        # Voxels atuais
        current_voxels = dwi_2d[mask_flat > 0]

        if current_voxels.shape[0] == 0:
            print("Máscara vazia.")
            break

        # =====================================
        # Sinal médio global
        # =====================================
        global_signal = np.mean(current_voxels, axis=0)

        # =====================================
        # Plot da curva média
        # =====================================
        if plot_curves:
            plt.plot(
                global_signal,
                linewidth=2,
                label=f"Iter {iteration + 1}"
            )

        # =====================================
        # Normalização
        # =====================================
        global_signal_norm = (
            global_signal - np.mean(global_signal)
        ) / (np.std(global_signal) + 1e-8)

        voxel_means = np.mean(dwi_2d, axis=1, keepdims=True)
        voxel_stds = np.std(dwi_2d, axis=1, keepdims=True)

        voxel_signals_norm = (
            dwi_2d - voxel_means
        ) / (voxel_stds + 1e-8)

        # =====================================
        # Correlação temporal
        # =====================================
        correlations = np.mean(
            voxel_signals_norm * global_signal_norm,
            axis=1
        )

        # =====================================
        # Atualiza máscara
        # =====================================
        new_mask_flat = (correlations > corr_threshold).astype(np.uint8)

        # Restringe à máscara inicial
        new_mask_flat *= (mask.flatten() > 0)

        new_mask = new_mask_flat.reshape(X, Y, Z)

        # =====================================
        # Convergência
        # =====================================
        current_volume = np.sum(new_mask)

        volume_change = (
            abs(current_volume - previous_volume)
            / (previous_volume + 1e-8)
        ) * 100

        if verbose:
            print(f"Volume atual: {current_volume}")
            print(f"Variação: {volume_change:.3f}%")

        if volume_change < convergence_threshold:
            if verbose:
                print("\nConvergência atingida.")

            mask = new_mask
            break

        mask = new_mask
        mask_flat = mask.flatten()
        previous_volume = current_volume

    # =========================================
    # Finaliza gráfico
    # =========================================
    if plot_curves:
        plt.xlabel("Tempo / Volume")
        plt.ylabel("Intensidade média")
        plt.title("Curva média da máscara por iteração")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    # =========================================
    # Salvar máscara final
    # =========================================
    output_img = nib.Nifti1Image(
        mask.astype(np.uint8),
        affine=mask_img.affine,
        header=mask_img.header
    )

    nib.save(output_img, output_mask_path)

    if verbose:
        print(f"\nMáscara salva em: {output_mask_path}")

    return mask


refined = refine_spvs_mask(
    initial_mask_path="/home/joao-oliveira/Documents/mapa-vasos/vessel_bin_05.nii.gz",
    dwi_4d_path="/home/joao-oliveira/Documents/mapa-vasos/dwi.nii.gz",
    output_mask_path="/home/joao-oliveira/Documents/mapa-vasos/pvs_refined.nii.gz",
    corr_threshold=0.4,
    convergence_threshold=1.0,
    plot_curves=True
)