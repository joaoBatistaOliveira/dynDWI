import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import pandas as pd
import nibabel as nib
import os
import os
import math
import numpy as np
import pandas as pd
import nibabel as nib
import matplotlib.pyplot as plt
from matplotlib import cm as cm
import matplotlib.colors as mcolors
from matplotlib.cm import get_cmap

import matplotlib.colors as mcolors
import matplotlib.gridspec as gridspec


def ordenar_imagens_por_fase(df, dwi_data):
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
    
def salvar_adc_ordenado(path_adc, df):
    img = nib.load(path_adc)
    header = img.header
    affine = img.affine
    data = img.get_fdata()
    adc_ordenado, fases_validas, indices_ordenacao = ordenar_imagens_por_fase(df, data)
    save_path = os.path.dirname(path_adc)
    nib.save(nib.Nifti1Image(adc_ordenado, affine=affine, header=header), os.path.join(save_path, "adc_ordenado.nii.gz"))

def gerar_gif(df, dwi_data, slice_idx=25, outfile="gif_ppu_fase.gif"):
    # Ordena imagens e fases
    adc_ordenado, fases_ordenadas, idx_ord = ordenar_imagens_por_fase(df, dwi_data)
    
    # Selecionar apenas os volumes com difusao == 1 para obter ppu e fases
    df_difusao = df[df["difusao"] == 1]
    valores_ppu = df_difusao['ppu'].values
    valores_ppu = valores_ppu[idx_ord]
    
    # adc_ordenado = suavizar_volumes_3d(adc_ordenado, sigma=1)
    
    # Repete os valores de fase e ppu 3 vezes
    n_cycles = 3
    fases_repetidas = np.concatenate([fases_ordenadas + k for k in range(n_cycles)])
    ppu_repetido = np.tile(valores_ppu, n_cycles)

    # Número de frames = n_cycles * número de fases
    frames = len(fases_repetidas)

    # Criar figura
    fig, axes = plt.subplots(2, 1, figsize=(8, 6), gridspec_kw={'height_ratios':[1,2]})
    plt.tight_layout()

    def update_frame(i):
        for ax in axes:
            ax.clear()

        # Painel superior: scatter do PPU repetido
        axes[0].scatter(fases_repetidas[:i], ppu_repetido[:i], color="black")   # pontos passados
        axes[0].scatter(fases_repetidas[i], ppu_repetido[i], color="red")       # ponto atual
        axes[0].set_xlim(0, n_cycles)  # eixo x de 0 a 3 ciclos
        # axes[0].set_xlabel("Ciclo cardíaco (fase)")
        # axes[0].set_ylabel("PPU")
        axes[0].set_title("pulsation")

        # Painel inferior: imagem correspondente
        # Cada ciclo deve reiniciar a sequência de imagens
        fase_idx = i % adc_ordenado.shape[3]  # reinicia a cada ciclo
        img = np.rot90(adc_ordenado[:, :, slice_idx, fase_idx], k=1)
        im = axes[1].imshow(img, cmap="inferno", vmin=0, vmax=0.0065)

        return im,

    ani = animation.FuncAnimation(fig, update_frame, frames=frames, interval=200, blit=False)
    ani.save(outfile, writer='pillow', fps=10)
    plt.close()

def gerar_mosaico_volumes(df, dwi_data, slice_idx=25, outfile="mosaico_volumes.png"):
    """
    Gera um mosaico com todos os volumes (subplots 6x10) mostrando uma fatia específica.
    Cada subplot mostra o índice original do volume e a fase relativa.
    """
    # Ordena imagens e fases
    adc_ordenado, fases_ordenadas, idx_ord = ordenar_imagens_por_fase(df, dwi_data)
    
    # Selecionar apenas os volumes com difusao == 1 para obter informações
    df_difusao = df[df["difusao"] == 1]
    
    # Número total de volumes
    n_volumes = adc_ordenado.shape[3]
    
    # Criar figura com subplots 6x10
    fig, axes = plt.subplots(6, 10, figsize=(20, 12))
    fig.suptitle(f'Mosaico de Volumes - Fatia {slice_idx}', fontsize=16, y=0.95)
    
    # Ajustar espaçamentos - mantemos pequenos entre imagens, mas subtítulos terão mais espaço
    plt.subplots_adjust(wspace=0.05, hspace=0.3)  # Aumentei hspace para dar mais espaço vertical aos subtítulos
    
    for i in range(n_volumes):
        # Calcular posição na grade 6x10
        row = i // 10
        col = i % 10
        
        # Obter índice original antes da ordenação
        idx_original = idx_ord[i]
        
        # Obter imagem (rotacionar 90 graus)
        img = np.rot90(adc_ordenado[:, :, slice_idx, i], k=1)
        
        # Plotar imagem
        im = axes[row, col].imshow(img, cmap="inferno", vmin=0, vmax=0.0065)
        
        # Adicionar título com índice original e fase - aumentando o padding
        axes[row, col].set_title(f'Vol {idx_original}\nFase: {fases_ordenadas[i]:.3f}', 
                               fontsize=8, pad=8)  # Aumentei o pad de 2 para 8
        
        # Remover eixos
        axes[row, col].set_xticks([])
        axes[row, col].set_yticks([])
        axes[row, col].axis('off')
    
    # Salvar figura
    plt.savefig(outfile, dpi=150, bbox_inches='tight', pad_inches=0.1)
    plt.close()
    
    print(f"Mosaico salvo como: {outfile}")
    return fig

def plotar_adc_medio_roi_ppu_global(path_base, subjects, direction, rois, escala = False, adc_var ='adc_ordenado'):
    """
    Plota curvas de ADC médio para múltiplos sujeitos e ROIs usando fase_flag como eixo X
    
    Parameters:
    path_base (str): Caminho base dos dados
    subjects (list): Lista de sujeitos
    direction (str): Direção dos dados
    rois (list): Lista de ROIs
    """
    
    # Calcular layout dos subplots (máximo 4 por linha)
    n_rois = len(rois)
    n_cols = min(4, n_rois)  # Máximo 4 colunas
    n_rows = math.ceil(n_rois / n_cols)
    
    # Configurar a figura com subplots organizados em grid
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
    
    # Se houver apenas uma ROI, transformar axes em array
    if n_rois == 1:
        axes = np.array([axes])
    
    # Achatar o array de axes para facilitar iteração
    axes_flat = axes.flatten() if n_rows > 1 or n_cols > 1 else [axes]
    
    # Dicionário para armazenar todos os dados
    all_data = {roi: [] for roi in rois}
    all_phase_flags = {roi: [] for roi in rois}  # Armazenar as phase_flags
    
    # Processar cada sujeito
    for subject in subjects:
        try:
            # Carregar dados do sujeito
            adc_path = os.path.join(path_base, subject, direction, f"Analysis/{adc}.nii.gz")
            adc_ordenado = nib.load(adc_path).get_fdata()
            b0_path = os.path.join(path_base, subject, "dynDWI_S/b0_brain_mask.nii.gz")
            b0 = nib.load(b0_path).get_fdata()
            
            # Carregar e processar o arquivo CSV com as fases
            csv_path = os.path.join(path_base, subject, direction, "Analysis/physiological_marked.csv")
            df = pd.read_csv(csv_path)
            
            # Filtrar difusao == 1 e ordenar por fase_flag
            df_filtrado = df[df['difusao'] == 1].sort_values('fase_flag')
            phase_flags_ordenadas = df_filtrado['fase_flag'].values
            
            # Verificar se o número de volumes coincide
            if len(phase_flags_ordenadas) != adc_ordenado.shape[3]:
                print(f"Aviso: Número de phase_flags ({len(phase_flags_ordenadas)}) não coincide com volumes ADC ({adc_ordenado.shape[3]}) para {subject}")
                # Usar o mínimo entre os dois
                n_volumes = min(len(phase_flags_ordenadas), adc_ordenado.shape[3])
                phase_flags_ordenadas = phase_flags_ordenadas[:n_volumes]
            else:
                n_volumes = adc_ordenado.shape[3]
            
            # Processar cada ROI para este sujeito
            for roi in rois:
                try:
                    # Carregar ROI
                    roi_path = os.path.join(path_base, subject, f"rois/{roi}.nii.gz")
                    roi_data = nib.load(roi_path).get_fdata()
                        # Verificar e ajustar as dimensões
                    # Criar máscara
                    mask = (roi_data * b0).astype(bool)
                    if adc_ordenado.shape[2] == 27 and mask.shape[2] == 30:
                        # Manter apenas as fatias de 2 a 28 (equivalente a 2:29 no indexing do Python)
                        mask = mask[:, :, 2:29]
                    
                    # Calcular ADC médio para cada volume temporal (já ordenado)
                    # e usar phase_flags_ordenadas como eixo X
                    adc_mean = []
                    for t in range(n_volumes):
                        volume_t = adc_ordenado[:, :, :, t]
                        voxels_na_mask = volume_t[mask]
                        if len(voxels_na_mask) > 0:
                            adc_mean.append(np.mean(voxels_na_mask))
                        else:
                            adc_mean.append(np.nan)
                    
                    all_data[roi].append(adc_mean)
                    all_phase_flags[roi].append(phase_flags_ordenadas)
                    
                except Exception as e:
                    print(f"Erro ao processar ROI {roi} para sujeito {subject}: {e}")
                    # Adicionar dados NaN
                    all_data[roi].append([np.nan] * n_volumes)
                    all_phase_flags[roi].append(phase_flags_ordenadas if 'phase_flags_ordenadas' in locals() else [np.nan] * n_volumes)
                    
        except Exception as e:
            print(f"Erro ao processar sujeito {subject}: {e}")
            # Adicionar dados vazios para todas as ROIs deste sujeito
            for roi in rois:
                all_data[roi].append([np.nan] * 60)
                all_phase_flags[roi].append([np.nan] * 60)
    
    # Plotar os dados
    for i, roi in enumerate(rois):
        if i < len(axes_flat):  # Garantir que não excedemos o número de subplots
            ax = axes_flat[i]
            roi_data = all_data[roi]
            roi_phase_flags = all_phase_flags[roi]
            
            # Encontrar phase_flags comuns para usar como eixo X
            # Usar as phase_flags do primeiro sujeito que tem dados válidos
            phase_flags_x = None
            for phase_flags in roi_phase_flags:
                if not np.all(np.isnan(phase_flags)) and len(phase_flags) > 0:
                    phase_flags_x = phase_flags
                    break
            
            # Se não encontrou phase_flags válidas, usar índice numérico
            if phase_flags_x is None:
                phase_flags_x = list(range(len(roi_data[0]) if roi_data else 0))
            
            # Converter dados para array numpy
            roi_array = np.array(roi_data)
            
            # Plotar cada sujeito individualmente
            for j, (subject_data, subject_phases) in enumerate(zip(roi_array, roi_phase_flags)):
                if not np.all(np.isnan(subject_data)) and not np.all(np.isnan(subject_phases)):
                    # Usar as phase_flags específicas de cada sujeito
                    ax.plot(subject_phases, subject_data, alpha=0.5, linewidth=1, label=f'S {j+1}')
            
            # Plotar média entre sujeitos (usando phase_flags_x como referência comum)
            if len(roi_array) > 0:
                mean_curve = np.nanmean(roi_array, axis=0)
                if not np.all(np.isnan(mean_curve)) and len(mean_curve) == len(phase_flags_x):
                    ax.plot(phase_flags_x, mean_curve, 'k-', linewidth=3, label='Média')
            
            ax.set_xlabel('Fase')
            ax.set_ylabel('Valor médio de ADC')
            if escala:
                ax.set_ylim((0.00065,0.008))
            ax.set_title(f'ROI: {roi}')
            ax.grid(True, alpha=0.3)
            
            # Adicionar legenda apenas se houver poucos sujeitos
            if len(subjects) <= 8:
                ax.legend(fontsize=8)
    
    # Ocultar subplots vazios se houver mais axes que ROIs
    for i in range(len(rois), len(axes_flat)):
        axes_flat[i].set_visible(False)
    
    plt.tight_layout()
    if escala:
        outfile = os.path.join(path_base, f"Analysis/adc_roi_global_ppu_{direction}_escala.png")
    else:
        outfile = os.path.join(path_base, f"Analysis/adc_roi_global_ppu_{direction}.png")
    plt.savefig(outfile, dpi=150, bbox_inches='tight', pad_inches=0.1)
    plt.show()
    plt.close()

def plotar_adc_medio_sujeito_rois_compacto(path_base, subjects, direction, rois, escala=False, adc_var ='adc_ordenado'):
    """
    Versão compacta - estilo similar à função plot_cc_global_subjects_zfmf
    Cada sujeito em subplot separado, todas ROIs juntas.
    """
    
    # Configurações estéticas compactas
    plt.rcParams.update({
        'font.size': 10,
        'axes.linewidth': 1.0,
        'lines.linewidth': 1.5,
        'lines.markersize': 4,
        'xtick.labelsize': 8,
        'ytick.labelsize': 8,
        'axes.titlesize': 11,
        'axes.labelsize': 10
    })
    
    n_subjects = len(subjects)
    n_cols = min(3, n_subjects)
    n_rows = math.ceil(n_subjects / n_cols)
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3*n_cols, 2.5*n_rows))
    
    # Flatten axes
    if n_subjects == 1:
        axes = [axes]
    elif n_rows > 1 and n_cols > 1:
        axes = axes.flatten()
    else:
        axes = [axes] if n_cols == 1 else list(axes)
    
    # Cores para ROIs
    n_rois = len(rois)
    colors = plt.cm.tab10(np.linspace(0, 1, n_rois))
    roi_colors = dict(zip(rois, colors))
    
    # Índices da última linha
    last_row_start = (n_rows - 1) * n_cols
    last_row_indices = range(last_row_start, last_row_start + n_cols)
    
    for subject_idx, subject in enumerate(subjects):
        if subject_idx >= len(axes):
            break
            
        ax = axes[subject_idx]
        
        try:
            # [CÓDIGO DE CARREGAMENTO IDÊNTICO AO DA FUNÇÃO ANTERIOR]
            adc_path = os.path.join(path_base, subject, direction, f"Analysis/{adc_var}.nii.gz")
            adc_ordenado = nib.load(adc_path).get_fdata()
            b0_path = os.path.join(path_base, subject, "dynDWI_S/b0_brain_mask.nii.gz")
            b0 = nib.load(b0_path).get_fdata()
            
            csv_path = os.path.join(path_base, subject, direction, "Analysis/physiological_marked.csv")
            df = pd.read_csv(csv_path)
            df_filtrado = df[df['difusao'] == 1].sort_values('fase_flag')
            phase_flags_ordenadas = df_filtrado['fase_flag'].values
            
            if len(phase_flags_ordenadas) != adc_ordenado.shape[3]:
                n_volumes = min(len(phase_flags_ordenadas), adc_ordenado.shape[3])
                phase_flags_ordenadas = phase_flags_ordenadas[:n_volumes]
            else:
                n_volumes = adc_ordenado.shape[3]
            
            # Processar ROIs
            for roi in rois:
                try:
                    roi_path = os.path.join(path_base, subject, f"rois/{roi}.nii.gz")
                    roi_data = nib.load(roi_path).get_fdata()
                    mask = (roi_data * b0).astype(bool)
                    
                    if adc_ordenado.shape[2] == 27 and mask.shape[2] == 30:
                        mask = mask[:, :, 2:29]
                    
                    adc_mean = []
                    for t in range(n_volumes):
                        volume_t = adc_ordenado[:, :, :, t]
                        voxels_na_mask = volume_t[mask]
                        if len(voxels_na_mask) > 0:
                            adc_mean.append(np.mean(voxels_na_mask))
                        else:
                            adc_mean.append(np.nan)
                    
                    color = roi_colors[roi]
                    ax.plot(phase_flags_ordenadas, adc_mean, 
                           color=color, alpha=0.8, linewidth=1.5, label=roi)
                    
                except Exception as e:
                    print(f"Erro na ROI {roi} para {subject}: {e}")
            
            # Configurações compactas
            ax.set_title(f'{subject}', fontsize=11, fontweight='bold', pad=8)
            
            # Eixo X apenas na última linha
            if subject_idx in last_row_indices:
                ax.set_xlabel('Fase', fontsize=10, fontweight='bold')
            else:
                ax.set_xlabel('')
                ax.tick_params(axis='x', labelbottom=False)
            
            # Eixo Y apenas na primeira coluna
            if subject_idx % n_cols == 0:
                ax.set_ylabel('ADC Médio', fontsize=10, fontweight='bold')
            else:
                ax.set_ylabel('')
                ax.tick_params(axis='y', labelleft=False)
            
            if escala:
                ax.set_ylim((0.00065, 0.0008))
            
            ax.grid(True, alpha=0.2)
            ax.axhline(y=0, color='gray', linestyle='-', alpha=0.3, linewidth=0.5)
            
            # Legenda compacta
            if n_rois <= 6 and subject_idx == 0:  # Legenda apenas no primeiro
                ax.legend(fontsize=7, loc='upper right', framealpha=0.9)
                
        except Exception as e:
            print(f"Erro no sujeito {subject}: {e}")
            ax.text(0.5, 0.5, 'Erro', ha='center', va='center', 
                   transform=ax.transAxes, fontsize=9)
            ax.set_title(f'{subject}', fontsize=11, fontweight='bold', pad=8)
    
    # Ocultar eixos vazios
    for idx in range(len(subjects), len(axes)):
        axes[idx].set_visible(False)
    
    plt.tight_layout()
    
    # Salvar
    suffix = "_escala" if escala else ""
    outfile = os.path.join(path_base, f"Analysis/adc_sujeito_rois_compacto_{direction}{suffix}.png")
    plt.savefig(outfile, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Figura compacta salva em: {outfile}")
    plt.show()
    plt.close()

#Analise por fase
def setup_output_directory(base_dir, output_folder="analises_ordenamento"):
    """Cria e retorna o diretório de saída."""
    output_dir = os.path.join(base_dir, output_folder)
    os.makedirs(output_dir, exist_ok=True)
    return output_dir

def get_roi_mask_path(base_dir, subject, direction, roi):
    """Retorna o caminho para o arquivo da máscara ROI."""
    return os.path.join(base_dir, subject, "rois", f"{roi}.nii.gz")

def extract_roi_timeseries(adc_img_data, mask_data):
    """
    Extrai a série temporal média do ADC de uma imagem 4D usando uma máscara 3D.
    """
    # Ajusta a máscara se necessário
    if adc_img_data.shape[2] == 27:
        mask_data = mask_data[:, :, 2:29]
    
    mask_bool = mask_data.astype(bool)
    n_volumes = adc_img_data.shape[3]
    adc_timeseries = np.zeros(n_volumes)
    
    for t in range(n_volumes):
        volume_t = adc_img_data[..., t]
        adc_timeseries[t] = np.nanmean(volume_t[mask_bool])
        
    return adc_timeseries

def calculate_stats_expanded(values):
    """
    Calcula métricas estatísticas expandidas: média, mediana, std, CV, amplitude e entropia.
    """
    values_clean = values[~np.isnan(values)]
    
    if len(values_clean) == 0:
        return np.nan, np.nan, np.nan, np.nan, np.nan, np.nan

    # Métricas básicas
    mean = np.mean(values_clean)
    median = np.median(values_clean)
    std_dev = np.std(values_clean)

    # Coeficiente de Variação
    cv = np.nan if mean == 0 else std_dev / mean
        
    # Amplitude
    amplitude = 0.0 if len(values_clean) <= 1 else np.max(values_clean) - np.min(values_clean)

    # Entropia de Shannon
    if len(np.unique(values_clean)) <= 1:
        entropy = 0.0
    else:
        counts, _ = np.histogram(values_clean, bins=10)
        counts = counts[counts > 0]
        
        if len(counts) == 0:
            entropy = 0.0
        else:
            probabilities = counts / np.sum(counts)
            entropy = -np.sum(probabilities * np.log2(probabilities))

    return mean, median, std_dev, cv, amplitude, entropy

def load_adc_and_physio_data(adc_path, csv_path):
    """Carrega os dados ADC e fisiológicos."""
    if not os.path.exists(adc_path) or not os.path.exists(csv_path):
        return None, None, "Arquivos não encontrados"
    
    try:
        adc_img = nib.load(adc_path)
        adc_data_4d = adc_img.get_fdata()
        physio_df = pd.read_csv(csv_path)
        return adc_data_4d, physio_df, None
    except Exception as e:
        return None, None, f"Erro ao carregar dados: {e}"

def synchronize_adc_physio_data(adc_data_4d, physio_df, subject, direction):
    """Sincroniza dados ADC com dados fisiológicos."""
    # Filtra pontos de tempo da aquisição do ADC
    adc_physio_df = physio_df[physio_df['difusao'] == 1].copy()
    
    # Verificação de consistência
    n_volumes_adc = adc_data_4d.shape[3]
    n_volumes_csv = len(adc_physio_df)
    
    if n_volumes_adc != n_volumes_csv:
        return None, f"Inconsistência de volumes: ADC={n_volumes_adc}, CSV={n_volumes_csv}"
    
    if n_volumes_adc != 60 or n_volumes_csv != 60:
        return None, f"Número de volumes diferente de 60: ADC={n_volumes_adc}, CSV={n_volumes_csv}"
    
    return adc_physio_df, None

def process_roi_for_subject_direction(subject, direction, adc_data_4d, adc_physio_df, base_dir):
    """Processa todas as ROIs para um sujeito e direção específicos."""
    roi_results = []
    
    for roi in ROIS:
        roi_mask_path = get_roi_mask_path(base_dir, subject, direction, roi)
        
        if not os.path.exists(roi_mask_path):
            continue
            
        try:
            roi_mask_img = nib.load(roi_mask_path)
            roi_mask_data = roi_mask_img.get_fdata()
        except Exception as e:
            continue

        # Extrai série temporal da ROI
        adc_timeseries_roi = extract_roi_timeseries(adc_data_4d, roi_mask_data)
        
        # Prepara dados para processamento
        adc_physio_df_reset = adc_physio_df.reset_index(drop=True)
        adc_timeseries_roi_series = pd.Series(adc_timeseries_roi, name="adc_value")
        data_to_process = pd.concat([adc_physio_df_reset, adc_timeseries_roi_series], axis=1)

        # Processa tasks
        task_results = process_tasks(data_to_process, subject, direction, roi)
        roi_results.extend(task_results)
    
    return roi_results

def process_tasks(data_to_process, subject, direction, roi):
    """Processa dados separados por task (Respiração Livre vs Lenta)."""
    tasks = {
        0: "Respiracao Livre",
        1: "Respiracao Lenta"
    }
    
    task_results = []
    
    for task_id, task_name in tasks.items():
        df_task = data_to_process[data_to_process['task'] == task_id].copy()
        
        if df_task.empty:
            continue
        
        # Processa intervalos de fase
        phase_results = process_phase_intervals(df_task, subject, direction, roi, task_name, task_id)
        task_results.extend(phase_results)
    
    return task_results

def process_phase_intervals(df_task, subject, direction, roi, task_name, task_id):
    """Processa os intervalos de fase para uma task específica."""
    df_task_sorted = df_task.sort_values(by='fase_flag', ascending=True)
    phase_results = []
    
    for interval_name, (start_phase, end_phase) in PHASE_INTERVALS:
        # Seleciona dados do intervalo de fase
        if end_phase == 1.0:
            mask_phase = (df_task_sorted['fase_flag'] >= start_phase) & \
                         (df_task_sorted['fase_flag'] <= end_phase)
        else:
            mask_phase = (df_task_sorted['fase_flag'] >= start_phase) & \
                         (df_task_sorted['fase_flag'] < end_phase)
        
        df_phase_bin = df_task_sorted[mask_phase]
        adc_values_in_bin = df_phase_bin['adc_value'].values
        
        # Calcula estatísticas
        mean_adc, median_adc, std_adc, cv_adc, amplitude_adc, entropy_adc = \
            calculate_stats_expanded(adc_values_in_bin)
        
        # Armazena resultados
        result_entry = {
            "sujeito": subject,
            "direcao": direction,
            "roi": roi,
            "task": task_name,
            "task_id": task_id,
            "intervalo_de_fase": interval_name,
            "fase_inicio": start_phase,
            "fase_fim": end_phase,
            "n_volumes": len(adc_values_in_bin),
            "media_adc": mean_adc,
            "mediana_adc": median_adc,
            "variabilidade_adc": std_adc,
            "CV": cv_adc,
            "amplitude": amplitude_adc,
            "entropy": entropy_adc
        }
        phase_results.append(result_entry)
    
    return phase_results

def process_single_subject_direction(subject, direction, base_dir):
    """Processa um único sujeito e direção."""
    # Caminhos dos arquivos
    adc_path = os.path.join(base_dir, subject, direction, "Analysis/adc.nii.gz")
    csv_path = os.path.join(base_dir, subject, direction, "Analysis/physiological_marked.csv")
    
    # Carrega dados
    adc_data_4d, physio_df, error = load_adc_and_physio_data(adc_path, csv_path)
    if error:
        return None, f"{subject}/{direction}: {error}"
    
    # Sincroniza dados
    adc_physio_df, error = synchronize_adc_physio_data(adc_data_4d, physio_df, subject, direction)
    if error:
        return None, f"{subject}/{direction}: {error}"
    
    # Processa ROIs
    results = process_roi_for_subject_direction(subject, direction, adc_data_4d, adc_physio_df, base_dir)
    return results, None

def main():
    """Função principal que executa toda a análise."""
    print("Iniciando análise organizada...")
    
    # Configura diretório de saída
    OUTPUT_DIR = setup_output_directory(BASE_DIR)
    OUTPUT_CSV = os.path.join(OUTPUT_DIR, "analise_adc_por_fase_2.csv")
    
    all_results = []
    
    # Processa todos os sujeitos e direções
    for subject in SUBJECTS:
        print(f"\nProcessando Sujeito: {subject}")
        for direction in DIRECTIONS:
            print(f"  Direção: {direction}")
            
            results, error = process_single_subject_direction(subject, direction, BASE_DIR)
            
            if error:
                print(f"    AVISO: {error}")
                continue
                
            if results:
                all_results.extend(results)
    
    # Salva resultados
    if all_results:
        final_results_df = pd.DataFrame(all_results)
        try:
            final_results_df.to_csv(OUTPUT_CSV, index=False, float_format='%.6f', encoding='utf-8')
            print(f"\nAnálise concluída com sucesso!")
            print(f"Resultados salvos em: {OUTPUT_CSV}")
            print(f"Total de registros processados: {len(all_results)}")
        except Exception as e:
            print(f"ERRO AO SALVAR ARQUIVO CSV: {e}")
    else:
        print("\nNenhum dado foi processado. Verifique os arquivos de entrada.")


# --- CONSTANTES E CONFIGURAÇÕES ---

BASE_DIR = "/media/joao/PortableSSD/dynDWI"
SUBJECTS = ["sub008", "sub009", "sub010", "sub011", "sub012", "sub013", "sub014", 
           "sub015", "sub016", "sub017", "sub018"]
DIRECTIONS = ["dynDWI_M", "dynDWI_S", "dynDWI_P"]
ROIS = ["CSF", "WM", "GM", "3V", "VL", "PC", "4V"]

PHASE_INTERVALS = [
    ("I1", [0, 0.35]),
    ("I2", [0.35, 1]),
]
main()