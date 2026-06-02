import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import nibabel as nib
import os
import json
import math
from scipy.signal import find_peaks
from matplotlib.cm import ScalarMappable
import matplotlib.colors as mcolors
from matplotlib import colormaps
import matplotlib.cm as cm
import physiological_marked

def substituir_none_por_nan(data):
    """
    Converte valores None para NaN em toda a estrutura de forma recursiva.
    """
    if isinstance(data, dict):
        result = {}
        for key, value in data.items():
            if isinstance(value, dict):
                result[key] = substituir_none_por_nan(value)
            elif isinstance(value, list):
                result[key] = [np.nan if x is None else x for x in value]
            else:
                result[key] = np.nan if value is None else value
        return result
    elif isinstance(data, list):
        return [np.nan if x is None else x for x in data]
    else:
        return np.nan if data is None else data

def substituir_nan_por_none(obj):
    """
    Converte valores NaN para None em toda a estrutura de forma recursiva.
    """
    if isinstance(obj, dict):
        return {k: substituir_nan_por_none(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [substituir_nan_por_none(v) for v in obj]
    elif isinstance(obj, float) and np.isnan(obj):
        return None
    else:
        return obj

def _interpolate_nans(signal):
    """
    Função auxiliar para preencher NaNs por interpolação linear.
    """
    nan_mask = np.isnan(signal)
    if not nan_mask.any():
        return signal

    signal_clean = signal.copy()
    valid_indices = np.where(~nan_mask)[0]
    nan_indices = np.where(nan_mask)[0]

    if len(valid_indices) == 0:
        signal_clean[:] = 0
        return signal_clean

    # Interpolar NaNs no meio do sinal
    mid_nans = nan_indices[(nan_indices > valid_indices[0]) & (nan_indices < valid_indices[-1])]
    if len(mid_nans) > 0:
        signal_clean[mid_nans] = np.interp(mid_nans, valid_indices, signal_clean[valid_indices])

    # Lidar com NaNs nas extremidades
    if np.isnan(signal_clean[0]):
        signal_clean[:valid_indices[0]] = signal_clean[valid_indices[0]]
            
    if np.isnan(signal_clean[-1]):
        signal_clean[valid_indices[-1]:] = signal_clean[valid_indices[-1]]

    return signal_clean

def fft_lowpass_filter(signal, sample_interval, cutoff_hz):
    """
    Aplica um filtro passa-baixa ideal no domínio da frequência (FFT).
    """
    N = len(signal)
    if N == 0 or sample_interval <= 0:
        return signal

    # Salva a máscara de NaNs originais
    original_nans_mask = np.isnan(signal)

    # Limpa os NaNs por interpolação
    signal_cleaned = _interpolate_nans(signal)

    # Calcular a FFT do sinal limpo
    signal_fft = np.fft.fft(signal_cleaned)
    
    # Calcular o vetor de frequências
    frequencies = np.fft.fftfreq(N, d=sample_interval)
    
    # Criar a máscara passa-baixa
    mask = np.abs(frequencies) > cutoff_hz
    
    # Aplicar a máscara
    signal_fft[mask] = 0
    
    # Calcular a IFFT
    signal_filtered_real = np.real(np.fft.ifft(signal_fft))
    
    # Reaplica os NaNs originais
    signal_filtered_real[original_nans_mask] = np.nan

    return signal_filtered_real

def calcular_correlacao_fisiologica(
    adc_series, df_dados_M, Tempo, 
    atraso_min=-3, atraso_max=3, passo=0.1, 
    respiracao_col=None, pulsacao_col=None, zscore=True,
    cutoff_ppu_hz=5.0, cutoff_resp_hz=1.0):
    """
    Calcula a correlação cruzada entre a série ADC e os sinais fisiológicos (PPU, Resp).
    
    Parâmetros:
    -----------
    adc_series : array
        Série temporal do ADC
    df_dados_M : DataFrame
        DataFrame com dados fisiológicos
    Tempo : str
        Nome da coluna que indica tempos de difusão
    atraso_min, atraso_max, passo : float
        Parâmetros para os atrasos
    respiracao_col : str, optional
        Coluna com respiração suavizada
    pulsacao_col : str, optional
        Coluna com pulsação suavizada
    zscore : bool
        Se True, aplica normalização Z-score no sinal ADC
    cutoff_ppu_hz, cutoff_resp_hz : float
        Frequências de corte para filtro FFT
    
    Retorna:
    --------
    tuple: (correlacoes_ppu, correlacoes_resp)
    """
    
    # Normalização Z-score do sinal ADC
    if zscore:
        adc_std = adc_series.std()
        if adc_std > 0:
            adc_series = (adc_series - adc_series.mean()) / adc_std
        else:
            adc_series = adc_series - adc_series.mean()
            
    # Verificar correspondência de tempos
    tempos_difusao = df_dados_M[df_dados_M[Tempo] == 1]['Time'].values
    if len(tempos_difusao) != len(adc_series):
        raise ValueError("O número de instantes de difusão não corresponde ao tamanho da série ADC.")

    # Extração e Filtragem dos Sinais Fisiológicos    
    if respiracao_col is None:
        resp_signal = df_dados_M['resp'].values.copy()
    else:
        resp_signal = df_dados_M[respiracao_col].values.copy()

    if pulsacao_col is None:
        ppu_signal = df_dados_M['ppu'].values.copy()
    else:
        ppu_signal = df_dados_M[pulsacao_col].values.copy()
        
    fisio_times = df_dados_M['Time'].values

    # Aplicar filtragem FFT se cutoffs forem fornecidos
    if len(fisio_times) >= 2:
        sample_interval = np.mean(np.diff(fisio_times))
        
        if sample_interval > 0:
            if cutoff_ppu_hz is not None:
                ppu_signal = fft_lowpass_filter(ppu_signal, sample_interval, cutoff_ppu_hz)
            
            if cutoff_resp_hz is not None:
                resp_signal = fft_lowpass_filter(resp_signal, sample_interval, cutoff_resp_hz)

    # 4. Cálculo da Correlação Cruzada
    atrasos = np.arange(atraso_min, atraso_max + passo, passo)
    correlacoes_ppu = []
    correlacoes_resp = []

    for atraso in atrasos:
        tempos_difusao_deslocados = tempos_difusao + atraso
        # Interpola os sinais fisiológicos
        ppu_interp = np.interp(tempos_difusao_deslocados, fisio_times, ppu_signal,
                               left=np.nan, right=np.nan)
        resp_interp = np.interp(tempos_difusao_deslocados, fisio_times, resp_signal,
                                left=np.nan, right=np.nan)
        valid_idx_ppu = ~np.isnan(ppu_interp)
        valid_idx_resp = ~np.isnan(resp_interp)

        # Calcular correlação com mínimo de 10 pontos válidos
        if np.sum(valid_idx_ppu) > 10: 
            corr_ppu = np.corrcoef(adc_series[valid_idx_ppu], ppu_interp[valid_idx_ppu])[0, 1]
            correlacoes_ppu.append(corr_ppu)
        else:
            correlacoes_ppu.append(np.nan)

        if np.sum(valid_idx_resp) > 10:
            corr_resp = np.corrcoef(adc_series[valid_idx_resp], resp_interp[valid_idx_resp])[0, 1]
            correlacoes_resp.append(corr_resp)
        else:
            correlacoes_resp.append(np.nan)

    return np.array(correlacoes_ppu), np.array(correlacoes_resp)

def calcular_correlacao_controle_negativo(
    tamanho_adc, df_dados_M, Tempo,
    n_simulacoes=1000,
    atraso_min=-3, atraso_max=3, passo=0.1, 
    respiracao_col=None, pulsacao_col=None, zscore=True,
    cutoff_ppu_hz=5.5, cutoff_resp_hz=1.0):
    """
    Calcula a correlação cruzada média de N simulações de ruído branco.
    """
    
    print(f"Calculando controle negativo ({n_simulacoes} simulações)...")
    
    # Pré-cálculo dos sinais fisiológicos (mesma lógica da função principal)
    tempos_difusao = df_dados_M[df_dados_M[Tempo] == 1]['Time'].values

    if len(tempos_difusao) != tamanho_adc:
        raise ValueError(f"Tamanho do ADC ({tamanho_adc}) não corresponde aos tempos de difusão ({len(tempos_difusao)})")

    if respiracao_col is None:
        resp_signal = df_dados_M['resp'].values.copy()
    else:
        resp_signal = df_dados_M[respiracao_col].values.copy()
    
    if pulsacao_col is None:
        ppu_signal = df_dados_M['resp'].values.copy()
    else:
        ppu_signal = df_dados_M[pulsacao_col].values.copy()
        
    fisio_times = df_dados_M['Time'].values

    # Aplicar filtragem
    if len(fisio_times) >= 2:
        sample_interval = np.mean(np.diff(fisio_times))
        if sample_interval > 0:
            if cutoff_ppu_hz is not None:
                ppu_signal = fft_lowpass_filter(ppu_signal, sample_interval, cutoff_ppu_hz)
            if cutoff_resp_hz is not None:
                resp_signal = fft_lowpass_filter(resp_signal, sample_interval, cutoff_resp_hz)

    # Pré-calcular interpolações
    atrasos = np.arange(atraso_min, atraso_max + passo, passo)
    ppu_interp_list = []
    resp_interp_list = []
    valid_idx_ppu_list = []
    valid_idx_resp_list = []

    for atraso in atrasos:
        tempos_difusao_deslocados = tempos_difusao + atraso

        ppu_interp = np.interp(tempos_difusao_deslocados, fisio_times, ppu_signal,
                              left=np.nan, right=np.nan)
        resp_interp = np.interp(tempos_difusao_deslocados, fisio_times, resp_signal,
                               left=np.nan, right=np.nan)

        ppu_interp_list.append(ppu_interp)
        resp_interp_list.append(resp_interp)
        valid_idx_ppu_list.append(~np.isnan(ppu_interp))
        valid_idx_resp_list.append(~np.isnan(resp_interp))

    # Simulações de ruído
    all_curves_ppu = []
    all_curves_resp = []

    for i in range(n_simulacoes):
        fake_adc = np.random.normal(size=tamanho_adc)
        
        if zscore:
            fake_adc_std = fake_adc.std()
            if fake_adc_std > 0:
                fake_adc = (fake_adc - fake_adc.mean()) / fake_adc_std
            else:
                fake_adc = fake_adc - fake_adc.mean()
        
        sim_corr_ppu = []
        sim_corr_resp = []

        for j in range(len(atrasos)):
            # PPU
            valid_idx_ppu = valid_idx_ppu_list[j]
            if np.sum(valid_idx_ppu) > 10:
                ppu_interp = ppu_interp_list[j]
                corr_ppu = np.corrcoef(fake_adc[valid_idx_ppu], ppu_interp[valid_idx_ppu])[0, 1]
                sim_corr_ppu.append(corr_ppu)
            else:
                sim_corr_ppu.append(np.nan)

            # Resp
            valid_idx_resp = valid_idx_resp_list[j]
            if np.sum(valid_idx_resp) > 10:
                resp_interp = resp_interp_list[j]
                corr_resp = np.corrcoef(fake_adc[valid_idx_resp], resp_interp[valid_idx_resp])[0, 1]
                sim_corr_resp.append(corr_resp)
            else:
                sim_corr_resp.append(np.nan)
        
        all_curves_ppu.append(sim_corr_ppu)
        all_curves_resp.append(sim_corr_resp)

    # Calcular médias
    media_ppu = np.nanmean(np.array(all_curves_ppu), axis=0)
    media_resp = np.nanmean(np.array(all_curves_resp), axis=0)
    
    return atrasos, media_ppu, media_resp

def processar_correlacao_adc(
    adc, df_dados, mask, brain, path_output_json,
    atraso_min=-3, atraso_max=3, passo=0.1, 
    respiracao_col=None, pulsacao_col=None, zscore=True,
    cutoff_ppu_hz=5.0, cutoff_resp_hz=1.0,
    n_simulacoes_controle=1000):
    """
    Processa correlação entre ADC e sinais fisiológicos para cada fatia e volume global.
    
    Parâmetros:
    -----------
    adc : array 4D
        Dados ADC (x, y, z, t)
    df_dados : DataFrame
        Dados fisiológicos
    mask : array 3D
        Máscara da ROI
    brain : array 3D  
        Máscara cerebral
    path_output_json : str
        Caminho para salvar resultados
    n_simulacoes_controle : int
        Número de simulações para controle negativo
    
    Retorna:
    --------
    dict: Dicionário com resultados
    """

    mask_combined = (mask * brain).astype(bool)
    adc_masked = adc.astype(float).copy()
    adc_masked[~mask_combined, :] = np.nan
    atrasos = np.arange(atraso_min, atraso_max + passo, passo)
    resultados_json = {}
    x_dim, y_dim, z_dim, t_dim = adc.shape
    
    # Calcular correlação para volume global
    print("Calculando correlação para volume global...")
    adc_mean_global = np.nanmean(adc_masked, axis=(0, 1, 2))
    corrs_ppu_global, corrs_resp_global = calcular_correlacao_fisiologica(
        adc_mean_global, df_dados, Tempo='difusao',
        atraso_min=atraso_min, atraso_max=atraso_max, passo=passo,
        respiracao_col=respiracao_col, pulsacao_col=pulsacao_col, zscore=zscore,
        cutoff_ppu_hz=cutoff_ppu_hz, cutoff_resp_hz=cutoff_resp_hz
    )
    
    # Calcular controle negativo
    print("Calculando controle negativo...")
    atrasos_controle, controle_ppu, controle_resp = calcular_correlacao_controle_negativo(
        len(adc_mean_global), df_dados, Tempo='difusao',
        n_simulacoes=n_simulacoes_controle,
        atraso_min=atraso_min, atraso_max=atraso_max, passo=passo,
        respiracao_col=respiracao_col, pulsacao_col=pulsacao_col, zscore=zscore,
        cutoff_ppu_hz=cutoff_ppu_hz, cutoff_resp_hz=cutoff_resp_hz
    )
    
    resultados_json['global'] = {
        'atrasos': atrasos.tolist(),
        'ppu': corrs_ppu_global.tolist(),
        'resp': corrs_resp_global.tolist()
    }
    
    resultados_json['controle_negativo'] = {
        'atrasos': atrasos_controle.tolist(),
        'ppu': controle_ppu.tolist(),
        'resp': controle_resp.tolist()
    }
    
    # Calcular correlação para cada fatia
    print(f"Calculando correlação para {z_dim} fatias...")
    for z in range(z_dim):
        adc_mean_slice = np.nanmean(adc_masked[:, :, z, :], axis=(0, 1))
        
        if np.all(np.isnan(adc_mean_slice)):
            corrs_ppu = np.full_like(atrasos, np.nan)
            corrs_resp = np.full_like(atrasos, np.nan)
        else:
            corrs_ppu, corrs_resp = calcular_correlacao_fisiologica(
                adc_mean_slice, df_dados, Tempo='difusao',
                atraso_min=atraso_min, atraso_max=atraso_max, passo=passo,
                respiracao_col=respiracao_col, pulsacao_col=pulsacao_col, zscore=zscore,
                cutoff_ppu_hz=cutoff_ppu_hz, cutoff_resp_hz=cutoff_resp_hz
            )

        resultados_json[f'slice_{z}'] = {
            'atrasos': atrasos.tolist(),
            'ppu': corrs_ppu.tolist(),
            'resp': corrs_resp.tolist()
        }

    with open(path_output_json, 'w') as f:
        json.dump(substituir_nan_por_none(resultados_json), f, indent=4)
    
    print(f"Dados de correlação salvos em {path_output_json}")
    return resultados_json

def plotar_curvas_cc_slice_roi(json_file, save_file_path=None):

    if isinstance(json_file, str):
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)
        except Exception as e:
            print(f"Erro ao carregar JSON do arquivo {json_file}: {e}")
            return None
    
    elif isinstance(json_file, dict):
        data = json_file
    
    else:
        print(f"Tipo de entrada não suportado: {type(json_file)}")
        return None

    data = substituir_none_por_nan(data)
    slice_keys = [k for k in data if k.startswith("slice_")]
    global_data = data.get("global", {})
    ctrl_data = data.get("controle_negativo", {})
    atrasos_seg = np.array(data[slice_keys[0]]['atrasos'])
    all_resp = np.array([np.array(data[k]["resp"]) for k in slice_keys])
    all_ppu  = np.array([np.array(data[k]["ppu"])  for k in slice_keys])

    # Média e DP
    mean_resp = np.nanmean(all_resp, axis=0)
    std_resp  = np.nanstd(all_resp,  axis=0)
    mean_ppu  = np.nanmean(all_ppu,  axis=0)
    std_ppu   = np.nanstd(all_ppu,   axis=0)

    # Globais
    g_resp = np.array(global_data.get("resp", []))
    g_ppu  = np.array(global_data.get("ppu", []))
    c_resp = np.array(ctrl_data.get("resp", []))
    c_ppu  = np.array(ctrl_data.get("ppu", []))

    fig, axes = plt.subplots(2, 2, figsize=(14, 10), constrained_layout=True)
    fig.suptitle("Cross Correlation Curve", fontsize=16, weight="bold")
    cmap = colormaps.get_cmap("managua")
    colors = cmap(np.linspace(0, 1, len(slice_keys)))

    ax = axes[0, 0]
    ax.set_title("Breathing – Curves per slice", fontsize=13, weight="bold")

    # Curvas individuais
    for i, curve in enumerate(all_resp):
        ax.plot(atrasos_seg, curve, color=colors[i], alpha=0.9)

    # Média ± DP
    ax.fill_between(atrasos_seg, mean_resp - std_resp, mean_resp + std_resp,
                    color="black", alpha=0.4)
    ax.plot(atrasos_seg, mean_resp, color="black", linewidth=2.5, label="Média")

    # Máximo da média
    idx = np.nanargmax(np.abs(mean_resp))
    ax.plot(atrasos_seg[idx], mean_resp[idx], "ko", markersize=8)

    ax.text(0.02, 0.95,
            f"Pico média: {mean_resp[idx]:.3f}\nAtraso: {atrasos_seg[idx]:.1f}s",
            transform=ax.transAxes, fontsize=10,
            verticalalignment="top",
            bbox=dict(facecolor="white", alpha=0.8))

    ax.axhline(0, color="gray", linestyle="--", alpha=0.7)
    ax.axvline(0, color="gray", linestyle="--", alpha=0.7)
    ax.grid(True, linestyle=":")

    ax = axes[0, 1]
    ax.set_title("Breathing – Global and Control", fontsize=13, weight="bold")
    L1 = ax.plot(atrasos_seg, g_resp, color="black", linewidth=2.5, label="Global")
    L2 = ax.plot(atrasos_seg, c_resp, color="gray", linestyle="--", linewidth=2, label="Controle negativo")
    if len(g_resp) > 0:
        i0 = np.nanargmax(np.abs(g_resp))
        ax.plot(atrasos_seg[i0], g_resp[i0], "ko", markersize=8)

    ax.legend(
        title=f"Pico = {g_resp[i0]:.3f}\nAtraso = {atrasos_seg[i0]:.1f}s",
        fontsize=10,
        loc="upper left"
    )

    ax.axhline(0, color="gray", linestyle="--")
    ax.grid(True, linestyle=":")

    ax = axes[1, 0]
    ax.set_title("Pulse – Curves per slice", fontsize=13, weight="bold")

    for i, curve in enumerate(all_ppu):
        ax.plot(atrasos_seg, curve, color=colors[i], alpha=0.9)

    ax.fill_between(atrasos_seg, mean_ppu - std_ppu, mean_ppu + std_ppu,
                    color="black", alpha=0.4)
    ax.plot(atrasos_seg, mean_ppu, color="black", linewidth=2.5)

    idx = np.nanargmax(np.abs(mean_ppu))
    ax.plot(atrasos_seg[idx], mean_ppu[idx], "ko", markersize=8)

    ax.text(0.02, 0.95,
            f"Pico média: {mean_ppu[idx]:.3f}\nAtraso: {atrasos_seg[idx]:.1f}s",
            transform=ax.transAxes, fontsize=10,
            verticalalignment="top",
            bbox=dict(facecolor="white", alpha=0.8))

    ax.axhline(0, color="gray", linestyle="--", alpha=0.7)
    ax.axvline(0, color="gray", linestyle="--", alpha=0.7)
    ax.grid(True, linestyle=":")
    ax = axes[1, 1]
    ax.set_title("Pulse – Global and Control", fontsize=13, weight="bold")
    L1 = ax.plot(atrasos_seg, g_ppu, color="black", linewidth=2.5, label="Global")
    L2 = ax.plot(atrasos_seg, c_ppu, color="gray", linestyle="--", linewidth=2, label="Controle negativo")

    i0 = np.nanargmax(np.abs(g_ppu))
    ax.plot(atrasos_seg[i0], g_ppu[i0], "ko", markersize=8)

    ax.legend(
        title=f"Pico = {g_ppu[i0]:.3f}\nAtraso = {atrasos_seg[i0]:.1f}s",
        fontsize=10,
        loc="upper left"
    )

    ax.axhline(0, color="gray", linestyle="--")
    ax.grid(True, linestyle=":")
    cax = fig.add_axes([0.4, 0.45, 0.1, 0.02])
    norm = mcolors.Normalize(vmin=0, vmax=len(slice_keys))
    cb = fig.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap),
                      cax=cax, orientation='horizontal')
    cb.set_label("Fatia", fontsize=10)
    cb.ax.tick_params(labelsize=8)
    if save_file_path:
        os.makedirs(os.path.dirname(save_file_path), exist_ok=True)
        fig.savefig(save_file_path, dpi=300, bbox_inches="tight", facecolor="white")
        print(f"Figura salva em: {save_file_path}")

    return fig

def plot_cc_global_subjects(dir_base, subjects, directions, rois, signal, show_image=False, legend_info=True, tipo="all"):
    """
    Plota a curva média do sinal global para cada ROI em subplots separados.
    Estilo compacto - rótulos do eixo X apenas nos plots da última linha.
    Inclui barra de cores para identificar os diferentes sujeitos.
    
    Parameters:
    -----------
    dir_base : str
        Diretório base onde estão os dados
    subjects : list
        Lista de sujeitos/subjects
    direction : str
        Direção dos dados (ex: 'AP', 'PA')
    rois : list
        Lista de ROIs a serem plotadas
    signal : str
        Tipo de sinal a ser extraído
    show_image : bool
        Se True, exibe o gráfico
    """
    
    # Configurações estéticas
    plt.rcParams.update({
        'font.size': 11,
        'axes.linewidth': 1.2,
        'lines.linewidth': 1.5,
        'lines.markersize': 6,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'axes.titlesize': 12,
        'axes.labelsize': 11
    })
    
    # Inicializar estrutura para armazenar todos os sinais
    all_signals = {roi: [] for roi in rois}
    delays = None
    
    # Coletar dados de todos os subjects
    for sub, direction in zip(subjects, directions):
        for roi in rois:

            path_json = os.path.join(dir_base, sub, direction, f"Analysis/cc_media_{roi}_completo.json")
            if os.path.exists(path_json):
                try:
                    data = physiological_marked.read_json(path_json)
                    data = substituir_none_por_nan(data)
                    
                    # Extrair sinal global
                    sinal = data['global'][signal]
                    
                    # Se ainda não temos os delays, extrair uma vez
                    if delays is None and 'atrasos' in data['global']:
                        delays = data['global']['atrasos']
                    
                    all_signals[roi].append(sinal)
                    
                except Exception as e:
                    print(f"Erro ao processar {path_json}: {e}")
                    continue
    # Verificar se temos dados para plotar
    if delays is None:
        print("Nenhum dado de delays encontrado!")
        return
    
    # Configurar colormap para os subjects
    n_subjects = len(subjects)
    colormap = colormaps.get_cmap("viridis")

    colors = [colormap(i / max(1, n_subjects - 1)) for i in range(n_subjects)]
    subject_colors = dict(zip(subjects, colors))
    
    # Configurar o layout dos subplots
    n_rois = len(rois)
    n_cols = min(3, n_rois)
    n_rows = (n_rois + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3*n_cols, 2.8*n_rows))
    
    # Flatten para manipulação fácil
    if n_rois == 1:
        axes = [axes]
    elif n_rows > 1 and n_cols > 1:
        axes = axes.flatten()
    else:
        axes = [axes] if n_cols == 1 else list(axes)
    
    # Índices da última linha para mostrar rótulos do eixo X
    last_row_start = (n_rows - 1) * n_cols
    last_row_indices = range(last_row_start, last_row_start + n_cols)
    
    # Plotar para cada ROI
    for idx, roi in enumerate(rois):
        if idx < len(axes):
            ax = axes[idx]
            roi_signals = all_signals[roi]
            
            if not roi_signals:
                ax.text(0.5, 0.5, f'No data\nfor {roi}', 
                        ha='center', va='center', transform=ax.transAxes, fontsize=10)
                ax.set_title(f'{roi}', fontsize=12, fontweight='bold', pad=10)
                ax.set_xticks([])
                ax.set_yticks([])
                continue
            
            # Converter para array numpy
            roi_signals_array = np.array(roi_signals)
            
            # Plotar curvas individuais dos subjects
            for i, (sinal_subject, subject) in enumerate(zip(roi_signals_array, subjects)):
                color = subject_colors[subject]
                ax.plot(delays, sinal_subject, color=color, alpha=0.3, linewidth=1.0)
            
            # Calcular média e desvio padrão
            sinal_medio = np.nanmean(roi_signals_array, axis=0)
            sinal_std = np.nanstd(roi_signals_array, axis=0)
            
            # Encontrar ponto de maior correlação (módulo) na média
            idx_max_corr = np.nanargmax(np.abs(sinal_medio))
            max_corr_value = sinal_medio[idx_max_corr]
            max_corr_delay = delays[idx_max_corr]
            
            # Plotar sombreado da variação (desvio padrão)
            ax.fill_between(delays, 
                           sinal_medio - sinal_std, 
                           sinal_medio + sinal_std, 
                           alpha=0.4, color='gray', label='_nolegend_')
            
            # Plotar curva média
            ax.plot(delays, sinal_medio, 'k-', alpha=1.0, linewidth=2.5,
                   label='Mean' if idx == 0 else "")
            
            # Plotar ponto de maior correlação
            ax.plot(max_corr_delay, max_corr_value, 'o', 
                   markersize=8, markerfacecolor='red', 
                   markeredgecolor='darkred', markeredgewidth=1.5,
                   label='Peak corr.' if idx == 0 else "")
            # ---------------------------------
            # NOVO: legenda informativa por ROI
            # ---------------------------------
            if legend_info:
                text_legend = (
                    f"Peak corr: {max_corr_value:.3f}\n"
                    f"Delay: {max_corr_delay:.2f} s"
                )

                ax.text(0.98, 0.02, text_legend,
                        transform=ax.transAxes,
                        ha='right', va='bottom',
                        fontsize=8,
                        color='black',
                        bbox=dict(boxstyle="round,pad=0.3",
                                facecolor="white", alpha=0.9,
                                edgecolor="black"))

            
            # Configurações do plot
            ax.set_title(f'{roi}', fontsize=12, fontweight='bold', pad=10)
            
            # Mostrar eixo X apenas na última linha
            if idx in last_row_indices:
                ax.set_xlabel('Time (s)', fontsize=12, fontweight='bold')
                ax.tick_params(axis='x', which='both', labelbottom=True)
            else:
                ax.set_xlabel('')
                ax.tick_params(axis='x', which='both', labelbottom=False)
            
            # Mostrar eixo Y apenas na primeira coluna
            if idx % n_cols == 0:
                ax.set_ylabel('Correlation', fontsize=12, fontweight='bold')
            else:
                ax.set_ylabel('')
                ax.tick_params(axis='y', which='both', labelleft=False)
            
            ax.grid(True, alpha=0.2)
            ax.axhline(y=0, color='gray', linestyle='-', alpha=0.5, linewidth=0.8)
            
            # Ajustar limites do eixo y para melhor visualização
            y_min = np.nanmin(roi_signals_array)
            y_max = np.nanmax(roi_signals_array)
            y_range = y_max - y_min
            if y_range > 0:  # Evitar erro se todos os valores forem iguais
                ax.set_ylim(y_min - 0.1*y_range, y_max + 0.1*y_range)
    
    # Ocultar eixos vazios
    for idx in range(len(rois), len(axes)):
        axes[idx].set_visible(False)
    

    
    # ADIÇÃO: Barra de cores para os subjects
    if n_subjects > 0:
        # Criar eixo para a colorbar
        cax = fig.add_axes([0.45, 0.085, 0.2, 0.02])  # [left, bottom, width, height]
        
        # Criar normalização para os subjects
        norm = mcolors.Normalize(vmin=0, vmax=n_subjects-1)
        
        # Criar colorbar
        cb = fig.colorbar(cm.ScalarMappable(norm=norm, cmap=colormap),
                         cax=cax, orientation='horizontal')
        
        # Configurar a colorbar
        cb.set_label("Subjects", fontsize=10, fontweight='bold')
        cb.ax.tick_params(labelsize=8)
        
        # Definir ticks para mostrar os índices dos subjects
        if n_subjects <= 20:  # Mostrar todos se não forem muitos
            tick_positions = np.arange(n_subjects)
            tick_labels = [f'S{i+1}' for i in range(n_subjects)]
        else:  # Mostrar apenas alguns se houver muitos subjects
            tick_positions = np.linspace(0, n_subjects-1, 5).astype(int)
            tick_labels = [f'S{i+1}' for i in tick_positions]
        
        cb.set_ticks(tick_positions)
        cb.set_ticklabels(tick_labels)
    
    # Ajustar layout para dar espaço à colorbar
    plt.tight_layout(rect=[0, 0.08, 1, 0.95])  # [left, bottom, right, top]
    
    # Salvar figura
    save_path = os.path.join(dir_base, f"Analysis/cc_global_medio_{direction}_{signal}_{tipo}.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Figura salva em: {save_path}")
    
    if show_image:
        plt.show()
    plt.close()
    
    # Retornar dados processados
    max_corr_points = {}
    for roi in rois:
        if all_signals[roi]:
            sinal_medio = np.nanmean(np.array(all_signals[roi]), axis=0)
            idx_max = np.nanargmax(np.abs(sinal_medio))
            max_corr_points[roi] = {
                'value': sinal_medio[idx_max],
                'delay': delays[idx_max],
                'abs_value': np.abs(sinal_medio[idx_max])
            }
    
    return {
        'delays': delays,
        'all_signals': all_signals,
        'mean_signals': {roi: np.nanmean(np.array(all_signals[roi]), axis=0) 
                        for roi in rois if all_signals[roi]},
        'max_correlation_points': max_corr_points,
        'subject_colors': subject_colors
    }


def plot_cc_slices_subjects(dir_base, subjects, directions, rois, signal, show_image=False, legend_info=True, tipo="all"):
    """
    Plota a curva média do sinal de todas as fatias para cada ROI em subplots separados.
    Estilo compacto - rótulos do eixo X apenas nos plots da última linha.
    
    Parameters:
    -----------
    dir_base : str
        Diretório base onde estão os dados
    subjects : list
        Lista de sujeitos/subjects
    direction : str
        Direção dos dados (ex: 'AP', 'PA')
    rois : list
        Lista de ROIs a serem plotadas
    signal : str
        Tipo de sinal a ser extraído
    show_image : bool
        Se True, exibe o gráfico
    """
    
    # Configurações estéticas
    plt.rcParams.update({
        'font.size': 11,
        'axes.linewidth': 1.2,
        'lines.linewidth': 1.5,
        'lines.markersize': 6,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'axes.titlesize': 12,
        'axes.labelsize': 11
    })
    
    # Inicializar estrutura para armazenar todos os sinais das fatias
    all_slices_signals = {roi: [] for roi in rois}
    delays = None
    
    # Coletar dados de todos os subjects e fatias
    for sub, direction in zip(subjects, directions):
        for roi in rois:
            path_json = os.path.join(dir_base, sub, direction, f"Analysis/cc_media_{roi}_completo.json")
            
            if os.path.exists(path_json):
                try:
                    data = physiological_marked.read_json(path_json)
                    data = substituir_none_por_nan(data)
                    
                    # Encontrar todas as fatias (slice_0, slice_1, ...)
                    slice_keys = [key for key in data.keys() if key.startswith('slice_')]
                    
                    # Se ainda não temos os delays, extrair uma vez (das fatias ou global)
                    if delays is None:
                        if slice_keys and 'atrasos' in data[slice_keys[0]]:
                            delays = data[slice_keys[0]]['atrasos']
                        elif 'global' in data and 'atrasos' in data['global']:
                            delays = data['global']['atrasos']
                    
                    # Coletar sinais de todas as fatias
                    for slice_key in slice_keys:
                        if signal in data[slice_key]:
                            sinal_slice = data[slice_key][signal]
                            all_slices_signals[roi].append(sinal_slice)
                            
                except Exception as e:
                    print(f"Erro ao processar {path_json}: {e}")
                    continue
    
    # Verificar se temos dados para plotar
    if delays is None:
        print("Nenhum dado de delays encontrado!")
        return
    
    # Configurar o layout dos subplots
    n_rois = len(rois)
    n_cols = min(3, n_rois)
    n_rows = (n_rois + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3*n_cols, 2.5*n_rows))
    
    # Flatten para manipulação fácil
    if n_rois == 1:
        axes = [axes]
    elif n_rows > 1 and n_cols > 1:
        axes = axes.flatten()
    else:
        axes = [axes] if n_cols == 1 else list(axes)
    
    # Índices da última linha para mostrar rótulos do eixo X
    last_row_start = (n_rows - 1) * n_cols
    last_row_indices = range(last_row_start, last_row_start + n_cols)
    
    # Plotar para cada ROI
    for idx, roi in enumerate(rois):
        if idx < len(axes):
            ax = axes[idx]
            roi_slices_signals = all_slices_signals[roi]
            
            if not roi_slices_signals:
                ax.text(0.5, 0.5, f'No slice data\nfor {roi}', 
                        ha='center', va='center', transform=ax.transAxes, fontsize=10)
                ax.set_title(f'{roi}', fontsize=12, fontweight='bold', pad=10)
                ax.set_xticks([])
                ax.set_yticks([])
                continue
            
            # Converter para array numpy
            roi_slices_array = np.array(roi_slices_signals)
            
            # Calcular média e desvio padrão de todas as fatias
            sinal_medio = np.nanmean(roi_slices_array, axis=0)
            sinal_std = np.nanstd(roi_slices_array, axis=0)
            
            # Encontrar ponto de maior correlação (módulo) na média
            idx_max_corr = np.nanargmax(np.abs(sinal_medio))
            max_corr_value = sinal_medio[idx_max_corr]
            max_corr_delay = delays[idx_max_corr]
            
            # Plotar sombreado da variação (desvio padrão)
            ax.fill_between(delays, 
                           sinal_medio - sinal_std, 
                           sinal_medio + sinal_std, 
                           alpha=0.4, color='gray', label='_nolegend_')
            
            # Plotar curva média
            ax.plot(delays, sinal_medio, 'k-', alpha=1.0, linewidth=2.5,
                   label='Mean slices' if idx == 0 else "")
            
            # Plotar ponto de maior correlação
            ax.plot(max_corr_delay, max_corr_value, 'o', 
                   markersize=8, markerfacecolor='red', 
                   markeredgecolor='darkred', markeredgewidth=1.5,
                   label='Peak corr.' if idx == 0 else "")

            if legend_info:
                text_legend = (
                    f"Peak corr: {max_corr_value:.3f}\n"
                    f"Delay: {max_corr_delay:.2f} s"
                )

                ax.text(0.98, 0.02, text_legend,
                        transform=ax.transAxes,
                        ha='right', va='bottom',
                        fontsize=8,
                        color='black',
                        bbox=dict(boxstyle="round,pad=0.3",
                                facecolor="white", alpha=0.9,
                                edgecolor="black"))

            # Adicionar informação do número de fatias
            text_str = f'N slices: {len(roi_slices_signals)}'
            bbox_props = dict(boxstyle="round,pad=0.3", facecolor="white", 
                            alpha=0.9, edgecolor="black")
            
            ax.text(0.95, 0.95, text_str, 
                   transform=ax.transAxes,
                   ha='right', va='top',
                   fontsize=9,
                   bbox=bbox_props)
            
            # Configurações do plot
            ax.set_title(f'{roi}', fontsize=12, fontweight='bold', pad=10)
            
            # Mostrar eixo X apenas na última linha
            if idx in last_row_indices:
                ax.set_xlabel('Time (s)', fontsize=12, fontweight='bold')
                ax.tick_params(axis='x', which='both', labelbottom=True)
            else:
                ax.set_xlabel('')
                ax.tick_params(axis='x', which='both', labelbottom=False)
            
            # Mostrar eixo Y apenas na primeira coluna
            if idx % n_cols == 0:
                ax.set_ylabel('Correlation', fontsize=12, fontweight='bold')
            else:
                ax.set_ylabel('')
                ax.tick_params(axis='y', which='both', labelleft=False)
            
            ax.grid(True, alpha=0.2)
            ax.axhline(y=0, color='gray', linestyle='-', alpha=0.5, linewidth=0.8)
            
            # Ajustar limites do eixo y para melhor visualização
            y_min = np.nanmin(roi_slices_array)
            y_max = np.nanmax(roi_slices_array)
            y_range = y_max - y_min
            if y_range > 0:  # Evitar erro se todos os valores forem iguais
                ax.set_ylim(y_min - 0.1*y_range, y_max + 0.1*y_range)
    
    # Ocultar eixos vazios
    for idx in range(len(rois), len(axes)):
        axes[idx].set_visible(False)
    
    # Adicionar legenda apenas se houver dados
    if len(rois) > 0 and all_slices_signals[rois[0]]:
        handles, labels = axes[0].get_legend_handles_labels()
        if handles:
            fig.legend(handles, labels, loc='upper center', 
                      bbox_to_anchor=(0.5, 0.02), ncol=2, fontsize=10)
    
    plt.tight_layout()
    
    # Salvar figura
    save_path = os.path.join(dir_base, f"Analysis/cc_slices_medio_{direction}_{signal}_{tipo}.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Figura salva em: {save_path}")
    
    if show_image:
        plt.show()
    plt.close()
    
    # Retornar dados processados
    max_corr_points = {}
    slice_stats = {}
    
    for roi in rois:
        if all_slices_signals[roi]:
            roi_slices_array = np.array(all_slices_signals[roi])
            sinal_medio = np.nanmean(roi_slices_array, axis=0)
            idx_max = np.nanargmax(np.abs(sinal_medio))
            
            max_corr_points[roi] = {
                'value': sinal_medio[idx_max],
                'delay': delays[idx_max],
                'abs_value': np.abs(sinal_medio[idx_max])
            }
            
            slice_stats[roi] = {
                'n_slices': len(all_slices_signals[roi]),
                'mean_signal': sinal_medio,
                'std_signal': np.nanstd(roi_slices_array, axis=0)
            }
    
    return {
        'delays': delays,
        'all_slices_signals': all_slices_signals,
        'mean_signals': {roi: np.nanmean(np.array(all_slices_signals[roi]), axis=0) 
                        for roi in rois if all_slices_signals[roi]},
        'max_correlation_points': max_corr_points,
        'slice_stats': slice_stats
    }


def processar_cc_completo(path, name='GM', intervalo=3, step=0.1, respiracao_col=None, pulsacao_col=None, 
                         zscore=True, cutoff_ppu=5.0, cutoff_resp=1.0, 
                         n_controle=1000, save_plot=True):
    """
    Função de conveniência para processamento completo.
    """
    #path_roi = os.path.join(os.path.dirname(path), f"rois/{name}.nii.gz")
    path_roi = os.path.join(path, f"roi/{name}.nii.gz")
    path_adc = os.path.join(path, "Analysis/adc.nii.gz")
    path_b0 = os.path.join(path, "b0_brain_mask.nii.gz")
    path_physiological_marked = os.path.join(path, "Analysis/physiological_marked.csv")
    
    # Carregar dados
    roi_mask = nib.load(path_roi).get_fdata()
    df_dados = pd.read_csv(path_physiological_marked)
    data_adc = nib.load(path_adc).get_fdata()
    b0 = nib.load(path_b0).get_fdata()
    
    if data_adc.shape[2] == 27 and roi_mask.shape[2] == 30:
        print("Ajustando dimensões da ROI...")
        roi_mask = roi_mask[:, :, 2:29]
    
    path_output_json = os.path.join(path, f"Analysis/cc_media_{name}_completo.json")
    
    # Processar correlação
    resultados = processar_correlacao_adc(
        adc=data_adc, df_dados=df_dados, mask=roi_mask, brain=b0, 
        path_output_json=path_output_json,
        atraso_min=-intervalo, atraso_max=intervalo, passo=step,
        respiracao_col=respiracao_col, pulsacao_col=pulsacao_col, zscore=zscore,
        cutoff_ppu_hz=cutoff_ppu, cutoff_resp_hz=cutoff_resp,
        n_simulacoes_controle=n_controle
    )
    
    # Plotar resultados
    if save_plot:
        save_plot_path = os.path.join(path, f"Analysis/cc_completo_{name}.png")
        plotar_curvas_cc_slice_roi(
            json_file=path_output_json,
            save_file_path=save_plot_path
        )
        #plt.close()
    plt.close()
    return resultados

# resultados = processar_cc_completo(
#     path="/home/joao/Documentos/Scripts_dynDWI/testes/sub014/dynDWI_S",
#     name="WM",
#     intervalo=5,
#     step=0.1,
#     save_plot=True,
#     cutoff_ppu=5, cutoff_resp=1, 
#                          n_controle=10,
# )
