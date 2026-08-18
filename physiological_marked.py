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
from scipy.signal import hilbert
from scipy.signal import butter, filtfilt
import numpy as np
import pandas as pd
from scipy.ndimage import uniform_filter1d
from scipy.signal import find_peaks

from dtw_alignment import calcular_fase_respiratoria_dtw, calcular_fase_ppu_dtw

from editor_respiracao import editar_respiracao

def load(path):
  df = pd.read_csv(
      path,
      skiprows=3,
      sep='\s+',
      engine='python')

  colunas = df.columns.tolist()
  novas_colunas = colunas[1:] + [colunas[0]]
  df.columns = novas_colunas
  df = df.drop(columns=[ "#"])
  return df

def read_json(path):
  with open(path, 'r') as f:
    data = json.load(f)
  return data

def find_PEy(df_fisio):
  intervalo_pulso = 0.023
  limite_negativo_min = -3300
  limite_negativo_max = -2800
  limite_positivo_min = 1900
  limite_positivo_max = 2300
  indices_pulso = []
  for i in range(1, len(df_fisio) - 1):
    # Verificação de mínimo local
    if (df_fisio['gy'].iloc[i] < df_fisio['gy'].iloc[i-1]) and \
      (df_fisio['gy'].iloc[i] <= df_fisio['gy'].iloc[i+1]) and \
      (limite_negativo_min <= df_fisio['gy'].iloc[i] <= limite_negativo_max):

      # Início da janela de busca para o pico positivo
      time_inicio_busca = df_fisio['Time'].iloc[i]
      # Encontra o próximo índice que está 0.025s à frente
      idx_fim_busca = df_fisio[df_fisio['Time'] > time_inicio_busca + intervalo_pulso].index.min()
      # Se não encontrar um índice válido para a janela, pare a busca
      if pd.isna(idx_fim_busca):
        idx_fim_busca = len(df_fisio)

      # Sub-busca para encontrar o máximo local dentro da janela de tempo
      for j in range(i + 1, idx_fim_busca):
        if (df_fisio['gy'].iloc[j] >= df_fisio['gy'].iloc[j-1]) and \
        (df_fisio['gy'].iloc[j] > df_fisio['gy'].iloc[j+1]) and \
        (limite_positivo_min <= df_fisio['gy'].iloc[j] <= limite_positivo_max):

          indices_pulso.append(j + 1)
          i = j
          break
  return indices_pulso

def find_PEy2(df_fisio):
  intervalo_pulso = 0.23
  limite_negativo_min = -3300
  limite_negativo_max = -2050
  limite_positivo_min = 1350
  limite_positivo_max = 2300
  indices_pulso = []
  for i in range(1, len(df_fisio) - 1):
    
    if (df_fisio['gy'].iloc[i] < df_fisio['gy'].iloc[i-1]) and \
      (df_fisio['gy'].iloc[i] <= df_fisio['gy'].iloc[i+1]) and \
      (limite_negativo_min <= df_fisio['gy'].iloc[i] <= limite_negativo_max):

      time_inicio_busca = df_fisio['Time'].iloc[i]
      idx_fim_busca = df_fisio[df_fisio['Time'] > time_inicio_busca + intervalo_pulso].index.min()
      if pd.isna(idx_fim_busca):
        idx_fim_busca = len(df_fisio)
  
      for j in range(i + 1, idx_fim_busca):
        if (df_fisio['gy'].iloc[j] >= df_fisio['gy'].iloc[j-1]) and \
        (df_fisio['gy'].iloc[j] > df_fisio['gy'].iloc[j+1]) and \
        (limite_positivo_min <= df_fisio['gy'].iloc[j] <= limite_positivo_max):

          indices_pulso.append(j + 1)
          i = j
          break
  return indices_pulso[2:]

def indice_medio_bloco(id):
  indices_medios = []
  for i in range(len(id) - 1):
    indice_medio = (id[i] + id[i + 1]) / 2
    indices_medios.append(indice_medio)

  # Calcular a distância média entre os pontos médios
  if len(indices_medios) > 1:
    distancias = [indices_medios[i+1] - indices_medios[i] for i in range(len(indices_medios)-1)]
    distancia_media = sum(distancias) / len(distancias)
  else:
    #Se só temos um ponto médio, usa a distância entre os dois primeiros pares
    distancia_media = id[1] - id[0] if len(id) > 1 else 0

  # Adicionar o último ponto médio baseado na distância média
  if len(id) > 0:
    if len(indices_medios) > 0:
      ultimo_medio = indices_medios[-1] + distancia_media #último ponto médio calculado + distância média
    else:
      ultimo_medio = id[0] + distancia_media / 2  # Se não há pontos médios calculados
    indices_medios.append(ultimo_medio)
  indices_medios_inteiros = [int(round(medio)) for medio in indices_medios]
  return indices_medios_inteiros

def indice_medio_b0(id): # (1,0), (3,2), (5,4), ...
    indices_medios = []
    for i in range(0, len(id) - 1, 2):
        indice_medio = (id[i] + id[i + 1]) / 2
        indices_medios.append(indice_medio)
    return [int(round(m)) for m in indices_medios]

def indice_medio_b0_inicio(id, df, repetition_time):
    tempos = df["Time"].to_numpy()
    indices = []
    for i in range(0, len(id) - 1, 2):
        tempo_alvo = tempos[id[i]] + repetition_time / 2
        novo_indice = np.argmin(np.abs(tempos - tempo_alvo))
        indices.append(int(novo_indice))

    return indices

def indice_medio_b0_fim(id, df, repetition_time):
    tempos = df["Time"].to_numpy()
    indices = []

    for i in range(0, len(id) - 1, 2):
        tempo_alvo = tempos[id[i + 1]] - repetition_time / 2
        novo_indice = np.argmin(np.abs(tempos - tempo_alvo))
        indices.append(int(novo_indice))

    return indices

def indice_b0_fim(id, df, repetition_time):

    tempos = df["Time"].to_numpy()
    indices = []

    for i in range(0, len(id) - 1, 2):
        tempo_alvo = tempos[id[i + 1]] - repetition_time
        novo_indice = np.argmin(np.abs(tempos - tempo_alvo))
        indices.append(int(novo_indice))

    return indices

def indice_medio_difusao(id):  # Média entre indices_restantes[i+1] e indices_restantes[i]
  indices_restantes = id[1:]
  indices_medios = []
  for i in range(0, len(indices_restantes) - 1, 2):
    if i + 1 < len(indices_restantes):
        indice_medio = (indices_restantes[i+1] + indices_restantes[i]) / 2
        indices_medios.append(indice_medio)
  if len(indices_medios) > 0:
    if len(indices_medios) > 1:
      distancias = [indices_medios[j+1] - indices_medios[j] for j in range(len(indices_medios)-1)]
      distancia_media = sum(distancias) / len(distancias)
    else:
      distancia_media = indices_restantes[2] - indices_restantes[0] if len(indices_restantes) >= 3 else indices_medios[0]
    ultimo_medio = indices_medios[-1] + distancia_media
    indices_medios.append(ultimo_medio)

  indices_medios_inteiros = [int(round(medio)) for medio in indices_medios]
  return indices_medios_inteiros

def paradigm_task(df, t_livre, t_passada, t0=0):
    """
    Constrói variáveis 'task' e 'ciclo' intercalando períodos de respiração
    livre e passada ao longo do tempo presente em df['Time'].

    Parâmetros:
        df : pandas.DataFrame (obrigatório ter coluna 'Time')
        t_livre : duração da respiração livre (em segundos)
        t_passada : duração da respiração passada (em segundos)
        t0 : tempo inicial a partir do qual começam os ciclos
    Retorna:
        df com colunas 'task' e 'ciclo'
    """
    tarefa = np.zeros(len(df), dtype=int)

    periodo_total = t_livre + t_passada

    for i, t in enumerate(df['Time']):
        if t < t0:
            tarefa[i] = 0  # antes do início → livre
        else:
            bloco = int((t - t0) // periodo_total)
            dentro = (t - t0) % periodo_total
            tarefa[i] = 1 if dentro >= t_livre else 0

    df['task'] = tarefa
    ciclo = np.ones(len(df), dtype=int)
    for i in range(1, len(df)):
        if df['task'].iloc[i] != df['task'].iloc[i-1]:
            ciclo[i:] += 1

    df['ciclo'] = ciclo

    return df


####### Peripheral pulse marking
def encontrar_pulsos_ppu(df, coluna_dados, distancia_minima=200, janela_suavizacao=15, altura_minima=None, janela_busca=100, sensibilidade=0.2, fallback_offset=65):
    """
    Parâmetros:
        df (pd.DataFrame): O DataFrame com os dados de pulsação.
        coluna_dados (str): O nome da coluna com os dados de pulsação (e.g., 'ppu').
        distancia_minima (int): A distância mínima entre os picos em número de amostras.
        janela_suavizacao (int): O tamanho da janela do filtro de média móvel para suavizar os dados.
        altura_minima (float): A altura mínima que um pico deve ter para ser considerado.
        janela_busca (int): A janela de busca (em amostras) para trás a partir de cada pico.
        sensibilidade (float): Fator de sensibilidade para encontrar o início da subida.
        fallback_offset (int): Offset a ser usado como início do pulso se a busca falhar.
    Retorna:
        pd.DataFrame: O DataFrame original com as novas colunas de flag adicionadas.
    """

    #df_copy = df.copy()
    df_copy = df
    dados_originais = df_copy[coluna_dados].values
    dados_suavizados = pd.Series(dados_originais).rolling(window=janela_suavizacao, center=True).mean().bfill().ffill().values

    # Determinar a altura mínima para picos (se não for especificado)
    if altura_minima is None:
        altura_minima = np.percentile(dados_suavizados, 75)

    picos, _ = find_peaks(dados_suavizados, height=altura_minima, distance=distancia_minima)
    # Encontrar o início da subida
    inicios_pulsos = []
    diferencas_suavizadas = np.diff(dados_suavizados, prepend=dados_suavizados[0])
    limiar_subida = np.mean(diferencas_suavizadas[diferencas_suavizadas > 0]) * sensibilidade

    for pico_idx in picos:
        janela_inicio = max(0, pico_idx - janela_busca)
        encontrado_inicio = False

        for i in range(pico_idx - 1, janela_inicio - 1, -1):
            if diferencas_suavizadas[i] < limiar_subida:
                inicios_pulsos.append(i)
                encontrado_inicio = True
                break

        if not encontrado_inicio or (pico_idx - inicios_pulsos[-1] < 5):
            fallback_idx = max(0, pico_idx - fallback_offset)
            if not encontrado_inicio:
                 inicios_pulsos.append(fallback_idx)
            else:
                inicios_pulsos.pop()
                inicios_pulsos.append(fallback_idx)

    # Corrigir a diferença de tamanho entre picos e inícios
    if len(picos) != len(inicios_pulsos):
        if len(picos) > len(inicios_pulsos):
            for i in range(len(inicios_pulsos), len(picos)):
                inicios_pulsos.append(max(0, picos[i] - fallback_offset))
        else:
            inicios_pulsos = inicios_pulsos[:len(picos)]
    finais_pulsos = inicios_pulsos[1:] + [len(df_copy) - 1]

    # Criar e preencher as novas colunas n0 DataFrame
    df_copy['ppu_peaks'] = 0
    df_copy['start_ppu_pulse'] = 0
    df_copy['end_ppu_pulse'] = 0

    if len(picos) > 0:
        df_copy.loc[picos, 'ppu_peaks'] = 1

    if len(inicios_pulsos) > 0:
        df_copy.loc[inicios_pulsos, 'start_ppu_pulse'] = 1

    if len(finais_pulsos) > 0:
        df_copy.loc[finais_pulsos, 'end_ppu_pulse'] = 1

    return df_copy

def quantificar_fase_ppu(df):
    """
    Calcula a fase (0–1) de cada instante em relação ao ciclo de pulso mais próximo,
    e armazena o resultado em uma nova coluna 'fase_flag'.
    A fase é calculada como:
        fase = (tempo_atual - tempo_inicio_pulso) / (tempo_fim_pulso - tempo_inicio_pulso)
    """
    #df_copy = df.copy()
    df_copy = df
    df_copy['ppu_linear_phase'] = np.nan

    tempos_inicio = df_copy.loc[df_copy['start_ppu_pulse'] == 1, 'Time'].values
    tempos_fim = df_copy.loc[df_copy['end_ppu_pulse'] == 1, 'Time'].values
    num_pulsos = min(len(tempos_inicio), len(tempos_fim))
    tempos_inicio = tempos_inicio[:num_pulsos]
    tempos_fim = tempos_fim[:num_pulsos]

    for i, tempo_atual in enumerate(df_copy['Time'].values):
        idx_pulso = np.where(tempos_inicio <= tempo_atual)[0]
        if len(idx_pulso) > 0:
            idx_pulso = idx_pulso[-1]
            if idx_pulso < num_pulsos and tempo_atual <= tempos_fim[idx_pulso]:
                t_ini = tempos_inicio[idx_pulso]
                t_fim = tempos_fim[idx_pulso]
                duracao = t_fim - t_ini
                if duracao > 0:
                    fase = (tempo_atual - t_ini) / duracao
                    df_copy.loc[i, 'ppu_linear_phase'] = fase
                else:
                    df_copy.loc[i, 'ppu_linear_phase'] = np.nan
            else:# Fora de um pulso ativo
                df_copy.loc[i, 'ppu_linear_phase'] = np.nan
        else:
            df_copy.loc[i, 'ppu_linear_phase'] = np.nan

    return df_copy

def quantificar_fase_ppu_rr(df):
    """
    Calcula a fase cardíaca usando os pontos médios entre picos.
    Definição:
        ponto_medio_anterior -> fase 0
        pico                 -> fase 0.5
        ponto_medio_posterior-> fase 1
    O ciclo é definido entre dois pontos médios consecutivos.
    """

    #df = df.copy()
    df['ppu_linear_phase_rr'] = np.nan
    picos = df.loc[df['ppu_peaks'] == 1, 'Time'].values
    picos_indices = df.loc[df['ppu_peaks'] == 1].index.values
    if len(picos) < 3:
        print("São necessários pelo menos 3 picos.")
        return df
    for i in range(1, len(picos) - 1):
        t_prev = picos[i - 1]
        t_peak = picos[i]
        t_next = picos[i + 1]
        t_mid_prev = (t_prev + t_peak) / 2
        t_mid_next = (t_peak + t_next) / 2

        # metade ascendente: 0 -> 0.5
        mask1 = ((df['Time'] >= t_mid_prev) & (df['Time'] <= t_peak))
        if t_peak > t_mid_prev:
            t = df.loc[mask1, 'Time']
            fase = 0.5 * ((t - t_mid_prev) /(t_peak - t_mid_prev))
            df.loc[mask1, 'ppu_linear_phase_rr'] = fase

        # metade descendente: 0.5 -> 1
        mask2 = ((df['Time'] > t_peak) & (df['Time'] <= t_mid_next))
        if t_mid_next > t_peak:
            t = df.loc[mask2, 'Time']
            fase = 0.5 + 0.5 * ((t - t_peak) /(t_mid_next - t_peak))
            df.loc[mask2, 'ppu_linear_phase_rr'] = fase

    primeiro_mid = (picos[0] + picos[1]) / 2
    segundo_mid  = (picos[1] + picos[2]) / 2
    idx_a = np.where(df['Time'] >= primeiro_mid)[0][0]
    idx_b = np.where(df['Time'] >= segundo_mid)[0][0]
    n_pre = idx_a

    if n_pre > 0:
        fase = df['ppu_linear_phase_rr'].values.copy()
        inicio_copia = idx_b - n_pre
        if inicio_copia >= 0:
            fase[:idx_a] = fase[inicio_copia:idx_b]
        else:
            trecho = fase[:idx_b]
            rep = int(np.ceil(n_pre / len(trecho)))
            trecho = np.tile(trecho, rep)
            fase[:idx_a] = trecho[-n_pre:]
        df['ppu_linear_phase_rr'] = fase

    ultimo_mid = (picos[-2] + picos[-1]) / 2
    mask_fim = df['Time'] > ultimo_mid
    idx_fim = df.index[mask_fim]
    if len(idx_fim) > 0:
        ciclo_ref = df.loc[(df['Time'] >= (picos[-3] + picos[-2]) / 2) & (df['Time'] <= ultimo_mid),'ppu_linear_phase_rr'].dropna().values
        if len(ciclo_ref) > 0:
            n = len(idx_fim)
            if n <= len(ciclo_ref):
                valores = ciclo_ref[:n]
            else:
                valores = np.resize(ciclo_ref, n)
            df.loc[idx_fim, 'ppu_linear_phase_rr'] = valores
    return df

def plot_gradients(df, nrows=10, figsize=(15, 20), title="Gradientes", save=False, show=False, path="."):
    n_points = len(df)
    step = n_points // nrows
    fig, axes = plt.subplots(nrows=nrows, ncols=1, figsize=figsize, sharex=False)

    if nrows == 1:
        axes = [axes]

    for i in range(nrows):
        start = i * step
        end = (i + 1) * step if i < nrows - 1 else n_points
        ax = axes[i]

        t = df['Time'].iloc[start:end].values
        if 'quadro_resp_hilbert' in df.columns:

            quadro = df['quadro_resp_hilbert'].iloc[start:end].values

            mudancas = np.where(np.diff(quadro.astype('float')) != 0)[0] + 1
            limites = np.concatenate(([0], mudancas, [len(quadro)]))

            for k in range(len(limites)-1):
                i0, i1 = limites[k], limites[k+1]

                valor = quadro[i0]

                if pd.isna(valor):
                    continue

                cor = 'lightgreen' if valor == 0 else 'lightpink'

                ax.axvspan(
                    t[i0],
                    t[i1-1],
                    color=cor,
                    alpha=0.25,
                    zorder=0
                )

        # Gradientes principais
        ax.plot(t, df['gx'].iloc[start:end].values, label='gx', color=plt.cm.tab10(0), alpha=0.9)
        ax.plot(t, df['gy'].iloc[start:end].values, label='gy', color=plt.cm.tab10(1), alpha=0.6)
        ax.plot(t, df['gz'].iloc[start:end].values, label='gz', color=plt.cm.tab10(2), alpha=0.5)

        # Flags principais
        mask_PEgy = (df['PEgy'].iloc[start:end] == 1)
        ax.plot(t[mask_PEgy], [2000] * mask_PEgy.sum(), 'o', label='PEgy', color=plt.cm.tab10(0))

        mask_bloco = (df['bloco'].iloc[start:end] == 1)
        ax.plot(t[mask_bloco], [2000] * mask_bloco.sum(), 'o', label='bloco', color=plt.cm.tab10(2))

        mask_dif = (df['difusao'].iloc[start:end] == 1)
        ax.plot(t[mask_dif], [2000] * mask_dif.sum(), 'o', label='difusao', color=plt.cm.tab10(4))

        mask_central_b0_i = (df['central_b0_inicio'].iloc[start:end] == 1)
        ax.plot(t[mask_central_b0_i], [2000] * mask_central_b0_i.sum(), 'o', label='central_b0_inicio', color=plt.cm.tab10(6))

        mask_central_b0_e = (df['central_b0_fim'].iloc[start:end] == 1)
        ax.plot(t[mask_central_b0_e], [2000] * mask_central_b0_e.sum(), 'o', label='central_b0_fim', color=plt.cm.tab10(8))

        mask_central_b0 = (df['central_b0'].iloc[start:end] == 1)
        ax.plot(t[mask_central_b0], [2000] * mask_central_b0.sum(), 'o', label='central_b0', color=plt.cm.tab10(10))
        
        slice_goup = (df['slice_mb_gradientes'].iloc[start:end] != -1)
        ax.plot(df['Time'].iloc[start:end][slice_goup], (df['slice_mb_gradientes'].iloc[start:end][slice_goup] + 1) / (df['slice_mb_gradientes'].iloc[start:end][slice_goup] + 1) * 3000,
         '^', label='slice_mb_times', color=plt.cm.tab10(10))

        slice_goup = (df['slice_mb_b0'].iloc[start:end] != -1)
        ax.plot(df['Time'].iloc[start:end][slice_goup], (df['slice_mb_b0'].iloc[start:end][slice_goup] + 1) / (df['slice_mb_b0'].iloc[start:end][slice_goup] + 1) * 3000,
         '^', label='slice_mb_b0', color=plt.cm.tab10(10))
        # Picos PPU (Implementação da requisição)
        if 'ppu_peaks' in df.columns and 'ppu' in df.columns:
            mask_pulso = (df['ppu_peaks'].iloc[start:end] == 1)

            # Garante que só plotamos se houver picos neste trecho
            if mask_pulso.any():
                # A coordenada Y é o valor da série 'ppu' nos pontos marcados
                ppu_peaks = df['ppu'].iloc[start:end][mask_pulso].values

                # Marcação: 'v' (triângulo invertido) vermelho
                ax.plot(t[mask_pulso], ppu_peaks, 'o', label='PPU Pico',
                        color='red', markersize=5, markeredgewidth=1.5)

        # Fisiológicos
        if 'ppu' in df.columns:
            ax.plot(t, df['ppu'].iloc[start:end].values, label='ppu', color="black")
        if 'resp' in df.columns:
            ax.plot(t, df['resp'].iloc[start:end].values, label='resp', color="gray")

        ax.legend(loc="upper right", fontsize=7, ncol=2)
        ax.grid(True, linestyle="--", alpha=0.5)

    axes[-1].set_xlabel("Tempo")
    fig.suptitle(title, fontsize=16, y=1.02)
    plt.tight_layout()

    if save:
        filename = f"{title.replace(' ', '_')}.png"
        save_path = os.path.join(path, filename)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Figura salva em: {save_path}")

    if show:
        plt.show()
    plt.close()

def computar_1_2_derivadas_ppu(df):
    window_length = 51  #(deve ser ímpar)
    polyorder = 3 # ordem do polinômio
    window_length_deriv = 51
    # Aplicar suavização no sinal ppu original
    if len(df) > window_length:
        try:
            # Suavizar o sinal ppu
            ppu_smooth = savgol_filter(df['ppu'].ffill.values, 
                                    window_length, polyorder)
        except:
            # Fallback se a suavização falhar
            ppu_smooth = df['ppu'].ffill().values
    else:
        # Sem suavização se dados insuficientes
        ppu_smooth = df['ppu'].fillna(method='ffill').values

    ppu_first_derivative = np.gradient(ppu_smooth)
    ppu_first_derivative = savgol_filter(ppu_first_derivative, window_length_deriv, polyorder)
    ppu_second_derivative = np.gradient(ppu_first_derivative)
    ppu_second_derivative = savgol_filter(ppu_second_derivative, window_length_deriv, polyorder)

    return ppu_first_derivative, ppu_second_derivative

######respiração
##abordagem usando mac (artigo Lu et al. 2006)

def estimate_period_fft(time, signal, duration= 45):
    """
    Estima o período respiratório dominante usando FFT nos primeiros 'duration' segundos.
    """
    idx = time <= duration
    t = time[idx]
    y = signal[idx]
    fs = 1 / np.mean(np.diff(t))
    y = y - np.mean(y)
    freq = np.fft.rfftfreq(len(y), d=1/fs)
    fft = np.abs(np.fft.rfft(y))
    fft[0] = 0
    mask = (freq > 0.05) & (freq < 1)
    f_resp = freq[mask][np.argmax(fft[mask])]
    T = 1 / f_resp
    return T, f_resp

def moving_average_curve(signal, fs, T):
    """Calcula a curva de média móvel (MAC)"""
    window = int(round(T * fs))
    if window % 2 == 0:
        window += 1
    mac = uniform_filter1d(signal, size=window, mode="nearest")
    return mac

def detect_intercepts(signal, mac):
    """Detecta cruzamentos ascendentes (up) e descendentes (down)."""
    diff = signal - mac
    s = np.sign(diff)
    crossings = np.where(np.diff(s))[0]
    up = []
    down = []
    for i in crossings:
        if diff[i] < 0 and diff[i+1] >= 0:
            up.append(i)
        elif diff[i] > 0 and diff[i+1] <= 0:
            down.append(i)
    return np.array(up), np.array(down)

def clean_intercepts(up, down, fs, T):
    """Remove cruzamentos muito próximos (distância < T/20) e garante alternância."""
    minimum = int(fs * T / 20)
    events = [(i, 'up') for i in up] + [(i, 'down') for i in down]
    events.sort()
    cleaned = []
    for idx, tp in events:
        if len(cleaned) == 0:
            cleaned.append((idx, tp))
            continue
        last_idx, last_tp = cleaned[-1]
        if idx - last_idx < minimum:
            continue
        if tp == last_tp:
            cleaned[-1] = (idx, tp)
        else:
            cleaned.append((idx, tp))
    up = np.array([i for i, t in cleaned if t == "up"])
    down = np.array([i for i, t in cleaned if t == "down"])
    return up, down

def detect_peaks_valleys(signal, up, down):
    """Define picos como máximos entre up e down; vales como mínimos entre down e up."""
    peaks = []
    valleys = []
    for u in up:
        candidates_down = down[down > u]
        if len(candidates_down) == 0:
            break
        d = candidates_down[0]
        i = u + np.argmax(signal[u:d+1])
        peaks.append(i)
    for d in down:
        candidates_up = up[up > d]
        if len(candidates_up) == 0:
            break
        u = candidates_up[0]
        i = d + np.argmin(signal[d:u+1])
        valleys.append(i)
    return np.array(peaks), np.array(valleys)

def remove_small_breaths(signal, peaks, valleys,threshold = 0.2):
    """Remove ciclos com amplitude < threshold * amplitude média."""
    n = min(len(peaks), len(valleys))
    amp = np.abs(signal[peaks[:n]] - signal[valleys[:n]])
    mean_amp = np.mean(amp)
    mask = amp > threshold * mean_amp
    return peaks[:n][mask], valleys[:n][mask]

def respiratory_analysis(df, resp_column= 'resp',interpolate = True, threshold_amp = 0.2):
    #df_work = df.copy()
    df_work = df

    if interpolate:
        df_work[resp_column] = df_work[resp_column].interpolate(method="linear").bfill().ffill()
    
    time = df_work.Time.values
    signal = df_work[resp_column].values
    fs = 1 / np.mean(np.diff(time))
    
    T, f = estimate_period_fft(time, signal)
    mac = moving_average_curve(signal, fs, T)
    up, down = detect_intercepts(signal, mac)
    up, down = clean_intercepts(up, down, fs, T)
    peaks, valleys = detect_peaks_valleys(signal, up, down)
    peaks, valleys = remove_small_breaths(signal, peaks, valleys, threshold=threshold_amp)

    return {
        "period": T,
        "frequency": f,
        "mac": mac,
        "up": up,
        "down": down,
        "peaks": peaks,
        "valleys": valleys,
        "signal": signal,
        "time": time,
        "fs": fs
    }

def marcar_picos_vales_resp(df, resp_column= 'resp',interpolate= True,threshold_amp= 0.2):
    """
    Adiciona as colunas 'resp_peaks' e 'resp_valleys' ao DataFrame,
    marcando com 1 os pontos que são picos ou vales (segundo o algoritmo do artigo).
    """
    # Executa a análise
    result = respiratory_analysis(
        df=df,
        resp_column=resp_column,
        interpolate=interpolate,
        threshold_amp=threshold_amp)

    #df_out = df.copy()
    df_out = df
    
    df_out['resp_peaks'] = 0
    df_out['resp_valleys'] = 0
    
    peaks_idx = result['peaks']
    valleys_idx = result['valleys']
    
    df_out.loc[peaks_idx, 'resp_peaks'] = 1
    df_out.loc[valleys_idx, 'resp_valleys'] = 1

    return df_out

def adicionar_breath_event(df, peak_col="resp_peaks", valley_col="resp_valleys", output_col="breath_event"):
    """
    Cria a coluna 'breath_event' indicando inspiração (0) e expiração (1).
    """
    df[output_col] = 0

    peaks = np.flatnonzero(df[peak_col].to_numpy() == 1)
    valleys = np.flatnonzero(df[valley_col].to_numpy() == 1)

    events = [(idx, "peak") for idx in peaks]
    events += [(idx, "valley") for idx in valleys]

    events.sort(key=lambda x: x[0])

    if not events:
        return df

    for i, (idx, event_type) in enumerate(events): # Definir o estado após o evento
        if event_type == "peak":
            value_after = 1
        else:
            value_after = 0
        if i + 1 < len(events):
            next_idx = events[i + 1][0]
            df.loc[idx + 1:next_idx - 1, output_col] = value_after

        else:
            df.loc[idx:, output_col] = value_after

    for idx, event_type in events:
        if event_type == "peak":
            df.loc[idx, output_col] = 1
        else:  # valley
            df.loc[idx, output_col] = 0

    return df


def adicionar_resp_linear_phase(df, valley_col="resp_valleys", output_col="resp_linear_phase"):
    """
    Calcula uma fase respiratória linear entre vales consecutivos.
    """
    df[output_col] = np.nan

    valleys = np.flatnonzero(df[valley_col].to_numpy() == 1)

    if len(valleys) == 0:
        df[output_col] = 0.0
        return df

    if len(valleys) == 1:
        df[output_col] = 0.0
        return df

    for i in range(len(valleys) - 1):

        start = valleys[i]
        end = valleys[i + 1]
        n = end - start
        phase = np.linspace(0, 1, n + 1)
        df.loc[start:end, output_col] = phase

    # Trecho antes do primeiro vale
    first = valleys[0]
    period = valleys[1] - valleys[0]
    if first > 0:
        indices = np.arange(0, first)
        phase = (indices - first) / period
        df.loc[0:first - 1, output_col] = phase % 1

    # Trecho depois do último vale
    last = valleys[-1]
    period = valleys[-1] - valleys[-2]
    if last < len(df) - 1:
        indices = np.arange(last + 1, len(df))
        phase = (indices - last) / period
        df.loc[last + 1:, output_col] = phase % 1

    return df

def adicionar_resp_linear_event_phase(df, peak_col="resp_peaks", valley_col="resp_valleys", output_col="resp_linear_event_phase"):
    """
    Calcula a fase respiratória linear utilizando picos e vales.
    A fase é definida como:
        vale -> pico : 0   -> 0.5
        pico -> vale : 0.5 -> 1

    """
    df[output_col] = np.nan
    peaks = np.flatnonzero(df[peak_col].to_numpy() == 1)

    valleys = np.flatnonzero(df[valley_col].to_numpy() == 1)

    events = [(idx, "peak") for idx in peaks]
    events += [(idx, "valley") for idx in valleys]
    events.sort(key=lambda x: x[0])
    if len(events) < 2:
        df[output_col] = 0.0
        return df

    for i in range(len(events) - 1):
        start_idx, start_type = events[i]
        end_idx, end_type = events[i + 1]
        n = end_idx - start_idx
        if n <= 0:
            continue
        # Vale -> Pico
        if start_type == "valley" and end_type == "peak":

            phase = np.linspace(0, 0.5, n + 1)
        # Pico -> Vale
        elif start_type == "peak" and end_type == "valley":
            phase = np.linspace(0.5, 1, n + 1)

        else:
            continue

        df.loc[start_idx:end_idx, output_col] = phase

    # Trecho antes do primeiro evento
    first_idx, first_type = events[0]
    if first_idx > 0:
        second_idx, second_type = events[1]
        interval = second_idx - first_idx

        if interval > 0:
            indices = np.arange(0, first_idx)
            if first_type == "valley":
                event_phase = 0.0
                previous_phase = 0.5

            else:  # peak
                event_phase = 0.5
                previous_phase = 0.0

            # Estimativa linear usando o intervalo seguinte
            phase = event_phase + ((indices - first_idx)/ interval* (event_phase - previous_phase)
            )
            # Mantém a fase no intervalo [0, 1]
            df.loc[0:first_idx - 1, output_col] = phase % 1

    # Trecho depois do último evento
    last_idx, last_type = events[-1]
    if last_idx < len(df) - 1:
        previous_idx, previous_type = events[-2]
        interval = last_idx - previous_idx
        if interval > 0:
            indices = np.arange(last_idx + 1, len(df))
            if last_type == "valley":
                event_phase = 0.0
                next_phase = 0.5
            else:  # peak
                event_phase = 0.5
                next_phase = 1.0
            phase = event_phase + ((indices - last_idx)/ interval* (next_phase - event_phase))
            df.loc[last_idx + 1:, output_col] = phase % 1
    return df

#coluna suavizada passa baixa
def adicionar_suv_resp_1(df, cutoff_hz=1.0, ordem=4): 
    """
    Cria a coluna 'suv_resp_1' contendo o sinal respiratório
    filtrado com passa-baixa de 1 Hz.
    """
    t = df["Time"].values
    resp = (df["resp"].interpolate().bfill().ffill().values)
    dt = np.median(np.diff(t))
    fs = 1.0 / dt
    nyquist = fs / 2.0
    if cutoff_hz >= nyquist:
        raise ValueError( f"Frequência de corte ({cutoff_hz} Hz) deve ser menor que Nyquist ({nyquist:.3f} Hz)")

    b, a = butter(ordem,cutoff_hz / nyquist,btype="low")
    df["suv_resp_1"] = filtfilt(b, a, resp)
    return df

def tempo_de_bloco_de_fatia(df, grad='gy'):
    """
    Marca o tempo de fatia (time_block_0 ... time_block_9) para cada bloco de difusão (b50=1),
    detectando ciclos no gradiente de interesse, adaptando a lógica para GX.

    Cada bloco (by=1) deve conter 10 ciclos de difusão.
    Cada ciclo gera uma marcação 0.015s após o fim do padrão.
    """

    # Parâmetros heurísticos
    tol = 1000
    amp_min_neg = -6000 - 3*tol
    amp_max_neg = -6000 + tol
    amp_min_pos =  6000 - tol
    amp_max_pos =  6000 + 3*tol
    dur_ciclo = 0.5  # s
    offset_flag = 0.015  # s após o fim do ciclo

    df = df.sort_values("Time").reset_index(drop=True)
    time = df["Time"].values
    mapa_coluna = {'M': 'gy',
                   'P': 'gx',
                   'S': 'gz',
                   'MS': 'gy',
                   'MP': 'gx',
                   'PS': 'gx'}
    g = df[mapa_coluna[grad]].values

    blocos = df.index[df["b150"] == 1].tolist()
    df["slice_mb_gradientes"] = -1

    for b, idx_inicio in enumerate(blocos):
        idx_fim = blocos[b+1] if b+1 < len(blocos) else len(df)
        t_bloco = time[idx_inicio:idx_fim]
        g_bloco = g[idx_inicio:idx_fim]

        tempos_ciclos = []
        i = 1

        #detecta até 10 ciclos dentro do bloco
        while i < len(g_bloco) - 1 and len(tempos_ciclos) < 10:

            #(GY/GZ): DESCE -> SOBE -> DESCE -> SOBE -> ZERA
            # Começa buscando um MÍNIMO LOCAL NEGATIVO
            if  (grad == 'S') or (grad == 'M'):
                is_min1 = (g_bloco[i] < g_bloco[i-1]) and (g_bloco[i] <= g_bloco[i+1]) and (amp_min_neg <= g_bloco[i] <= amp_max_neg)
                if is_min1:
                    t_min1 = t_bloco[i]
                    idx_start_search = i + 1
                    idx_lim_busca = np.where(t_bloco > t_min1 + dur_ciclo / 2)[0]
                    jmax = idx_lim_busca[0] if len(idx_lim_busca) else len(g_bloco)
                    for j in range(idx_start_search, min(jmax, len(g_bloco) - 1)):
                        if (g_bloco[j] >= g_bloco[j-1]) and \
                           (g_bloco[j] > g_bloco[j+1]) and \
                           (amp_min_pos <= g_bloco[j] <= amp_max_pos):
                            for k in range(j + 1, len(g_bloco) - 1):
                                if (g_bloco[k] < g_bloco[k-1]) and (g_bloco[k] <= g_bloco[k+1]) and \
                                   (amp_min_neg <= g_bloco[k] <= amp_max_neg):
                                    for m in range(k + 1, len(g_bloco) - 1):
                                        if (g_bloco[m] >= g_bloco[m-1]) and \
                                           (g_bloco[m] > g_bloco[m+1]) and \
                                           (amp_min_pos <= g_bloco[m] <= amp_max_pos):
                                            for n in range(m + 1, len(g_bloco) - 1):
                                                if abs(g_bloco[n]) < 500:
                                                    t_fim = t_bloco[n]
                                                    tempos_ciclos.append(t_fim + offset_flag)
                                                    i = n # O loop principal continua a partir daqui
                                                    break
                                            break
                                    break
                            break

            #GX: SOBE -> DESCE -> SOBE -> DESCE -> ZERA
            # Começa buscando um MÁXIMO LOCAL POSITIVO
            elif(grad == 'P'): # grad == 'gx'

                is_max1 = (g_bloco[i] >= g_bloco[i-1]) and (g_bloco[i] > g_bloco[i+1]) and (amp_min_pos <= g_bloco[i] <= amp_max_pos)
                if is_max1:
                    t_max1 = t_bloco[i]
                    idx_start_search = i + 1

                    idx_lim_busca = np.where(t_bloco > t_max1 + dur_ciclo / 2)[0]
                    jmax = idx_lim_busca[0] if len(idx_lim_busca) else len(g_bloco)
                    for j in range(idx_start_search, min(jmax, len(g_bloco) - 1)):
                        if (g_bloco[j] < g_bloco[j-1]) and \
                           (g_bloco[j] <= g_bloco[j+1]) and \
                           (amp_min_neg <= g_bloco[j] <= amp_max_neg):

                            for k in range(j + 1, len(g_bloco) - 1):
                                if (g_bloco[k] >= g_bloco[k-1]) and (g_bloco[k] > g_bloco[k+1]) and \
                                   (amp_min_pos <= g_bloco[k] <= amp_max_pos):

                                    for m in range(k + 1, len(g_bloco) - 1):
                                        if (g_bloco[m] < g_bloco[m-1]) and \
                                           (g_bloco[m] <= g_bloco[m+1]) and \
                                           (amp_min_neg <= g_bloco[m] <= amp_max_neg):

                                            for n in range(m + 1, len(g_bloco) - 1):
                                                if abs(g_bloco[n]) < 500:
                                                    t_fim = t_bloco[n]
                                                    tempos_ciclos.append(t_fim + offset_flag)
                                                    i = n # O loop principal continua a partir daqui
                                                    break
                                            break
                                    break
                            break

            elif grad == 'MP':
                #tol = 500
                #amp_min_neg = -4000 - 3*tol
                #amp_max_neg = -4000 + tol
                #amp_min_pos =  4000 - tol
                #amp_max_pos =  4000 + 3*tol
                #dur_ciclo = 0.5  # s
                #offset_flag = 0.015  # s após o fim do ciclo
                tol = 2000
                amp_min_neg = -6000 - 3*tol
                amp_max_neg = -6000 + 2*tol
                amp_min_pos =  5000 - 2*tol
                amp_max_pos =  5000 + 3*tol
                offset_flag = 0.015  # s após o fim do ciclo
                
                is_min1 = (g_bloco[i] < g_bloco[i-1]) and (g_bloco[i] <= g_bloco[i+1]) and (amp_min_neg <= g_bloco[i] <= amp_max_neg)
                if is_min1:
                    t_min1 = t_bloco[i]
                    idx_start_search = i + 1

                    idx_lim_busca = np.where(t_bloco > t_min1 + dur_ciclo / 2)[0]
                    jmax = idx_lim_busca[0] if len(idx_lim_busca) else len(g_bloco)

                    # Continue a busca pelo ciclo (Max1 -> Min2 -> Max2 -> Zero)
                    for j in range(idx_start_search, min(jmax, len(g_bloco) - 1)):
                        if (g_bloco[j] >= g_bloco[j-1]) and \
                           (g_bloco[j] > g_bloco[j+1]) and \
                           (amp_min_pos <= g_bloco[j] <= amp_max_pos):

                            for k in range(j + 1, len(g_bloco) - 1):
                                if (g_bloco[k] < g_bloco[k-1]) and (g_bloco[k] <= g_bloco[k+1]) and \
                                   (amp_min_neg <= g_bloco[k] <= amp_max_neg):

                                    for m in range(k + 1, len(g_bloco) - 1):
                                        if (g_bloco[m] >= g_bloco[m-1]) and \
                                           (g_bloco[m] > g_bloco[m+1]) and \
                                           (amp_min_pos <= g_bloco[m] <= amp_max_pos):

                                            for n in range(m + 1, len(g_bloco) - 1):
                                                if abs(g_bloco[n]) < 500:
                                                    t_fim = t_bloco[n]
                                                    tempos_ciclos.append(t_fim + offset_flag)
                                                    i = n # O loop principal continua a partir daqui
                                                    break
                                            break
                                    break
                            break
            elif(grad == 'MS'):
                #tol = 500
                #amp_min_neg = -4000 - 3*tol
                #amp_max_neg = -4000 + tol
                #amp_min_pos =  4000 - tol
                #amp_max_pos =  4000 + 3*tol
                #dur_ciclo = 0.5  # s
                #offset_flag = 0.015  # s após o fim do ciclo
                tol = 2000
                amp_min_neg = -6000 - 3*tol
                amp_max_neg = -6000 + 2*tol
                amp_min_pos =  5000 - 2*tol
                amp_max_pos =  5000 + 3*tol
                offset_flag = 0.015  # s após o fim do ciclo

                is_max1 = (g_bloco[i] >= g_bloco[i-1]) and (g_bloco[i] > g_bloco[i+1]) and \
                          (amp_min_pos <= g_bloco[i] <= amp_max_pos)

                if is_max1:
                    t_max1 = t_bloco[i]
                    idx_start_search = i + 1

                    # O ciclo é: Max1 -> Min1 -> Max2 -> Min2 -> Zero

                    idx_lim_busca = np.where(t_bloco > t_max1 + dur_ciclo / 2)[0]
                    jmax = idx_lim_busca[0] if len(idx_lim_busca) else len(g_bloco)

                    for j in range(idx_start_search, min(jmax, len(g_bloco) - 1)):
                        if (g_bloco[j] < g_bloco[j-1]) and \
                           (g_bloco[j] <= g_bloco[j+1]) and \
                           (amp_min_neg <= g_bloco[j] <= amp_max_neg):

                            for k in range(j + 1, len(g_bloco) - 1):
                                if (g_bloco[k] >= g_bloco[k-1]) and (g_bloco[k] > g_bloco[k+1]) and \
                                   (amp_min_pos <= g_bloco[k] <= amp_max_pos):

                                    for m in range(k + 1, len(g_bloco) - 1):
                                        if (g_bloco[m] < g_bloco[m-1]) and \
                                           (g_bloco[m] <= g_bloco[m+1]) and \
                                           (amp_min_neg <= g_bloco[m] <= amp_max_neg):

                                            for n in range(m + 1, len(g_bloco) - 1):
                                                if abs(g_bloco[n]) < 500:
                                                    t_fim = t_bloco[n]
                                                    tempos_ciclos.append(t_fim + offset_flag)
                                                    i = n # O loop principal continua a partir daqui
                                                    break
                                            break
                                    break
                            break
            else:
                tol = 2000
                amp_min_neg = -6000 - 3*tol
                amp_max_neg = -6000 + 2*tol
                amp_min_pos =  5000 - 2*tol
                amp_max_pos =  5000 + 3*tol
                dur_ciclo = 3  # s
                offset_flag = 0.015  # s após o fim do ciclo

                is_min1 = (g_bloco[i] < g_bloco[i-1]) and (g_bloco[i] <= g_bloco[i+1]) and (amp_min_neg <= g_bloco[i] <= amp_max_neg)

                if is_min1:
                    t_min1 = t_bloco[i]
                    idx_start_search = i + 1

                    idx_lim_busca = np.where(t_bloco > t_min1 + dur_ciclo / 2)[0]
                    jmax = idx_lim_busca[0] if len(idx_lim_busca) else len(g_bloco)

                    for j in range(idx_start_search, min(jmax, len(g_bloco) - 1)):
                        if (g_bloco[j] >= g_bloco[j-1]) and \
                           (g_bloco[j] > g_bloco[j+1]) and \
                           (amp_min_pos <= g_bloco[j] <= amp_max_pos):

                            for k in range(j + 1, len(g_bloco) - 1):
                                if (g_bloco[k] < g_bloco[k-1]) and (g_bloco[k] <= g_bloco[k+1]) and \
                                   (amp_min_neg <= g_bloco[k] <= amp_max_neg):

                                    for m in range(k + 1, len(g_bloco) - 1):
                                        if (g_bloco[m] >= g_bloco[m-1]) and \
                                           (g_bloco[m] > g_bloco[m+1]) and \
                                           (amp_min_pos <= g_bloco[m] <= amp_max_pos):

                                            for n in range(m + 1, len(g_bloco) - 1):
                                                if abs(g_bloco[n]) < 500:
                                                    t_fim = t_bloco[n]
                                                    tempos_ciclos.append(t_fim + offset_flag)
                                                    i = n # O loop principal continua a partir daqui
                                                    break
                                            break
                                    break
                            break

            i += 1 # Incrementa o índice do loop principal
        for i_ciclo, t_flag in enumerate(tempos_ciclos[:10]):
            idx_near = np.abs(df["Time"] - t_flag).argmin()
            df.loc[idx_near, "slice_mb_gradientes"] = i_ciclo
    return df

def cq_ppu(df, prominence_factor=0.5):
  """
  Controle de qualidade dos picos PPU.
  mark_ppu:
      nan: normal
      1: pico adicionado
      3: intervalo curto
      5: intervalo longo
  Retorna:
      dataframe atualizado
  """
  print("CONTROLE DE QUALIDADE MARCAÇÃO DA PULSAÇÃO")
  if 'mark_ppu' not in df.columns:
    df['mark_ppu'] = np.nan

  d_pico = df[df['ppu_peaks'] == 1].copy()
  picos = len(d_pico)

  if picos < 2:
    print("Poucos picos detectados.")
    return df

  period = d_pico['Time'].diff().dropna()
  media = period.mean()
  std = period.std()
  fc_media = 60 / media
  cv = std / media
  tempo_total = df['Time'].iloc[-1]
  picos_esperados = tempo_total / media

  print(f"Período médio: {media:.3f} s")
  print(f"Desvio padrão: {std:.3f} s")
  print(f"Frequência cardíaca média: {fc_media:.1f} bpm")
  print(f"CV: {cv:.4f}")

  limiar_superior = 1.5 * media
  limiar_inferior = 0.5 * media
  idx_picos = d_pico.index.to_numpy()
  tempos_picos = d_pico['Time'].to_numpy()
  intervalos_longos = 0
  intervalos_curtos = 0
  picos_perdidos = 0
  sinal = df['ppu'].values

  for i in range(1, len(idx_picos)):
    idx_ini = idx_picos[i - 1]
    idx_fim = idx_picos[i]
    dt = tempos_picos[i] - tempos_picos[i - 1]

    if dt > limiar_superior:
      intervalos_longos += 1
      mascara = (df.index >= idx_ini) & (df.index <= idx_fim)
      df.loc[mascara, 'mark_ppu'] = 5 # não sobrescrever picos adicionados posteriormente
      n_esperado = round(dt / media)
      if n_esperado > 1:
        picos_perdidos += n_esperado - 1

      trecho = sinal[idx_ini:idx_fim + 1]
      if len(trecho) > 5:
        peaks_local, props = find_peaks(trecho, prominence=np.std(trecho) * prominence_factor)

        if len(peaks_local) > 0:
          prominences = props["prominences"]
          ordem = np.argsort(prominences)[::-1]
          n_adicionar = max(1, n_esperado - 1)
          for j in ordem[:n_adicionar]:
            pico_global = idx_ini + peaks_local[j]
            if df.loc[pico_global, 'ppu_peaks'] == 0:
              df.loc[pico_global, 'ppu_peaks'] = 1
              df.loc[pico_global, 'mark_ppu'] = 1
    elif dt < limiar_inferior:
      intervalos_curtos += 1
      mascara = (df.index >= idx_ini) & (df.index <= idx_fim)
      df.loc[mascara, 'mark_ppu'] = 3

  cobertura = picos / (picos + picos_perdidos) if (picos + picos_perdidos) > 0 else 1
  print(f"Intervalos longos: {intervalos_longos}")
  print(f"Intervalos curtos: {intervalos_curtos}")
  print(f"Picos possivelmente perdidos: {picos_perdidos}")
  print(f"Cobertura estimada: {100*cobertura:.2f}%")
  print(f"Cobertura temporal: {100*picos/picos_esperados:.2f}%")
  novos_picos = np.sum(df['mark_ppu'] == 1)
  print(f"Novos picos adicionados: {novos_picos}")
  return df

def cq_temporal(df):
    """
    Controle de qualidade da marcação temporal.
    Verifica:
    1. Quantidade de ocorrências de difusao == 1
    2. Quantidade de ocorrências de cada valor de
       slice_mb_gradientes = 0,1,2,3,4,5,6,7,8,9
    """
    print("CONTROLE DE QUALIDADE DA MARCAÇÃO TEMPORAL")
    n_difusao = (df['difusao'] == 1).sum()
    if n_difusao == 60:
        print(f"[OK] difusao == 1 : {n_difusao}")
    else:
        print(
            f"[ERRO] difusao == 1 : "
            f"{n_difusao} ocorrências (esperado: 60)"
        )

    print("\nVerificando slice_mb_gradientes:")
    for grad in range(10):
        n = (df['slice_mb_gradientes'] == grad).sum()
        if n == 60:
            print(f"[OK] gradiente {grad}: {n}")
        else:
            print(
                f"[ERRO] gradiente {grad}: "
                f"{n} ocorrências (esperado: 60)"
            )

def adicionar_volume_b150(df):
    df["volumes"] = -1
    mask = df["slice_mb_gradientes"] == 0
    df.loc[mask, "volumes"] = np.arange(mask.sum())
    return df
##marcação mb
def add_slice_mb_time_b150(df, n_grupos, TR):
    df = df.copy()
    df["volume_b150"] = -1
    df["slice_mb_b150"] = -1
    dt = TR / n_grupos

    idx_volumes = df.index[df["b150"] == 1]
    for vol, idx_vol in enumerate(idx_volumes):
        df.loc[idx_vol, "volume_b150"] = vol
        t0 = df.loc[idx_vol, "Time"]
        # marca os grupos MB
        for mb in range(n_grupos):
            t_alvo = t0 + mb * dt
            idx_prox = (df["Time"] - t_alvo).abs().idxmin()
            df.loc[idx_prox, "slice_mb_b150"] = mb
    return df

def add_slice_mb_time_b0(df, n_grupos, TR):
    df = df.copy()
    df["volume_b0"] = -1
    df["slice_mb_b0"] = -1
    dt = TR / n_grupos

    idx_volumes = df.index[df["b0_final"] == 1]
    for vol, idx_vol in enumerate(idx_volumes):
        df.loc[idx_vol, "volume_b0"] = vol
        t0 = df.loc[idx_vol, "Time"]
        # marca os grupos MB
        for mb in range(n_grupos):
            t_alvo = t0 + mb * dt
            idx_prox = (df["Time"] - t_alvo).abs().idxmin()
            df.loc[idx_prox, "slice_mb_b0"] = mb
    return df


def fisio_add_flags(
    phys_path,
    aquisition_duration=183,
    repetition_time = 0.99,
    direcao = "S",
    paradigm = None,
    salvar_df = None,
    salvar_imagem=None,
    mostrar_imagem=False):

    df = load(phys_path)
    df['Time'] = np.linspace(0, aquisition_duration, len(df))
    indices_PE = find_PEy(df)
    if len(indices_PE) == 0 or len(indices_PE) <=118:
      indices_PE = find_PEy2(df)
    id_par = indices_PE[::2]
    id_impar = indices_PE[1::2]
    id_medio_bloco = indice_medio_bloco(id_par)
    
    id_medio_difusao = indice_medio_difusao(indices_PE)
    id_medio_b0 = indice_medio_b0(indices_PE)
    id_medio_b0_i = indice_medio_b0_inicio(indices_PE,df, repetition_time)
    id_medio_b0_f = indice_medio_b0_fim(indices_PE,df, repetition_time)
    limite_linhas = len(df)
    id_medio_bloco = [idx for idx in id_medio_bloco if idx < limite_linhas]
    id_medio_difusao = [idx for idx in id_medio_difusao if idx < limite_linhas]
    indice_b0_final = indice_b0_fim(indices_PE,df, repetition_time)
    df['PEgy'] = 0
    df['b0'] = 0
    df['b150'] = 0
    df['b0_final'] = 0
    df['bloco'] = 0
    df['difusao'] = 0
    df['central_b0'] = 0
    df['central_b0_inicio'] = 0
    df['central_b0_fim'] = 0
    df.loc[indices_PE, 'PEgy'] = 1
    df.loc[id_par, 'b0'] = 1
    df.loc[id_impar, 'b150'] = 1
    df.loc[indice_b0_final, 'b0_final'] = 1
    df.loc[id_medio_bloco, 'bloco'] = 1
    df.loc[id_medio_difusao, 'difusao'] = 1
    df.loc[id_medio_b0, 'central_b0'] = 1
    df.loc[id_medio_b0_i, 'central_b0_inicio'] = 1
    df.loc[id_medio_b0_f, 'central_b0_fim'] = 1

    df["ppu"] = (df["ppu"].interpolate(method="linear").bfill().ffill())
    df["resp"] = (df["resp"].interpolate(method="linear").bfill().ffill())
    df['suv_resp'] = savgol_filter(df['resp'], window_length=250, polyorder=3)
    df['suv_ppu'] = savgol_filter(df['ppu'], window_length=51, polyorder=3)
    #df = adicionar_suv_resp_1(df, cutoff_hz=1.0, ordem=4)

    df = encontrar_pulsos_ppu(df, coluna_dados='ppu', distancia_minima=200, janela_suavizacao=15, altura_minima=None, janela_busca=100, sensibilidade=0.2, fallback_offset=65)
    df = marcar_picos_vales_resp(df, resp_column= 'resp', interpolate = True, threshold_amp = 0.2)

    #Variaveis de bloco de atividade
    if paradigm is not None:
        if (df['PEgy'] == 1).any():
            t0 = df.loc[df['PEgy'] == 1, 'Time'].iloc[0]
        else:
            t0 = df['Time'].iloc[0]
        t_livre = paradigm[0]
        t_passada = paradigm[1]
        df = paradigm_task(df, t_livre, t_passada, t0=t0)

    base_dir = os.path.dirname(phys_path)
    base_dir = os.path.join(base_dir, "Analysis")
    os.makedirs(base_dir, exist_ok=True)
    #tempos mb 
    df = tempo_de_bloco_de_fatia(df, grad=direcao)
    cq_temporal(df)

    df = add_slice_mb_time_b150(df, n_grupos=10, TR=repetition_time)
    df = add_slice_mb_time_b0(df, n_grupos=10, TR=repetition_time)
    df = adicionar_volume_b150(df)

    if salvar_imagem is not None:
        image_name = salvar_imagem
        save_image = True
    else:
        image_name = "physiological_marked"
        save_image = False

    if save_image or mostrar_imagem:
        plot_gradients(
            df,
            nrows=10,
            figsize=(15, 20),
            title=image_name,
            save=save_image,  
            show=mostrar_imagem,
            path=base_dir)

    df = df.drop(columns=['v1raw', 'v2raw','vsc', 'v1', 'v2', 'mark', 'mark2'])
    if salvar_df is not None:
        save_dir = os.path.join(base_dir, f"{salvar_df}.csv")
        df.to_csv(save_dir)
        print(f"dados salvos em: {save_dir}")
    return df

def inspecao_visual_marcacao(csv_path):
    df = pd.read_csv(csv_path)
    df = cq_ppu(df)
    df.to_csv(csv_path)
    df = editar_respiracao(caminho_csv = csv_path, caminho_saida=csv_path)
    return df

def fisio_add_phase(csv_path):
    df = pd.read_csv(csv_path)
    #fase pulsação
    df = quantificar_fase_ppu(df)
    df = quantificar_fase_ppu_rr(df)
    df =  calcular_fase_ppu_dtw(df,n_media=10)

    #fase respiratória
    df = adicionar_breath_event(df,peak_col="resp_peaks",valley_col="resp_valleys",output_col="breath_event")
    df = adicionar_resp_linear_event_phase(df,peak_col="resp_peaks",valley_col="resp_valleys",output_col="resp_linear_event_phase")
    df = adicionar_resp_linear_phase(df,valley_col="resp_valleys",output_col="resp_linear_phase")
    df = calcular_fase_respiratoria_dtw(df_input=df,referencia="all")
    

    df.to_csv(csv_path)
    return df

def fisio_t1(
    phys_path,
    aquisition_time=183,
    salvar_df = "physiological_t1",):
    df = load(phys_path)
    df['Time'] = np.linspace(0, aquisition_time, len(df))
    df['suv_resp'] = savgol_filter(df['resp'], window_length=250, polyorder=3)
    df['task'] = np.ones(len(df), dtype=int)
    df['ciclo'] = np.ones(len(df), dtype=int)
    df = encontrar_pulsos_ppu(df, coluna_dados='ppu', distancia_minima=200, janela_suavizacao=15, altura_minima=None, janela_busca=100, sensibilidade=0.2, fallback_offset=65)

    base_dir = os.path.dirname(phys_path)
    base_dir = os.path.join(base_dir, "Analysis")
    os.makedirs(base_dir, exist_ok=True)

    df = df.drop(columns=['v1raw', 'v2raw','vsc', 'v1', 'v2', 'mark', 'mark2', 'gx', 'gy', 'gz'])
    if salvar_df is not None:
        save_dir = os.path.join(base_dir, f"{salvar_df}.csv")
        df.to_csv(save_dir)
        print(f"dados salvos em: {save_dir}")
    return df

"""json_path = '/run/media/joao-oliveira/PortableSSD/dynDWI_V2/sub032/dynDWI_S/ScanPsaLog.log'
csv_path = '/run/media/joao-oliveira/PortableSSD/dynDWI_V2/sub032/dynDWI_S/Analysis/Physiological_marked.csv'
df =fisio_add_flags(
    phys_path=json_path,
    aquisition_duration=183,
    repetition_time = 0.98,
    direcao = "S",
    paradigm = [30,30],
    salvar_df = 'Physiological_marked',
    mostrar_imagem=True,
    )
df = inspecao_visual_marcacao(csv_path)
df = fisio_add_phase(csv_path)
print(df.columns)
#df = pd.read_csv("/run/media/joao-oliveira/PortableSSD/dynDWI_V2/epi003/dynDWI_MS/Analysis/physiological_marked.csv")
#df = tempo_de_bloco_de_fatia(df, grad='MS')
#cq_temporal(df)
"""