from scipy.signal import savgol_filter
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
    # Se só temos um ponto médio, usa a distância entre os dois primeiros pares
    distancia_media = id[1] - id[0] if len(id) > 1 else 0

  # Adicionar o último ponto médio baseado na distância média
  if len(id) > 0:
    if len(indices_medios) > 0:
      # Último ponto médio = último ponto médio calculado + distância média
      ultimo_medio = indices_medios[-1] + distancia_media
    else:
      # Se não há pontos médios calculados (apenas 1 par)
      ultimo_medio = id[0] + distancia_media / 2
    indices_medios.append(ultimo_medio)
  indices_medios_inteiros = [int(round(medio)) for medio in indices_medios]
  return indices_medios_inteiros

def indice_medio_difusao(id):
  indices_restantes = id[1:]
  indices_medios = []
  for i in range(0, len(indices_restantes) - 1, 2):
    if i + 1 < len(indices_restantes):
      # Média entre indices_restantes[i+1] e indices_restantes[i]
      # (padrão: (2,1), (4,3), (6,5), ...)
      indice_medio = (indices_restantes[i+1] + indices_restantes[i]) / 2
      indices_medios.append(indice_medio)

  # Adicionar último ponto médio se necessário
  if len(indices_medios) > 0:
    # Calcular distância média entre pontos médios
    if len(indices_medios) > 1:
      distancias = [indices_medios[j+1] - indices_medios[j] for j in range(len(indices_medios)-1)]
      distancia_media = sum(distancias) / len(distancias)
    else:
      # Se só tem um ponto médio, usa distância dos índices originais
      distancia_media = indices_restantes[2] - indices_restantes[0] if len(indices_restantes) >= 3 else indices_medios[0]

    ultimo_medio = indices_medios[-1] + distancia_media
    indices_medios.append(ultimo_medio)

  indices_medios_inteiros = [int(round(medio)) for medio in indices_medios]
  return indices_medios_inteiros

def encontrar_pulsos_ppu(df, coluna_dados, distancia_minima=200, janela_suavizacao=15, altura_minima=None, janela_busca=100, sensibilidade=0.2, fallback_offset=65):
    """
    Args:
        df (pd.DataFrame): O DataFrame com os dados de pulsação.
        coluna_dados (str): O nome da coluna com os dados de pulsação (e.g., 'ppu').
        distancia_minima (int): A distância mínima entre os picos em número de amostras.
        janela_suavizacao (int): O tamanho da janela do filtro de média móvel para suavizar os dados.
        altura_minima (float): A altura mínima que um pico deve ter para ser considerado.
        janela_busca (int): A janela de busca (em amostras) para trás a partir de cada pico.
        sensibilidade (float): Fator de sensibilidade para encontrar o início da subida.
        fallback_offset (int): Offset a ser usado como início do pulso se a busca falhar.
    Returns:
        pd.DataFrame: O DataFrame original com as novas colunas de flag adicionadas.
    """

    df_copy = df.copy()
    dados_originais = df_copy[coluna_dados].values
    dados_suavizados = pd.Series(dados_originais).rolling(window=janela_suavizacao, center=True).mean().fillna(method='bfill').fillna(method='ffill').values

    # Determinar a altura mínima para picos (se não for especificado)
    if altura_minima is None:
        altura_minima = np.percentile(dados_suavizados, 75)

    # encontrar os picos sistólicos nos dados suavizados
    picos, _ = find_peaks(dados_suavizados, height=altura_minima, distance=distancia_minima)

    # Encontrar o início da subida para cada pulso com lógica de fallback
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

    # Corrigir a diferença de tamanho entre picos e inícios, se houver
    if len(picos) != len(inicios_pulsos):
        print("Aviso: O número de picos e inícios de pulsos não coincide. Corrigindo...")
        if len(picos) > len(inicios_pulsos):
            for i in range(len(inicios_pulsos), len(picos)):
                inicios_pulsos.append(max(0, picos[i] - fallback_offset))
        else:
            inicios_pulsos = inicios_pulsos[:len(picos)]

    # Delimitar os pulsos com base nos inícios encontrados
    # Criar uma lista para os finais dos pulsos
    finais_pulsos = inicios_pulsos[1:] + [len(df_copy) - 1] # O final do último pulso é o último índice do DataFrame

    # Criar e preencher as novas colunas de flag no DataFrame
    df_copy['pico_flag'] = 0
    df_copy['inicio_pulso_flag'] = 0
    df_copy['fim_pulso_flag'] = 0

    # Marcar os picos
    if len(picos) > 0:
        df_copy.loc[picos, 'pico_flag'] = 1

    # Marcar os inícios dos pulsos
    if len(inicios_pulsos) > 0:
        df_copy.loc[inicios_pulsos, 'inicio_pulso_flag'] = 1

    # Marcar os finais dos pulsos
    if len(finais_pulsos) > 0:
        df_copy.loc[finais_pulsos, 'fim_pulso_flag'] = 1

    # 8. Retornar o DataFrame modificado
    return df_copy

def quantificar_fase_ppu(df):
    """
    Calcula a fase (0–1) de cada instante em relação ao ciclo de pulso mais próximo,
    e armazena o resultado em uma nova coluna 'fase_flag'.
    A fase é calculada como:
        fase = (tempo_atual - tempo_inicio_pulso) / (tempo_fim_pulso - tempo_inicio_pulso)
    Args:
        df (pd.DataFrame): Deve conter as colunas 'Time', 'inicio_pulso_flag', 'fim_pulso_flag'.

    Returns:
        pd.DataFrame: O DataFrame original com a nova coluna 'fase_flag' adicionada.
    """
    df_copy = df.copy()
    df_copy['fase_flag'] = np.nan

    # Identificar os tempos de início e fim dos pulsos
    tempos_inicio = df_copy.loc[df_copy['inicio_pulso_flag'] == 1, 'Time'].values
    tempos_fim = df_copy.loc[df_copy['fim_pulso_flag'] == 1, 'Time'].values

    num_pulsos = min(len(tempos_inicio), len(tempos_fim))

    # Garantir que só consideramos pares válidos de (inicio, fim)
    tempos_inicio = tempos_inicio[:num_pulsos]
    tempos_fim = tempos_fim[:num_pulsos]

    # Percorrer todos os instantes de tempo no DataFrame
    for i, tempo_atual in enumerate(df_copy['Time'].values):
        # Encontrar o último início de pulso antes (ou igual) a este tempo
        idx_pulso = np.where(tempos_inicio <= tempo_atual)[0]

        if len(idx_pulso) > 0:
            idx_pulso = idx_pulso[-1]
            # Verificar se o tempo atual está dentro do intervalo desse pulso
            if idx_pulso < num_pulsos and tempo_atual <= tempos_fim[idx_pulso]:
                t_ini = tempos_inicio[idx_pulso]
                t_fim = tempos_fim[idx_pulso]
                duracao = t_fim - t_ini
                if duracao > 0:
                    fase = (tempo_atual - t_ini) / duracao
                    df_copy.loc[i, 'fase_flag'] = fase
                else:
                    df_copy.loc[i, 'fase_flag'] = np.nan
            else:
                # Fora de um pulso ativo
                df_copy.loc[i, 'fase_flag'] = np.nan
        else:
            df_copy.loc[i, 'fase_flag'] = np.nan

    return df_copy

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

        # Gradientes principais
        ax.plot(t, df['gx'].iloc[start:end].values, label='gx', color=plt.cm.tab10(0), alpha=0.9)
        ax.plot(t, df['gy'].iloc[start:end].values, label='gy', color=plt.cm.tab10(1), alpha=0.6)
        ax.plot(t, df['gz'].iloc[start:end].values, label='gz', color=plt.cm.tab10(2), alpha=0.5)

        # Flags principais
        mask_PEgy = (df['PEgy'].iloc[start:end] == 1)
        ax.plot(t[mask_PEgy], [2000] * mask_PEgy.sum(), 'o', label='PEgy', color=plt.cm.tab10(3))

        # Picos PPU (Implementação da requisição)
        if 'pico_flag' in df.columns and 'ppu' in df.columns:
            mask_pulso = (df['pico_flag'].iloc[start:end] == 1)

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

    # duração total de um ciclo: livre → passada
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
def computar_1_2_derivadas_ppu(df):
    window_length = 51  #(deve ser ímpar)
    polyorder = 3 # ordem do polinômio
    window_length_deriv = 51
    # Aplicar suavização no sinal ppu original
    if len(df) > window_length:
        try:
            # Suavizar o sinal ppu
            ppu_smooth = savgol_filter(df['ppu'].fillna(method='ffill').values, 
                                    window_length, polyorder)
        except:
            # Fallback se a suavização falhar
            ppu_smooth = df['ppu'].fillna(method='ffill').values
    else:
        # Sem suavização se dados insuficientes
        ppu_smooth = df['ppu'].fillna(method='ffill').values

    ppu_first_derivative = np.gradient(ppu_smooth)
    ppu_first_derivative = savgol_filter(ppu_first_derivative, window_length_deriv, polyorder)
    ppu_second_derivative = np.gradient(ppu_first_derivative)
    ppu_second_derivative = savgol_filter(ppu_second_derivative, window_length_deriv, polyorder)

    return ppu_first_derivative, ppu_second_derivative
def fisio_add_flags(
    phys_path,
    aquisition_duration=183,
    paradigm = None,
    derivadas_ppu=False,
    salvar_df = None,
    salvar_imagem=None,
    mostrar_imagem=False,
    ):
    df = load(phys_path)
    df['Time'] = np.linspace(0, aquisition_duration, len(df))
    indices_PE = find_PEy(df)
    if len(indices_PE) == 0 or len(indices_PE) == 114:
      indices_PE = find_PEy2(df)

    id_par = indices_PE[::2]
    id_impar = indices_PE[1::2]
    id_medio_bloco = indice_medio_bloco(id_par)
    id_medio_difusao = indice_medio_difusao(indices_PE)

    df['PEgy'] = 0
    df['b0'] = 0
    df['b150'] = 0
    df['bloco'] = 0
    df['difusao'] = 0
    df.loc[indices_PE, 'PEgy'] = 1
    df.loc[id_par, 'b0'] = 1
    df.loc[id_impar, 'b150'] = 1
    df.loc[id_medio_bloco, 'bloco'] = 1
    df.loc[id_medio_difusao, 'difusao'] = 1
    df['suv_resp'] = savgol_filter(df['resp'], window_length=250, polyorder=3)
    df['suv_ppu'] = savgol_filter(df['ppu'], window_length=51, polyorder=3)

    df = encontrar_pulsos_ppu(df, coluna_dados='ppu', distancia_minima=200, janela_suavizacao=15, altura_minima=None, janela_busca=100, sensibilidade=0.2, fallback_offset=65)
    df = quantificar_fase_ppu(df)
    #Variaveis de bloco de atividade
    if paradigm is not None:
        if (df['PEgy'] == 1).any():
            t0 = df.loc[df['PEgy'] == 1, 'Time'].iloc[0]
        else:
            t0 = df['Time'].iloc[0]
        t_livre = paradigm[0]
        t_passada = paradigm[1]
        df = paradigm_task(df, t_livre, t_passada, t0=t0)

    if derivadas_ppu:
        df['ppu_first_derivative'], df['ppu_second_derivative'] = computar_1_2_derivadas_ppu(df)
        df['ppu_first_derivative'] = df['ppu_first_derivative'].fillna(0)
        df['ppu_second_derivative'] = df['ppu_second_derivative'].fillna(0)

    base_dir = os.path.dirname(phys_path)
    base_dir = os.path.join(base_dir, "Analysis")
    os.makedirs(base_dir, exist_ok=True)
    if mostrar_imagem:
        if salvar_imagem is None:
            image_name = "physiological_marked"
            save_image = False
        else:
            image_name = salvar_imagem
            save_image = True

        plot_gradients(
            df,
            nrows=10,
            figsize=(15, 20),
            title=image_name,
            save=save_image,
            show=mostrar_imagem,
            path=base_dir
        )

    df = df.drop(columns=['v1raw', 'v2raw','vsc', 'v1', 'v2', 'mark', 'mark2'])
    if salvar_df is not None:
        save_dir = os.path.join(base_dir, f"{salvar_df}.csv")
        df.to_csv(save_dir)
        print(f"dados salvos em: {save_dir}")
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
