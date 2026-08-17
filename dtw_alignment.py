"""
Módulo de Alinhamento por DTW para Sinais Respiratórios e Cardíacos
Implementa funções para cálculo de fase utilizando Dynamic Time Warping (DTW)
para sinais respiratórios (resp) e de pulsação (ppu).
"""

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
from scipy.interpolate import interp1d
from fastdtw import fastdtw

def fase_dtw(sinal, sinal_ref, normalizar= False):
    """
    Calcula a fase de um sinal usando DTW.
    Parâmetros
        Sinal a ser alinhado, sinal de referência.
        normalizar:
            True -> força fase entre 0 e 1 (pulso completo)
            False -> preserva a posição na referência (pulso parcial)
    Retorna: Fase calculada para cada ponto do sinal.
    """
    sinal = np.asarray(sinal).ravel().astype(float)
    sinal_ref = np.asarray(sinal_ref).ravel().astype(float)

    if len(sinal) < 2 or len(sinal_ref) < 2:
        return np.zeros(len(sinal))

    mn, mx = sinal.min(), sinal.max()
    if mx > mn:
        sinal = (sinal - mn) / (mx - mn)

    mn, mx = sinal_ref.min(), sinal_ref.max()
    if mx > mn:
        sinal_ref = (sinal_ref - mn) / (mx - mn)

    # DTW
    _, path = fastdtw(sinal, sinal_ref)

    # Mapeamento de índices
    mapeamento = {}
    for idx_sinal, idx_ref in path:
        mapeamento.setdefault(idx_sinal, []).append(idx_ref)

    # Cálculo da fase
    fase = np.zeros(len(sinal))
    for i in range(len(sinal)):
        if i in mapeamento:
            fase[i] = np.mean(mapeamento[i]) / (len(sinal_ref) - 1)
        elif i > 0:
            fase[i] = fase[i - 1]

    if normalizar:
        mn, mx = fase.min(), fase.max()
        if mx > mn:
            fase = (fase - mn) / (mx - mn)

    return fase

def construir_referencia_resp_dtw(resp_dtw, recortes, modo= "base", task= None):
    """
    Constrói o ciclo respiratório de referência.
    Parâmetros
    resp_dtw = Sinal respiratório suavizado.
    recortes = Lista de tuplas (ini, fim) de cada ciclo.
    modo:'base' -> primeiro ciclo
        'media' -> média dos primeiros ciclos
        'padrao' -> referência sintética (seno)
    task =Série com marcadores de tarefa para selecionar ciclo base.
    """
    if modo not in ["base", "media", "padrao"]:
        raise ValueError("modo deve ser 'base', 'media' ou 'padrao'")

    # Selecionar índice de referência
    indice_ref = 0
    if task is not None:
        for i, (ini, _) in enumerate(recortes):
            if task.iloc[ini] == 1:
                indice_ref = i
                break

    ini_ref, fim_ref = recortes[indice_ref]

    # Modo base
    if modo == "base":
        return resp_dtw[ini_ref:fim_ref + 1]

    # Modo padrão (senoide)
    if modo == "padrao":
        n_ref = fim_ref - ini_ref + 1
        phi = np.linspace(0, 1, n_ref)
        return 0.5 * (1 - np.cos(2 * np.pi * phi))

    # Modo média
    ciclos = recortes[indice_ref:min(indice_ref + 5, len(recortes))]
    if len(ciclos) == 0:
        raise ValueError("Nenhum ciclo disponível para média.")

    tamanho = int(np.mean([fim - ini + 1 for ini, fim in ciclos]))
    x_novo = np.linspace(0, 1, tamanho)
    referencia = []

    for ini, fim in ciclos:
        sinal = resp_dtw[ini:fim + 1]
        x = np.linspace(0, 1, len(sinal))
        f = interp1d(x, sinal, kind="linear")
        referencia.append(f(x_novo))

    return np.mean(referencia, axis=0)

def preencher_extremos_respiracao_resp(df, resp_dtw, sinal_ref, coluna_fase):
    """
    Completa a fase respiratória antes do primeiro vale e após o último
    utilizando DTW com o ciclo de referência."""

    indices_vales = df.index[df["resp_valleys"] == 1].tolist()

    if len(indices_vales) == 0:
        return df
    primeiro_vale = indices_vales[0]
    if primeiro_vale > 0:
        sinal_inicio = resp_dtw[:primeiro_vale]
        if len(sinal_inicio) >= 2:
            fase_inicio = fase_dtw(sinal_inicio, sinal_ref)
            df.loc[:primeiro_vale - 1, coluna_fase] = fase_inicio

    ultimo_vale = indices_vales[-1]
    if ultimo_vale < len(df) - 1:
        sinal_final = resp_dtw[ultimo_vale:]
        if len(sinal_final) >= 2:
            fase_final = fase_dtw(sinal_final, sinal_ref)
            df.loc[ultimo_vale:, coluna_fase] = fase_final

    return df

def calcular_fase_respiratoria_dtw(df_input, referencia= "base"):
    """
    Calcula a fase respiratória utilizando DTW.
        'base' -> primeiro ciclo (task==1 quando existir)
        'media' -> média dos cinco primeiros ciclos
        'padrao' -> referência sintética
        'all' -> calcula todas as referências
    """
    if referencia == "all":
        referencias = ["base", "media", "padrao"]
    elif referencia in ["base", "media", "padrao"]:
        referencias = [referencia]
    else:
        raise ValueError("referencia deve ser 'base', 'media', 'padrao' ou 'all'.")

    df = df_input.copy().reset_index(drop=True)
    n = len(df)

    # Suavização do sinal
    window = min(31, n)
    if window % 2 == 0:
        window -= 1
    if window < 5:
        window = 5 if n >= 5 else n
    if window % 2 == 0:
        window -= 1

    resp_dtw = savgol_filter(df["resp"].values, window_length=max(5, window), polyorder=3) if window >= 5 else df["resp"].values.copy()

    indices_vales = df.index[df["resp_valleys"] == 1].tolist()
    if len(indices_vales) < 2:
        raise ValueError("São necessários pelo menos dois vales.")

    # Construir ciclos vale -> vale
    recortes = [(indices_vales[i], indices_vales[i + 1]) for i in range(len(indices_vales) - 1)]

    for ref in referencias:
        sinal_ref = construir_referencia_resp_dtw(
            resp_dtw=resp_dtw,
            recortes=recortes,
            modo=ref,
            task=df["task"] if "task" in df.columns else None)

        coluna_fase = f"resp_dtw_phase_{ref}"
        df[coluna_fase] = np.nan

        # DTW ciclo a ciclo
        for ini, fim in recortes:
            sinal = resp_dtw[ini:fim + 1]
            if len(sinal) >= 2:
                df.loc[ini:fim, coluna_fase] = fase_dtw(sinal, sinal_ref)

        df = preencher_extremos_respiracao_resp(df, resp_dtw, sinal_ref, coluna_fase)
    return df

def construir_referencia_ppu_dtw(ppu, recortes, n_media = 5):
    """
    Constrói o pulso de referência para o DTW a partir da média dos primeiros pulsos.
    """
    if len(recortes) == 0:
        raise ValueError("Nenhum pulso disponível.")

    pulsos = recortes[:min(n_media, len(recortes))]

    # Comprimento alvo
    tamanho = max(int(np.mean([fim - ini + 1 for ini, fim in pulsos])), 2)
    x_novo = np.linspace(0, 1, tamanho)

    # Interpolar e normalizar cada pulso
    pulsos_interp = []
    for ini, fim in pulsos:
        sinal = np.asarray(ppu[ini:fim + 1], dtype=float)
        mn, mx = sinal.min(), sinal.max()
        if mx > mn:
            sinal = (sinal - mn) / (mx - mn)
        if len(sinal) >= 2:
            f = interp1d(np.linspace(0, 1, len(sinal)), sinal, kind="linear")
            pulsos_interp.append(f(x_novo))

    if len(pulsos_interp) == 0:
        raise ValueError("Não foi possível construir o pulso de referência.")

    return np.mean(pulsos_interp, axis=0)

def preencher_pulsos_parciais(df, sinal_ref, coluna_fase = "ppu_dtw_phase"):
    """
    Completa a fase cardíaca antes do primeiro pulso e após o último
    utilizando DTW com o pulso de referência.
    """
    indices_inicio = df.index[df["start_ppu_pulse"] == 1].tolist()
    if len(indices_inicio) == 0:
        return df

    primeiro_inicio = indices_inicio[0]
    if primeiro_inicio > 0:
        sinal_inicio = df.loc[:primeiro_inicio - 1, "ppu"].values
        if len(sinal_inicio) >= 2:
            df.loc[:primeiro_inicio - 1, coluna_fase] = fase_dtw(sinal_inicio, sinal_ref)

    ultimo_inicio = indices_inicio[-1]
    if ultimo_inicio < len(df) - 1:
        sinal_final = df.loc[ultimo_inicio:, "ppu"].values
        if len(sinal_final) >= 2:
            df.loc[ultimo_inicio:, coluna_fase] = fase_dtw(sinal_final, sinal_ref)
    return df

def calcular_fase_ppu_dtw(df_input, n_media = 5):
    df = df_input.copy().reset_index(drop=True)

    indices_inicio = df.index[df["start_ppu_pulse"] == 1].tolist()
    if len(indices_inicio) == 0:
        raise ValueError("Nenhum pulso encontrado.")

    recortes = [(indices_inicio[i], indices_inicio[i + 1] - 1) for i in range(len(indices_inicio) - 1)]
    recortes.append((indices_inicio[-1], len(df) - 1))

    sinal_ref = construir_referencia_ppu_dtw(df["ppu"].values, recortes, n_media)
    coluna_fase = "ppu_dtw_phase"
    df[coluna_fase] = np.nan

    for ini, fim in recortes:
        sinal = df.loc[ini:fim, "ppu"].values
        if len(sinal) >= 2:
            df.loc[ini:fim, coluna_fase] = fase_dtw(sinal, sinal_ref)

    df = preencher_pulsos_parciais(df, sinal_ref, coluna_fase)

    return df