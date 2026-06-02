import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import correlate
import numpy as np
import matplotlib.pyplot as plt

def poincare_sliding_window(
    df,
    janela=60,
    passo=5,
    tempo_col="Time",
    pico_col="pico_flag",
    min_rr=3):
    """
    Calcula SD1 e SD2 em janelas deslizantes.

    Parâmetros
    ----------
    df : DataFrame
        Deve conter:
            Time      -> tempo em segundos
            pico_flag -> 1 nos picos cardíacos

    janela : float
        Tamanho da janela (s)

    passo : float
        Deslocamento da janela (s)

    min_rr : int
        Número mínimo de intervalos RR para calcular SD1/SD2

    Retorna
    -------
    resultados : DataFrame
        tempo_central, sd1, sd2
    """

    t_inicio = df[tempo_col].min()
    t_fim = df[tempo_col].max()

    tempos_centro = []
    sd1_list = []
    sd2_list = []

    inicio_janela = t_inicio

    while inicio_janela + janela <= t_fim:

        fim_janela = inicio_janela + janela

        sub = df[
            (df[tempo_col] >= inicio_janela)
            & (df[tempo_col] < fim_janela)
        ]

        tempos_picos = sub.loc[
            sub[pico_col] == 1,
            tempo_col
        ].values

        rr = np.diff(tempos_picos)

        if len(rr) >= min_rr:

            rr_n = rr[:-1]
            rr_n1 = rr[1:]

            diff_rr = rr_n1 - rr_n

            var_rr = np.var(rr, ddof=1)
            var_diff = np.var(diff_rr, ddof=1)

            sd1 = np.sqrt(0.5 * var_diff)

            termo = 2 * var_rr - 0.5 * var_diff

            sd2 = np.sqrt(max(termo, 0))

        else:
            sd1 = np.nan
            sd2 = np.nan

        tempos_centro.append(inicio_janela + janela / 2)
        sd1_list.append(sd1)
        sd2_list.append(sd2)

        inicio_janela += passo

    resultados = pd.DataFrame({
        "tempo_central": tempos_centro,
        "sd1": sd1_list,
        "sd2": sd2_list
    })

    # -----------------------------
    # Plot SD1
    # -----------------------------
    plt.figure(figsize=(10, 4))
    plt.plot(
        resultados["tempo_central"],
        resultados["sd1"],
        "-o",
        markersize=3
    )
    plt.xlabel("Tempo (s)")
    plt.ylabel("SD1 (s)")
    plt.title("Evolução temporal do SD1")
    plt.grid(True)
    plt.tight_layout()

    # -----------------------------
    # Plot SD2
    # -----------------------------
    plt.figure(figsize=(10, 4))
    plt.plot(
        resultados["tempo_central"],
        resultados["sd2"],
        "-o",
        markersize=3
    )
    plt.xlabel("Tempo (s)")
    plt.ylabel("SD2 (s)")
    plt.title("Evolução temporal do SD2")
    plt.grid(True)
    plt.tight_layout()

    plt.show()

    return resultados

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def poincare_sliding_windows(
    df,
    janelas,
    passo=5,
    tempo_col="Time",
    pico_col="pico_flag",
    min_rr=3,):
    """
    Calcula SD1 e SD2 em múltiplas janelas deslizantes.

    Parâmetros
    ----------
    df : DataFrame

    janelas : float ou lista
        Tamanho(s) da janela em segundos.

    passo : float
        Deslocamento da janela em segundos.

    Retorna
    -------
    resultados : dict
        resultados[janela] = DataFrame com:
            tempo_central
            sd1
            sd2
    """

    if np.isscalar(janelas):
        janelas = [janelas]

    t_inicio = df[tempo_col].min()
    t_fim = df[tempo_col].max()

    resultados = {}

    for janela in janelas:

        tempos_centro = []
        sd1_list = []
        sd2_list = []

        inicio_janela = t_inicio

        while inicio_janela + janela <= t_fim:

            fim_janela = inicio_janela + janela

            sub = df[
                (df[tempo_col] >= inicio_janela)
                & (df[tempo_col] < fim_janela)
            ]

            tempos_picos = sub.loc[
                sub[pico_col] == 1,
                tempo_col
            ].values

            rr = np.diff(tempos_picos)

            if len(rr) >= min_rr:

                rr_n = rr[:-1]
                rr_n1 = rr[1:]

                diff_rr = rr_n1 - rr_n

                var_rr = np.var(rr, ddof=1)
                var_diff = np.var(diff_rr, ddof=1)

                sd1 = np.sqrt(0.5 * var_diff)

                termo = 2 * var_rr - 0.5 * var_diff
                sd2 = np.sqrt(max(termo, 0))

            else:
                sd1 = np.nan
                sd2 = np.nan

            tempos_centro.append(
                inicio_janela + janela / 2
            )
            sd1_list.append(sd1)
            sd2_list.append(sd2)

            inicio_janela += passo

        resultados[janela] = pd.DataFrame({
            "tempo_central": tempos_centro,
            "sd1": sd1_list,
            "sd2": sd2_list
        })
    
    resultados_norm = {}
    for janela, res in resultados.items():

        res_norm = res.copy()

        media_sd1 = np.nanmean(res["sd1"])
        media_sd2 = np.nanmean(res["sd2"])

        res_norm["sd1"] = (res["sd1"] - np.nanmean(res["sd1"])) / np.nanstd(res["sd1"])
        res_norm["sd2"] = (res["sd2"] - np.nanmean(res["sd2"])) / np.nanstd(res["sd2"])
        
        resultados_norm[janela] = res_norm
    # =====================
    # Plot SD1
    # =====================

    plt.figure(figsize=(12, 5))

    for janela, res in resultados_norm.items():
        plt.plot(
            res["tempo_central"],
            res["sd1"],
            label=f"{janela}s"
        )

    plt.xlabel("Tempo (s)")
    plt.ylabel("SD1 (s)")
    plt.title("SD1 ao longo do tempo")
    plt.legend(title="Janela")
    plt.grid(True)
    plt.tight_layout()

    # =====================
    # Plot SD2
    # =====================

    plt.figure(figsize=(12, 5))

    for janela, res in resultados_norm.items():
        plt.plot(
            res["tempo_central"],
            res["sd2"],
            label=f"{janela}s"
        )

    plt.xlabel("Tempo (s)")
    plt.ylabel("SD2 (s)")
    plt.title("SD2 ao longo do tempo")
    plt.legend(title="Janela")
    plt.grid(True)
    plt.tight_layout()

    plt.show()

    return resultados

sub = "sub018"
dire = "S"

df = pd.read_csv(f'/media/joao-oliveira/PortableSSD/dynDWI_V2/{sub}/dynDWI_{dire}/Analysis/physiological_marked.csv')

res = poincare_sliding_windows(
    df,
    janelas=[20, 25, 30, 35,40],
    passo=5
)
def plot_res(res):
    # Dados da janela de 25 s
    r = res[25]

    # Remover pontos com NaN
    mask = (
        np.isfinite(r["sd1"]) &
        np.isfinite(r["sd2"])
    )

    sd1 = r.loc[mask, "sd1"].values
    sd2 = r.loc[mask, "sd2"].values

    # Remover média
    sd1 = sd1 - np.mean(sd1)
    sd2 = sd2 - np.mean(sd2)

    # Correlação cruzada
    cc = correlate(sd1, sd2, mode="full")

    lags = np.arange(-len(sd1) + 1, len(sd1))

    lag_max = lags[np.argmax(cc)]

    print(f"Lag máximo = {lag_max} janelas")
    print(f"Lag máximo = {lag_max * 5:.1f} s")

    plt.figure(figsize=(8,4))

    plt.plot(lags * 5, cc)

    plt.axvline(
        lag_max * 5,
        linestyle="--"
    )

    plt.xlabel("Lag (s)")
    plt.ylabel("Correlação cruzada")
    plt.title("SD1 × SD2 (janela = 25 s)")
    plt.grid(True)

    plt.show()
plot_res(res)

import numpy as np
import matplotlib.pyplot as plt


def plot_periodo_cardiaco(
    df,
    tempo_col="Time",
    pico_col="pico_flag"
):
    """
    Plota o período cardíaco (RR ou PP) ao longo do tempo.

    Parâmetros
    ----------
    df : pandas.DataFrame
        Deve conter:
            Time
            pico_flag (1 nos picos)
    """

    tempos_picos = df.loc[
        df[pico_col] == 1,
        tempo_col
    ].values

    if len(tempos_picos) < 2:
        raise ValueError("Menos de 2 picos encontrados.")

    rr = np.diff(tempos_picos)

    # associar cada RR ao segundo pico
    tempos_rr = tempos_picos[1:]

    plt.figure(figsize=(12, 4))

    plt.plot(
        tempos_rr,
        rr,
        '-o',
        markersize=3
    )

    plt.xlabel("Tempo (s)")
    plt.ylabel("Período cardíaco (s)")
    plt.title("Período cardíaco ao longo do tempo")
    plt.grid(True)
    plt.tight_layout()

    plt.show()

    return tempos_rr, rr

tempos_rr, rr = plot_periodo_cardiaco(df)