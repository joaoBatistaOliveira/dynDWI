import os
import numpy as np
import pandas as pd
import nibabel as nib
import subprocess



def selecionar_volumes_dwi(dynDWI, bvals):
    indices_dwi = np.where(bvals > 10)[0]
    dwi = dynDWI[..., indices_dwi]
    return dwi, indices_dwi


def tipo_agrupamento_fase(nfase, tipo):
    """
    Retorna uma lista de intervalos de fase (inicio, fim)
    no domínio [0,1), considerando circularidade.

    Tipos:
    -------
    continuo:
        Divide o ciclo em nfase bins sem sobreposição.

    suavizado:
        Implementa duas binagens deslocadas seguindo a lógica
        do retrospective binning:
        
        - cada bin possui largura = 2/nfase
        - a segunda binagem é deslocada de 1/nfase
        - cada conjunto possui nfase/2 bins
        
        Exemplo:
        nfase = 6

        Binagem 1:
            [0 → 2/6]
            [2/6 → 4/6]
            [4/6 → 6/6]

        Binagem 2 (deslocada):
            [1/6 → 3/6]
            [3/6 → 5/6]
            [5/6 → 1/6]
    """

    intervalos = []

    if tipo == "continuo":

        passo = 1.0 / nfase

        for i in range(nfase):
            inicio = i * passo
            fim = (i + 1) * passo
            intervalos.append((inicio % 1, fim % 1))

    elif tipo == "suavizado":

        if nfase % 2 != 0:
            raise ValueError(
                "Para agrupamento suavizado, nfase deve ser par."
            )

        deslocamento = 1.0 / nfase
        largura = 2.0 / nfase

        n_bins_base = nfase // 2

        # -----------------------------
        # Primeira binagem
        # -----------------------------
        for i in range(n_bins_base):

            inicio = i * largura
            fim = inicio + largura

            intervalos.append((inicio % 1, fim % 1))

        # -----------------------------
        # Segunda binagem deslocada
        # -----------------------------
        for i in range(n_bins_base):

            inicio = i * largura + deslocamento
            fim = inicio + largura

            intervalos.append((inicio % 1, fim % 1))

    else:
        raise ValueError("tipo deve ser 'continuo' ou 'suavizado'")

    return intervalos


def pertence_intervalo(fase, inicio, fim):
    """
    Verifica se fase pertence ao intervalo circular.
    """

    if inicio < fim:
        return (fase >= inicio) & (fase < fim)

    else:
        return (fase >= inicio) | (fase < fim)


def distancia_circular(fase, centro):
    """
    Distância circular entre fases no domínio [0,1).
    """

    d = np.abs(fase - centro)

    return np.minimum(d, 1 - d)


def pesos_triangulares(fases, centro, largura):
    """
    Peso máximo no centro e decaimento linear até zero
    nas bordas do intervalo.
    """

    dist = distancia_circular(fases, centro)

    pesos = 1 - (dist / largura)

    pesos[pesos < 0] = 0

    return pesos


def media_ponderada(volumes, pesos):
    """
    Média ponderada no eixo temporal.
    """

    pesos = np.asarray(pesos, dtype=np.float32)

    soma_pesos = np.sum(pesos)

    if soma_pesos == 0:
        return np.zeros(volumes.shape[:3], dtype=np.float32)

    pesos = pesos / soma_pesos

    return np.sum(volumes * pesos[None, None, None, :], axis=-1)


def mapa_por_fase(
    dire,
    n_fases,
    tipo="continuo",
    metodo="media"
):
    """
    Parameters
    ----------
    tipo:
        'continuo' ou 'suavizado'

    metodo:
        'media'      -> média simples no intervalo
        'ponderado'  -> média ponderada pela distância da fase central
    """

    path_dynDWI = os.path.join(dire, "dwi_corrected.nii.gz")
    path_bvals = os.path.join(dire, "bvals.bval")
    path_df = os.path.join(dire, "Analysis/physiological_marked.csv")
    path_b0 = os.path.join(dire, "b0_final.nii.gz")

    if not os.path.exists(path_bvals):
        raise FileNotFoundError(f"Arquivo bvals não encontrado: {path_bvals}")

    bvals = np.loadtxt(path_bvals)

    df = pd.read_csv(path_df)

    dwi_img = nib.load(path_dynDWI)

    dynDWI = dwi_img.get_fdata()

    affine = dwi_img.affine
    header = dwi_img.header

    b0 = nib.load(path_b0).get_fdata()

    # apenas DWI
    dwi, indices_dwi = selecionar_volumes_dwi(dynDWI, bvals)

    # alinhar dataframe
    df_dif = df[df["difusao"] == 1].reset_index(drop=True)

    if len(df_dif) != dwi.shape[-1]:
        raise ValueError(
            "Número de volumes DWI não bate com df['difusao']==1"
        )

    fases = df_dif["fase_flag"].values

    intervalos = tipo_agrupamento_fase(n_fases, tipo)

    mapas_fase = []

    for inicio, fim in intervalos:

        mask = pertence_intervalo(fases, inicio, fim)

        if np.sum(mask) == 0:

            print(
                f"Aviso: fase vazia no intervalo "
                f"({inicio:.2f}, {fim:.2f})"
            )

            mapa_medio = np.zeros(dwi.shape[:3], dtype=np.float32)

        else:

            volumes = dwi[..., mask]

            fases_intervalo = fases[mask]

            # ==========================
            # MÉDIA SIMPLES
            # ==========================

            if metodo == "media":

                mapa_medio = np.nanmean(volumes, axis=-1)

            # ==========================
            # MÉDIA PONDERADA
            # ==========================

            elif metodo == "ponderado":

                # centro circular do intervalo

                largura = ((fim - inicio) % 1) / 2

                centro = (inicio + largura) % 1

                pesos = pesos_triangulares(
                    fases_intervalo,
                    centro=centro,
                    largura=largura
                )

                mapa_medio = media_ponderada(
                    volumes,
                    pesos
                )

            else:
                raise ValueError(
                    "metodo deve ser 'media' ou 'ponderado'"
                )

        # normalização pelo b0
        mapas_fase.append(mapa_medio)

    mapas_medio_fases = np.stack(mapas_fase, axis=-1)

    return mapas_medio_fases, affine, header


import os
import numpy as np
import nibabel as nib


import os
import numpy as np
import nibabel as nib


import os
import numpy as np
import nibabel as nib


def organizacao_mapa_medio(
    sub_dir,
    nfases,
    tipo="continuo",
    metodo="media"
):

    directions = [
        "dynDWI_S",
        "dynDWI_M",
        "dynDWI_P",
        "dynDWI_MS",
        "dynDWI_MP",
        "dynDWI_PS"
    ]
    directions = [
        "dynDWI_S",

    ]
    # ==========================================================
    # LOOP NAS DIREÇÕES
    # ==========================================================

    for dire in directions:

        path_dir = os.path.join(sub_dir, dire)

        # ------------------------------------------------------
        # verificar existência
        # ------------------------------------------------------

        if not os.path.exists(path_dir):

            print(f"[IGNORADO] {dire} não encontrado")

            continue

        print(f"\n[PROCESSANDO] {dire}")

        # ======================================================
        # CAMINHOS
        # ======================================================

        path_dwi = os.path.join(
            path_dir,
            "dwi_corrected.nii.gz"
        )

        path_bvals = os.path.join(
            path_dir,
            "bvals.bval"
        )

        path_b0 = os.path.join(
            path_dir,
            "b0_final.nii.gz"
        )

        path_mask = os.path.join(
            path_dir,
            "b0_brain_mask.nii.gz"
        )

        # ======================================================
        # CARREGAR B0
        # ======================================================

        b0_img = nib.load(path_b0)

        b0 = b0_img.get_fdata()

        affine = b0_img.affine
        header = b0_img.header

        # ======================================================
        # MÁSCARA
        # ======================================================

        if os.path.exists(path_mask):

            mask = nib.load(path_mask).get_fdata()

        else:

            mask = None

        # ======================================================
        # CRIAR DIRETÓRIO Analysis
        # ======================================================

        analysis_dir = os.path.join(
            path_dir,
            "Analysis"
        )

        os.makedirs(
            analysis_dir,
            exist_ok=True
        )

        # ======================================================
        # DWI BASE
        # ======================================================

        path_base = os.path.join(
            analysis_dir,
            "dwi_base.nii.gz"
        )

        if not os.path.exists(path_base):

            print("  criando dwi_base.nii.gz")

            dyn_img = nib.load(path_dwi)

            dyn_data = dyn_img.get_fdata()

            bvals = np.loadtxt(path_bvals)

            dwi, _ = selecionar_volumes_dwi(
                dyn_data,
                bvals
            )

            dwi_base = np.nanmean(
                dwi,
                axis=-1
            )

            if mask is not None:

                dwi_base *= mask

            new_header = dyn_img.header.copy()

            new_header.set_data_shape(
                dwi_base.shape
            )

            nib.save(
                nib.Nifti1Image(
                    dwi_base.astype(np.float32),
                    affine=affine,
                    header=new_header
                ),
                path_base
            )

        # ======================================================
        # MAPA POR FASE
        # ======================================================

        print("  calculando mapas por fase")

        mapas, aff, hdr = mapa_por_fase(
            path_dir,
            nfases,
            tipo=tipo,
            metodo=metodo
        )

        # ======================================================
        # SALVAR MAPA 4D ORGANIZADO
        # ======================================================

        out_dir = os.path.join(
            analysis_dir,
            "ordenamento"
        )

        os.makedirs(
            out_dir,
            exist_ok=True
        )

        out_name = (
            f"dwi_pulse_"
            f"{nfases}_"
            f"{tipo}_"
            f"{metodo}.nii.gz"
        )

        out_path = os.path.join(
            out_dir,
            out_name
        )

        print("  salvando mapa organizado")

        new_header = header.copy()

        new_header.set_data_shape(
            mapas.shape
        )

        nib.save(
            nib.Nifti1Image(
                mapas.astype(np.float32),
                affine=affine,
                header=new_header
            ),
            out_path
        )

    print("\nProcessamento concluído.")

organizacao_mapa_medio(
    sub_dir ='/media/joao-oliveira/PortableSSD/dynDWI_V2/sub028_2',
    nfases = 6,
    tipo="continuo",
    metodo="media")
organizacao_mapa_medio(
    sub_dir ='/media/joao-oliveira/PortableSSD/dynDWI_V2/sub028_2',
    nfases = 6,
    tipo="continuo",
    metodo="ponderado")
organizacao_mapa_medio(
    sub_dir ='/media/joao-oliveira/PortableSSD/dynDWI_V2/sub028_2',
    nfases = 6,
    tipo="suavizado",
    metodo="media")
organizacao_mapa_medio(
    sub_dir ='/media/joao-oliveira/PortableSSD/dynDWI_V2/sub028_2',
    nfases = 6,
    tipo="suavizado",
    metodo="ponderado")