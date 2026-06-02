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
    """

    intervalos = []

    if tipo == "continuo":
        passo = 1.0 / nfase
        for i in range(nfase):
            inicio = i * passo
            fim = (i + 1) * passo
            intervalos.append((inicio, fim))

    elif tipo == "suavizado":
        passo = 1.0 / nfase
        largura = 2 * passo  # sobreposição

        for i in range(nfase):
            centro = i * passo
            inicio = centro - passo / 2
            fim = centro + 3 * passo / 2
            intervalos.append((inicio % 1, fim % 1))

    else:
        raise ValueError("tipo deve ser 'continuo' ou 'suavizado'")

    return intervalos


def pertence_intervalo(fase, inicio, fim):
    """
    Verifica se uma fase pertence ao intervalo circular [inicio, fim]
    """
    if inicio < fim:
        return (fase >= inicio) & (fase < fim)
    else:
        # caso circular (ex: 0.75 → 0.25)
        return (fase >= inicio) | (fase < fim)


def mapa_por_fase(dire, n_fases, tipo):
    path_dynDWI = os.path.join(dire, "dwi_corrected.nii.gz")
    path_bvals = os.path.join(dire, "bvals.bval")
    path_df = os.path.join(dire, "Analysis/physiological_marked.csv")
    path_b0 = os.path.join(dire, "b0_final.nii.gz")

    # ---- carregamento ----
    if not os.path.exists(path_bvals):
        raise FileNotFoundError(f"Arquivo bvals não encontrado: {path_bvals}")

    bvals = np.loadtxt(path_bvals)
    df = pd.read_csv(path_df)

    dwi_img = nib.load(path_dynDWI)
    dynDWI = dwi_img.get_fdata()
    affine = dwi_img.affine
    header = dwi_img.header

    b0 = nib.load(path_b0).get_fdata()

    # ---- selecionar apenas volumes de difusão ----
    dwi, indices_dwi = selecionar_volumes_dwi(dynDWI, bvals)

    # ---- alinhar com dataframe ----
    df_dif = df[df['difusao'] == 1].reset_index(drop=True)

    if len(df_dif) != dwi.shape[-1]:
        raise ValueError("Número de volumes DWI não bate com df['difusao']==1")

    fases = df_dif['fase_flag'].values  # valores entre 0 e 1

    # ---- obter intervalos de fase ----
    intervalos = tipo_agrupamento_fase(n_fases, tipo=tipo)

    mapas_fase = []

    for inicio, fim in intervalos:
        mask = pertence_intervalo(fases, inicio, fim)

        if np.sum(mask) == 0:
            print(f"Aviso: fase vazia no intervalo ({inicio:.2f}, {fim:.2f})")
            mapa_medio = np.zeros(dwi.shape[:3])
        else:
            volumes = dwi[..., mask]
            mapa_medio = np.mean(volumes, axis=-1)

        # divisão pelo b0 (evitar divisão por zero)
        mapa_norm = np.divide(
            mapa_medio,
            b0,
            out=np.zeros_like(mapa_medio),
            where=b0 > 0
        )

        mapas_fase.append(mapa_norm)

    # ---- concatenar no eixo temporal ----
    mapas_medio_fases = np.stack(mapas_fase, axis=-1)

    return mapas_medio_fases, affine, header



def organizacao_processamento_dtifit(sub_dir, nfases, tipo="continuo"):

    directions = ["dynDWI_S", "dynDWI_M", "dynDWI_P",
                  "dynDWI_MS", "dynDWI_MP", "dynDWI_PS"]

    # ---- carregar b0 e máscara ----
    path_b0 = os.path.join(sub_dir, "b0_final.nii.gz")
    b0_img = nib.load(path_b0)
    b0_ref = b0_img.get_fdata()
    affine = b0_img.affine
    header = b0_img.header

    path_mask = os.path.join(sub_dir, "b0_brain_mask.nii.gz")
    b0_brain_mask = nib.load(path_mask).get_fdata()

    # ---- rodar mapa_por_fase para cada direção ----
    mapas_direcoes = {}

    for dire in directions:
        path_dir = os.path.join(sub_dir, dire)

        mapas, aff, hdr = mapa_por_fase(path_dir, nfases, tipo=tipo)
        mapas_direcoes[dire] = mapas  # shape: (X,Y,Z,nfases)

    # ---- criar diretório base ----
    base_out = os.path.join(sub_dir, f"DTI/DTI_f{nfases}_{tipo}")
    os.makedirs(base_out, exist_ok=True)

    # ---- carregar bvals/bvecs ----
    path_bvals = os.path.join(sub_dir, "DTI/bvals.bval")
    path_bvecs = os.path.join(sub_dir, "DTI/bvecs.bvec")

    bvals = np.loadtxt(path_bvals)
    bvecs = np.loadtxt(path_bvecs)

    # ---- loop por fase ----
    for i in range(nfases):

        fase_dir = os.path.join(base_out, f"dti_{i}")
        os.makedirs(fase_dir, exist_ok=True)

        volumes = []

        # ---- b0 como primeiro volume ----
        b0_4d = np.expand_dims(b0_ref, axis=-1)
        volumes.append(b0_4d)

        # ---- adicionar cada direção ----
        for dire in directions:
            mapa = mapas_direcoes[dire][..., i]
            mapa_4d = np.expand_dims(mapa, axis=-1)
            volumes.append(mapa_4d)

        # ---- concatenar ----
        dti_data = np.concatenate(volumes, axis=-1)

        # ---- salvar nifti ----
        new_header = header.copy()
        new_header.set_data_shape(dti_data.shape)

        dti_path = os.path.join(fase_dir, f"dti_{i}.nii.gz")

        nib.save(
            nib.Nifti1Image(dti_data, affine=affine, header=new_header),
            dti_path
        )

        # # ---- salvar máscara (necessário para FSL) ----
        mask_path = os.path.join(fase_dir, "mask.nii.gz")
        # nib.save(
        #     nib.Nifti1Image(b0_brain_mask, affine=affine, header=header),
        #     mask_path
        # )

        # ---- comando dtifit ----
        cmd = [
            "dtifit",
            "-k", dti_path,
            "-o", os.path.join(fase_dir, "dti"),
            "-m", mask_path,
            "-r", path_bvecs,
            "-b", path_bvals
        ]

        print(f"Rodando dtifit para fase {i}...")
        subprocess.run(cmd, check=True)


def mapas_principais_dti(DTI_path):
    a função deverá recebr o caminho do diretorio base do dti a ser analisado, por exemplo: sub027/DTI/DTI_f6_continuo"
    desntro desse diretorio, ela deverá navegra por cada sub diretorio de fase dti_0, dti_1... para constryuir os arquivos de interesse:
    dti_FA.nii.gz, concatenando os arquivos dti_j/dti_FA.nii.gz
    dti_MD.nii.gz, concatenando os arquivos dti_j/dti_MD.nii.gz
    dti_V1.nii.gz, concatenando os arquivos dti_j/dti_V1.nii.gz
    O mesmo para dti_V2.nii.gz e dti_V3.nii.gzsalve esses mapas concatenados no sub027/DTI/DTI_f6_continuo