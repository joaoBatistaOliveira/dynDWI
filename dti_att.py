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
            nfase +=1

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

    if not os.path.exists(path_bvals):
        raise FileNotFoundError(f"Arquivo bvals não encontrado: {path_bvals}")

    bvals = np.loadtxt(path_bvals)
    df = pd.read_csv(path_df)
    dwi_img = nib.load(path_dynDWI)
    dynDWI = dwi_img.get_fdata()
    affine = dwi_img.affine
    header = dwi_img.header

    b0 = nib.load(path_b0).get_fdata()

    #seleciona apenas volumes de difusão
    dwi, indices_dwi = selecionar_volumes_dwi(dynDWI, bvals)

    #alinhar com dataframe
    df_dif = df[df['difusao'] == 1].reset_index(drop=True)

    if len(df_dif) != dwi.shape[-1]:
        raise ValueError("Número de volumes DWI não bate com df['difusao']==1")

    fases = df_dif['fase_flag'].values  # valores entre 0 e 1

    #obter intervalos de fase
    intervalos = tipo_agrupamento_fase(n_fases, tipo=tipo)

    mapas_fase = []

    for inicio, fim in intervalos:
        mask = pertence_intervalo(fases, inicio, fim)

        if np.sum(mask) == 0:
            print(f"Aviso: fase vazia no intervalo ({inicio:.2f}, {fim:.2f})")
            mapa_medio = np.zeros(dwi.shape[:3])
        else:
            volumes = dwi[..., mask]
            mapa_medio = np.nanmean(volumes, axis=-1)

        #Normalização pelo respectivo b0
        mapa_norm = np.divide(
            mapa_medio,
            b0,
            out=np.zeros_like(mapa_medio),
            where=b0 > 0
        )

        mapas_fase.append(mapa_norm)

    #concatenar no eixo temporal (volumes ordenados pela fase)
    mapas_medio_fases = np.stack(mapas_fase, axis=-1)

    return mapas_medio_fases, affine, header


def organizacao_mapa_medio(sub_dir, nfases, tipo="continuo"):

    directions = ["dynDWI_S", "dynDWI_M", "dynDWI_P", "dynDWI_MS", "dynDWI_MP", "dynDWI_PS"]

    path_b0 = os.path.join(sub_dir, "b0_final.nii.gz")
    b0_img = nib.load(path_b0)
    b0_ref = b0_img.get_fdata()
    affine = b0_img.affine
    header = b0_img.header
    b0_ref = np.divide(
            b0_ref,
            b0_ref,
            out=np.zeros_like(b0_ref),
            where=b0_ref > 0
        )
    path_mask = os.path.join(sub_dir, "b0_brain_mask.nii.gz")
    b0_brain_mask = nib.load(path_mask).get_fdata()

    #rodar mapa_por_fase para cada direção
    mapas_direcoes = {}
    mapas_baseline = {}

    for dire in directions:
        path_dir = os.path.join(sub_dir, dire)

        mapas, aff, hdr = mapa_por_fase(path_dir, nfases, tipo=tipo)
        mapas_direcoes[dire] = mapas  # shape: (X,Y,Z,nfases)
        
        #mapa de referencia pra avaliar oscilações
        mapas_base, aff_base, hdr_base = mapa_por_fase(path_dir, 1, tipo="continuo")
        mapas_baseline[dire] = mapas_base  # shape: (X,Y,Z)

    #criar diretório base
    base_out = os.path.join(sub_dir, f"DTI/DTI_f{nfases}_{tipo}")
    os.makedirs(base_out, exist_ok=True)

    path_bvals = os.path.join(sub_dir, "DTI/bvals.bval")
    path_bvecs = os.path.join(sub_dir, "DTI/bvecs.bvec")
    bvals = np.loadtxt(path_bvals)
    bvecs = np.loadtxt(path_bvecs)

    #loop por fase
    for i in range(nfases):

        fase_dir = os.path.join(base_out, f"dti_{i}")
        os.makedirs(fase_dir, exist_ok=True)

        volumes = []

        #b0 primeiro volume -> unitário, pois já normalizamos anteriormente
        #obrigatorio para o dtifit
        b0_4d = np.expand_dims(b0_ref, axis=-1)
        volumes.append(b0_4d)

        #adicionar cada direção
        for dire in directions:
            mapa = mapas_direcoes[dire][..., i]
            mapa_4d = np.expand_dims(mapa, axis=-1)
            volumes.append(mapa_4d)
    
        dti_data = np.concatenate(volumes, axis=-1)

        new_header = header.copy()
        new_header.set_data_shape(dti_data.shape)

        dti_path = os.path.join(fase_dir, f"dti_{i}.nii.gz")
        nib.save(nib.Nifti1Image(dti_data, affine=affine, header=new_header),dti_path)


        #dtifit - FSL
        cmd = [
            "dtifit",
            "-k", dti_path,
            "-o", os.path.join(fase_dir, "dti"),
            "-m", path_mask,
            "-r", path_bvecs,
            "-b", path_bvals,
            "--save_tensor"
        ]

        print(f"Rodando dtifit para fase {i}...")
        subprocess.run(cmd, check=True)

    #baseline
    base_dir = os.path.join(base_out, f"dti_base")
    os.makedirs(base_dir, exist_ok=True)
    volumes_base = []

    b0_4d_b = np.expand_dims(b0_ref, axis=-1)
    volumes_base.append(b0_4d_b)
    for dire in directions:
        mapa = mapas_baseline[dire][..., 0]
        mapa_4d = np.expand_dims(mapa, axis=-1)
        volumes_base.append(mapa_4d)

    dti_data_b = np.concatenate(volumes_base, axis=-1)

    new_header = header.copy()
    new_header.set_data_shape(dti_data_b.shape)

    dti_path_b = os.path.join(base_dir, f"dti.nii.gz")
    nib.save(nib.Nifti1Image(dti_data_b, affine=affine, header=new_header),dti_path_b)

    cmd = [
        "dtifit",
        "-k", dti_path_b,
        "-o", os.path.join(base_dir, "dti"),
        "-m", path_mask,
        "-r", path_bvecs,
        "-b", path_bvals,
        "--save_tensor"
    ]
    print("Rodando dtifit para baseline...")
    subprocess.run(cmd, check=True)



def organizacao_processamento_dtifit(sub_dir, nfases, tipo="continuo"):

    directions = ["dynDWI_S", "dynDWI_M", "dynDWI_P", "dynDWI_MS", "dynDWI_MP", "dynDWI_PS"]

    path_b0 = os.path.join(sub_dir, "b0_final.nii.gz")
    b0_img = nib.load(path_b0)
    b0_ref = b0_img.get_fdata()
    affine = b0_img.affine
    header = b0_img.header
    b0_ref = np.divide(
            b0_ref,
            b0_ref,
            out=np.zeros_like(b0_ref),
            where=b0_ref > 0
        )
    path_mask = os.path.join(sub_dir, "b0_brain_mask.nii.gz")
    b0_brain_mask = nib.load(path_mask).get_fdata()

    #rodar mapa_por_fase para cada direção
    mapas_direcoes = {}
    mapas_baseline = {}

    for dire in directions:
        path_dir = os.path.join(sub_dir, dire)

        mapas, aff, hdr = mapa_por_fase(path_dir, nfases, tipo=tipo)
        mapas_direcoes[dire] = mapas  # shape: (X,Y,Z,nfases)
        
        #mapa de referencia pra avaliar oscilações
        mapas_base, aff_base, hdr_base = mapa_por_fase(path_dir, 1, tipo="continuo")
        mapas_baseline[dire] = mapas_base  # shape: (X,Y,Z)

    #criar diretório base
    base_out = os.path.join(sub_dir, f"DTI/DTI_f{nfases}_{tipo}")
    os.makedirs(base_out, exist_ok=True)

    path_bvals = os.path.join(sub_dir, "DTI/bvals.bval")
    path_bvecs = os.path.join(sub_dir, "DTI/bvecs.bvec")
    bvals = np.loadtxt(path_bvals)
    bvecs = np.loadtxt(path_bvecs)

    #loop por fase
    for i in range(nfases):

        fase_dir = os.path.join(base_out, f"dti_{i}")
        os.makedirs(fase_dir, exist_ok=True)

        volumes = []

        #b0 primeiro volume -> unitário, pois já normalizamos anteriormente
        #obrigatorio para o dtifit
        b0_4d = np.expand_dims(b0_ref, axis=-1)
        volumes.append(b0_4d)

        #adicionar cada direção
        for dire in directions:
            mapa = mapas_direcoes[dire][..., i]
            mapa_4d = np.expand_dims(mapa, axis=-1)
            volumes.append(mapa_4d)
    
        dti_data = np.concatenate(volumes, axis=-1)

        new_header = header.copy()
        new_header.set_data_shape(dti_data.shape)

        dti_path = os.path.join(fase_dir, f"dti_{i}.nii.gz")
        nib.save(nib.Nifti1Image(dti_data, affine=affine, header=new_header),dti_path)


        #dtifit - FSL
        cmd = [
            "dtifit",
            "-k", dti_path,
            "-o", os.path.join(fase_dir, "dti"),
            "-m", path_mask,
            "-r", path_bvecs,
            "-b", path_bvals,
            "--save_tensor"
        ]

        print(f"Rodando dtifit para fase {i}...")
        subprocess.run(cmd, check=True)

    #baseline
    base_dir = os.path.join(base_out, f"dti_base")
    os.makedirs(base_dir, exist_ok=True)
    volumes_base = []

    b0_4d_b = np.expand_dims(b0_ref, axis=-1)
    volumes_base.append(b0_4d_b)
    for dire in directions:
        mapa = mapas_baseline[dire][..., 0]
        mapa_4d = np.expand_dims(mapa, axis=-1)
        volumes_base.append(mapa_4d)

    dti_data_b = np.concatenate(volumes_base, axis=-1)

    new_header = header.copy()
    new_header.set_data_shape(dti_data_b.shape)

    dti_path_b = os.path.join(base_dir, f"dti.nii.gz")
    nib.save(nib.Nifti1Image(dti_data_b, affine=affine, header=new_header),dti_path_b)

    cmd = [
        "dtifit",
        "-k", dti_path_b,
        "-o", os.path.join(base_dir, "dti"),
        "-m", path_mask,
        "-r", path_bvecs,
        "-b", path_bvals,
        "--save_tensor"
    ]
    print("Rodando dtifit para baseline...")
    subprocess.run(cmd, check=True)


def mapas_principais_dti(DTI_path):

    fases_dirs = [
        d for d in os.listdir(DTI_path)
        if d.startswith("dti_")
    ]
    print(fases_dirs)
    fases_dirs.remove("dti_base")
    # ordenar corretamente: dti_0, dti_1, ..., dti_n
    fases_dirs = sorted(fases_dirs, key=lambda x: int(x.split("_")[1]))

    #estruturas para armazenar
    mapas_FA = []
    mapas_MD = []
    mapas_RD = []
    mapas_L1 = []

    affine = None
    header = None

    #loop nas fases
    for d in fases_dirs:
        fase_path = os.path.join(DTI_path, d)

        path_FA = os.path.join(fase_path, "dti_FA.nii.gz")
        path_MD = os.path.join(fase_path, "dti_MD.nii.gz")
        path_RD = os.path.join(fase_path, "dti_RD.nii.gz")
        path_L1 = os.path.join(fase_path, "dti_L1.nii.gz")
        path_L2 = os.path.join(fase_path, "dti_L2.nii.gz")
        path_L3 = os.path.join(fase_path, "dti_L3.nii.gz")

        FA_img = nib.load(path_FA)
        MD_img = nib.load(path_MD)
        L1_img = nib.load(path_L1)
        L2 = nib.load(path_L2).get_fdata()
        L3 = nib.load(path_L3).get_fdata()

        FA = FA_img.get_fdata()
        MD = MD_img.get_fdata()
        L1 = L1_img.get_fdata()
        
        # salvar affine/header da primeira fase
        if affine is None:
            affine = FA_img.affine
            header = FA_img.header
        RD = (L2+L3)/23
        out_path = os.path.join(fase_path, "dti_RD.nii.gz")
        nib.save(nib.Nifti1Image(RD, affine=L1_img.affine, header=L1_img.header), out_path)
        # empilhar
        mapas_FA.append(FA)
        mapas_MD.append(MD)
        mapas_L1.append(L1)
        mapas_RD.apende(RD)

    #concatenar no eixo de fase
    FA_4D = np.stack(mapas_FA, axis=-1)
    MD_4D = np.stack(mapas_MD, axis=-1)
    L1_4D = np.stack(mapas_L1, axis=-1)
    RD_4D = np.stack(mapas_RD, axis=-1)

    def salvar_nifti(data, nome):
        new_header = header.copy()
        new_header.set_data_shape(data.shape)
        out_path = os.path.join(DTI_path, nome)
        nib.save(
            nib.Nifti1Image(data, affine=affine, header=new_header),
            out_path
        )
        
    salvar_nifti(FA_4D, "dti_FA.nii.gz")
    salvar_nifti(MD_4D, "dti_MD.nii.gz")
    salvar_nifti(L1_4D, "dti_L1.nii.gz")
    salvar_nifti(RD_4D, "dti_RD.nii.gz")

    print("Mapas concatenados salvos com sucesso.")

def diferenca_relativa(dti_i, dti_base):
    diff = dti_i - dti_base
    return np.divide(
        diff,
        dti_base,
        out=np.zeros_like(dti_base),
        where=dti_base > 0
    )

def amplitude_temporal(delta_series):
    return np.max(delta_series, axis=0) - np.min(delta_series, axis=0)

def variancia_temporal(delta_series):
    return np.var(delta_series, axis=0)

def mad_temporal(delta_series):
    med = np.median(delta_series, axis=0)
    return np.median(np.abs(delta_series - med), axis=0)
    
def cos_angular(v, v_base):
    dot = np.sum(v * v_base, axis=-1)
    norm_v = np.linalg.norm(v, axis=-1)
    norm_base = np.linalg.norm(v_base, axis=-1)

    cos_theta = np.divide(
        dot,
        norm_v * norm_base,
        out=np.zeros_like(dot),
        where=(norm_v > 0) & (norm_base > 0)
    )

    return np.abs(cos_theta)

def angulo(cos_theta):
    return np.arccos(np.clip(cos_theta, -1, 1))

def dispersao_angular(theta_series):
    return np.std(theta_series, axis=0)

def coerencia_temporal(cos_series):
    return np.mean(cos_series, axis=0)



def metricas_mapas_dti(dir_files):

    fases_dirs = [
        d for d in os.listdir(DTI_path)
        if d.startswith("dti_")
    ]
    fases_dirs.remove("dti_base")
    # ordenar corretamente: dti_0, dti_1, ..., dti_n
    fases_dirs = sorted(fases_dirs, key=lambda x: int(x.split("_")[1]))

    #estruturas para armazenar
    mapas_FA = []
    mapas_MD = []

    affine = None
    header = None

    #loop nas fases
    for d in fases_dirs:
        fase_path = os.path.join(DTI_path, d)

        path_FA = os.path.join(fase_path, "dti_FA.nii.gz")
        path_MD = os.path.join(fase_path, "dti_MD.nii.gz")

        FA_img = nib.load(path_FA)
        MD_img = nib.load(path_MD)

        FA = FA_img.get_fdata()
        MD = MD_img.get_fdata()

        # salvar affine/header da primeira fase
        if affine is None:
            affine = FA_img.affine
            header = FA_img.header

        # empilhar
        mapas_FA.append(FA)
        mapas_MD.append(MD)

    #concatenar no eixo de fase
    FA_4D = np.stack(mapas_FA, axis=-1)
    MD_4D = np.stack(mapas_MD, axis=-1)
    
    salvar_nifti(FA_4D, "dti_FA.nii.gz")
    salvar_nifti(MD_4D, "dti_MD.nii.gz")

    print("Mapas concatenados salvos com sucesso.")
