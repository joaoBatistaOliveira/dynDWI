import os
import json
import numpy as np
import pandas as pd
import physiological_marked
# import dti_fsl
import mapas
import cross_correlation
import subject_database

def plotar_cc(dir_base = dir_base, subjects = subjects, directions = directions,  rois = rois, tipo = "p90"r):
    dic = cross_correlation.plot_cc_global_subjects(dir_base = dir_base, subjects = subjects, directions = directions,  rois = rois, signal = 'resp', show_image = False, tipo = tipo)
    dic = cross_correlation.plot_cc_global_subjects(dir_base = dir_base, subjects = subjects,  directions = directions,  rois = rois, signal = 'ppu',  show_image = False, tipo = tipo )

path_db = "/media/joao-oliveira/PortableSSD/dynDWI_V2/subjects.json"
base_dir = "/media/joao-oliveira/PortableSSD/dynDWI_V2"
rois = ['PC', 'V4', 'V3', 'VL', 'WM', 'GM', 'CSF', 'M1', 'V1', 'MT', 'FRONTAL']
directions_analise=["S", "M", "P", "MS","MP", "PS"]
directions_analise=["MS","MP", "PS"]

#############################################################################
#paradigma 30/30 saudável
for direction in directions_analise:
    subjects, directions = db.get_subjects(protocol=30, direction=direction,grupo="saudavel")
    
    #    remover = {"hpn001"}
    #    subjects_filtrado = []
    #    directions_filtrado = []
    #    for sub, dire in zip(subjects, directions):
    #        if sub not in remover:
    #            subjects_filtrado.append(sub)
    #            directions_filtrado.append(dire)
    #    subjects = subjects_filtrado
    #    directions = directions_filtrado

    for sub, dire in zip(subjects, directions):
        sub_dir = os.path.join(base_dir, sub)
        dire_base = os.path.join(sub_dir, dire)
        if not os.path.isdir(dire_base):
            print(f"  {dire} não encontrado. Pulando.")
            continue

        meta = physiological_marked.read_json(os.path.join(dire_base, "dynDWI.json"))

        df = physiological_marked.fisio_add_flags(
            phys_path=os.path.join(dire_base, "ScanPsaLog.log"),
            aquisition_duration=meta["AcquisitionDuration"],
            repetition_time=meta["RepetitionTime"],
            paradigm=[30, 30],
            derivadas_ppu=True,
            salvar_df="physiological_marked",
            salvar_imagem=f"physiological_marked_{dire}",
            mostrar_imagem=False,
        )
        adc = mapas.processar_mapa_adc_b0_mean( dire_base, return_adc=True)

        for roi in rois:    
            resultados = cross_correlation.processar_cc_completo(
                path=dire_base,
                name=roi,
                intervalo=5,
                step=0.1,
                save_plot=True,
                cutoff_ppu=5,
                cutoff_resp=1, 
                n_controle=500)
    plotar_cc(dir_base = base_dir, subjects = subjects, directions = directions,  rois = rois, tipo = "p30")

###########################################################################
#paradigma 90/90 saudaveis
for direction in directions_analise:
    subjects, directions = db.get_subjects(protocol=90, direction=direction,grupo="saudavel")
    
    #    remover = {"hpn001"}
    #    subjects_filtrado = []
    #    directions_filtrado = []
    #    for sub, dire in zip(subjects, directions):
    #        if sub not in remover:
    #            subjects_filtrado.append(sub)
    #            directions_filtrado.append(dire)
    #    subjects = subjects_filtrado
    #    directions = directions_filtrado

    for sub, dire in zip(subjects, directions):
        sub_dir = os.path.join(base_dir, sub)
        dire_base = os.path.join(sub_dir, dire)
        if not os.path.isdir(dire_base):
            print(f"  {dire} não encontrado. Pulando.")
            continue

        meta = physiological_marked.read_json(os.path.join(dire_base, "dynDWI.json"))

        df = physiological_marked.fisio_add_flags(
            phys_path=os.path.join(dire_base, "ScanPsaLog.log"),
            aquisition_duration=meta["AcquisitionDuration"],
            repetition_time=meta["RepetitionTime"],
            paradigm=[90,90],
            derivadas_ppu=True,
            salvar_df="physiological_marked",
            salvar_imagem=f"physiological_marked_{dire}",
            mostrar_imagem=False,
        )
        adc = mapas.processar_mapa_adc_b0_mean(dire_base, return_adc=True)

        for roi in rois:    
            resultados = cross_correlation.processar_cc_completo(
                path=dire_base,
                name=roi,
                intervalo=5,
                step=0.1,
                save_plot=True,
                cutoff_ppu=5,
                cutoff_resp=1, 
                n_controle=500)
    plotar_cc(dir_base = base_dir, subjects = subjects, directions = directions,  rois = rois, tipo = "p90")
############################################################################
#hpn 90/90

for direction in directions_analise:
    subjects, directions = db.get_subjects(protocol=90, direction=direction,grupo="hidrocefalia")
    
    remover = {"hpn001"}
    subjects_filtrado = []
    directions_filtrado = []
    for sub, dire in zip(subjects, directions):
        if sub not in remover:
            subjects_filtrado.append(sub)
            directions_filtrado.append(dire)
    subjects = subjects_filtrado
    directions = directions_filtrado

    for sub, dire in zip(subjects, directions):
        sub_dir = os.path.join(base_dir, sub)
        dire_base = os.path.join(sub_dir, dire)
        if not os.path.isdir(dire_base):
            print(f"  {dire} não encontrado. Pulando.")
            continue

        meta = physiological_marked.read_json(os.path.join(dire_base, "dynDWI.json"))

        df = physiological_marked.fisio_add_flags(
            phys_path=os.path.join(dire_base, "ScanPsaLog.log"),
            aquisition_duration=meta["AcquisitionDuration"],
            repetition_time=meta["RepetitionTime"],
            paradigm=[90,90],
            derivadas_ppu=True,
            salvar_df="physiological_marked",
            salvar_imagem=f"physiological_marked_{dire}",
            mostrar_imagem=False,
        )
        adc = mapas.processar_mapa_adc_b0_mean( dire_base, return_adc=True)

        for roi in rois:    
            resultados = cross_correlation.processar_cc_completo(
                path=dire_base,
                name=roi,
                intervalo=5,
                step=0.1,
                save_plot=True,
                cutoff_ppu=5,
                cutoff_resp=1, 
                n_controle=500)
    plotar_cc(dir_base = base_dir, subjects = subjects, directions = directions,  rois = rois, tipo = "p90")

############################################################################
#epi 0
for direction in directions_analise:
    subjects, directions = db.get_subjects(protocol=0, direction=direction,grupo="epilepsia")
    
    remover = {"epi002"}
    subjects_filtrado = []
    directions_filtrado = []
    for sub, dire in zip(subjects, directions):
        if sub not in remover:
            subjects_filtrado.append(sub)
            directions_filtrado.append(dire)
    subjects = subjects_filtrado
    directions = directions_filtrado

    for sub, dire in zip(subjects, directions):
        sub_dir = os.path.join(base_dir, sub)
        dire_base = os.path.join(sub_dir, dire)
        if not os.path.isdir(dire_base):
            print(f"  {dire} não encontrado. Pulando.")
            continue

        meta = physiological_marked.read_json(os.path.join(dire_base, "dynDWI.json"))

        df = physiological_marked.fisio_add_flags(
            phys_path=os.path.join(dire_base, "ScanPsaLog.log"),
            aquisition_duration=meta["AcquisitionDuration"],
            repetition_time=meta["RepetitionTime"],
            paradigm=[meta['AcquisitionDuration'], 0],
            derivadas_ppu=True,
            salvar_df="physiological_marked",
            salvar_imagem=f"physiological_marked_{dire}",
            mostrar_imagem=False,
        )
        adc = mapas.processar_mapa_adc_b0_mean(dire_base,return_adc=True)
        for roi in rois:    
            resultados = cross_correlation.processar_cc_completo(
                path=dire_base,
                name=roi,
                intervalo=5,
                step=0.1,
                save_plot=True,
                cutoff_ppu=5,
                cutoff_resp=1, 
                n_controle=500)
    plotar_cc(dir_base = base_dir, subjects = subjects, directions = directions,  rois = rois, tipo = "p00")


