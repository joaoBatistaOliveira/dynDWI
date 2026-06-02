import nibabel as nib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import finufft
from scipy.signal import detrend


def get_slice_groups(slice_scan_order="default"):

    if slice_scan_order.lower() == "default":
        groups = [[1,11,21],
                    [3,13,23],
                    [5,15,25],
                    [7,17,27],
                    [9,19,29],
                    [2,12,22],
                    [4,14,24],
                    [6,16,26],
                    [8,18,28],
                    [10,20,30]]

    elif slice_scan_order.lower() == "hf":
        groups = [[10,20,30],
                    [9,19,29],
                    [8,18,28],
                    [7,17,27],
                    [6,16,26],
                    [5,15,25],
                    [4,14,24],
                    [3,13,23],
                    [2,12,22],
                    [1,11,21]]

    groups = [[s - 1 for s in g] for g in groups]

    return groups

def extract_group_signals(dwi, roi_mask, slice_scan_order="default"):
    groups = get_slice_groups(slice_scan_order)
    signals = []

    for slices in groups:
        mask = np.zeros_like(roi_mask, dtype=bool)
        mask[:, :, slices] = roi_mask[:, :, slices]
        voxels = dwi[mask]
        voxels = voxels.reshape(-1,dwi.shape[-1])
        mean_signal = np.nanmean(voxels,axis=0)
        signals.append(mean_signal)
    return np.array(signals)

def build_group_times(tempo_difusao, TR, n_groups):
    dt = TR / n_groups
    times = []

    for g in range(n_groups):
        times.append(tempo_difusao + g * dt)

    return np.array(times)

def plot_slice_group_signals(dwi, roi_mask, tempo_difusao, TR, slice_scan_order="default"):

    signals = extract_group_signals(dwi,roi_mask,slice_scan_order)
    groups = get_slice_groups(slice_scan_order)
    times = build_group_times(tempo_difusao,TR,len(groups))

    plt.figure(figsize=(15,8))

    for g, slices in enumerate(groups):

        label = (
            f"{slices[0]+1}-"
            f"{slices[1]+1}-"
            f"{slices[2]+1}"
        )

        plt.plot(times[g],signals[g],".-",alpha=0.8,label=label)

    plt.xlabel("Tempo (s)")
    plt.ylabel("Sinal médio")
    plt.title( f"Curvas por grupo MB ({slice_scan_order})")
    plt.grid(True)
    plt.legend(ncol=2,fontsize=8)
    plt.tight_layout()
    plt.show()

def build_hypersampled_curve(dwi, roi_mask,tempo_difusao,TR,slice_scan_order="default"):

    signals = extract_group_signals(dwi,roi_mask,slice_scan_order)
    groups = get_slice_groups(slice_scan_order)
    times = build_group_times(tempo_difusao,TR,len(groups))

    t_hyper = []
    s_hyper = []

    n_groups = signals.shape[0]
    n_volumes = signals.shape[1]

    for vol in range(n_volumes):
        for g in range(n_groups):
            t_hyper.append(times[g, vol])
            s_hyper.append(signals[g, vol])
    t_hyper = np.array(t_hyper)
    s_hyper = np.array(s_hyper)
    valid = np.isfinite(s_hyper)
    #return t_hyper[valid], s_hyper[valid]
    return (np.array(t_hyper[valid]),np.array(s_hyper[valid]))

def plot_hypersampled_curve(dwi,roi_mask, tempo_difusao, TR,slice_scan_order="default",zscore=True):

    t_hyper, s_hyper = build_hypersampled_curve(dwi,roi_mask,tempo_difusao, TR,slice_scan_order)

    if zscore:
        s_hyper = (s_hyper - np.mean(s_hyper)) / np.std(s_hyper)

    plt.figure(figsize=(15,6))
    plt.plot(t_hyper,s_hyper, ".-", markersize=4)
    plt.xlabel("Tempo (s)")
    plt.ylabel("Sinal")
    plt.title( f"Curva hiperamostrada ({slice_scan_order})")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    return t_hyper, s_hyper

sub = "sub027"
dire = "S"
roi = "GM"

dwi_path = f"/media/joao-oliveira/PortableSSD/dynDWI_V2/{sub}/dynDWI_{dire}/Analysis/adc.nii.gz"
gm_path = f"/media/joao-oliveira/PortableSSD/dynDWI_V2/{sub}/dynDWI_{dire}/roi/{roi}.nii.gz"
dwi = nib.load(dwi_path).get_fdata()
gm = nib.load(gm_path).get_fdata()
df = pd.read_csv(f'/media/joao-oliveira/PortableSSD/dynDWI_V2/{sub}/dynDWI_{dire}/Analysis/physiological_marked.csv')
tempo_difusao = df[df['b150']==1]['Time']
#dwi = nib.load(dwi_path).get_fdata()[..., :60]

dwi = nib.load(dwi_path).get_fdata()
gm = nib.load(gm_path).get_fdata()

gm_mask = gm > 0

tempo_difusao = (df[df["b150"] == 1]["Time"].values[:60])

plot_slice_group_signals(dwi,gm_mask, tempo_difusao,TR=0.99,slice_scan_order="hf")

t_hyper, s_hyper = plot_hypersampled_curve(dwi, gm_mask,tempo_difusao,TR=0.99, slice_scan_order="hf")



y = detrend(s_hyper)
t0 = t_hyper.min()
T = t_hyper.max() - t_hyper.min()

x = (t_hyper - t0) / T
x = 2*np.pi*x - np.pi

n_modes = 512

spec_signal = finufft.nufft1d1(x, y.astype(np.complex128), n_modes)

power_signal = np.abs(spec_signal)**2

# NUFFT da amostragem
sampling = np.ones_like(t_hyper)
spec_sampling = finufft.nufft1d1( x, sampling.astype(np.complex128), n_modes)

power_sampling = np.abs(spec_sampling)**2

k = np.arange(-n_modes//2, n_modes//2)

freqs = k / T
power_signal = np.fft.fftshift(power_signal)
power_sampling = np.fft.fftshift(power_sampling)

idx = freqs >= 0
freqs = freqs[idx]
power_signal = power_signal[idx]
power_sampling = power_sampling[idx]

power_signal /= power_signal.max()
power_sampling /= power_sampling.max()

plt.figure(figsize=(12,6))

plt.plot(freqs,power_signal,label="Sinal hiperamostrado")
plt.plot(freqs,power_sampling,label="Operador de amostragem")
plt.xlim(0, 1.3)
plt.xlabel("Frequência (Hz)")
plt.ylabel("Potência normalizada")
plt.title("NUFFT: sinal vs operador de amostragem")
plt.grid(True)
plt.legend()
plt.show()


plt.figure(figsize=(12,6))
plt.semilogy(freqs, power_signal,label="Sinal")
plt.semilogy(freqs,power_sampling, label="Amostragem")
plt.xlim(0, 1.3)
plt.xlabel("Frequência (Hz)")
plt.ylabel("Potência normalizada")
plt.title( "NUFFT: sinal vs operador de amostragem")
plt.grid(True)
plt.legend()
plt.show()

bands = {
    "LFO": (0.00, 0.10),
    "Resp": (0.20, 0.40),
    "Card": (0.60, 1.40)}

band_powers = {}

for name, (fmin, fmax) in bands.items():
    mask = (freqs >= fmin) & (freqs <= fmax)
    band_powers[name] = np.trapz(power_signal[mask],freqs[mask])

text = "\n".join([f"{name}: {power:.3e}" for name, power in band_powers.items()])

plt.figure(figsize=(12,6))
plt.plot(freqs,power_signal,label="Sinal hiperamostrado",linewidth=2)
plt.plot(freqs,power_sampling,label="Operador de amostragem",alpha=0.8)

plt.axvspan( 0.00, 0.10,alpha=0.15,label="LFO")
plt.axvspan(0.20, 0.40,alpha=0.15,label="Respiração")
plt.axvspan(0.60, 1.40,alpha=0.15,label="Cardíaco")

plt.text(0.98,0.98,text,transform=plt.gca().transAxes,
    verticalalignment='top',
    horizontalalignment='right',
    bbox=dict(boxstyle="round",alpha=0.9))

plt.xlim(0, 1.5)
plt.xlabel("Frequência (Hz)")
plt.ylabel("Potência normalizada")
plt.title("NUFFT: sinal vs operador de amostragem")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()