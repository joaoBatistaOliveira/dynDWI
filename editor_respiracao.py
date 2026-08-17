import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class EditorRespiracao:
    def __init__(self, df, time_col="Time", resp_col="resp", 
                 peak_col="resp_peaks", valley_col="resp_valleys", 
                 click_tolerance=15):
        self.df = df.copy()
        self.time_col, self.resp_col = time_col, resp_col
        self.peak_col, self.valley_col = peak_col, valley_col
        self.click_tolerance = click_tolerance
        self.modo = "pico"
        
        for col in [time_col, resp_col]:
            if col not in self.df.columns:
                raise ValueError(f"Coluna '{col}' não encontrada.")
        
        for col in [peak_col, valley_col]:
            if col not in self.df.columns:
                self.df[col] = 0
            self.df[col] = pd.to_numeric(self.df[col], errors="coerce").fillna(0).astype(int)
        
        mask = self.df[time_col].notna() & self.df[resp_col].notna()
        self.idx = self.df.index[mask].to_numpy()
        self.x, self.y = self.df.loc[self.idx, time_col].to_numpy(), self.df.loc[self.idx, resp_col].to_numpy()
        self.fig = self.ax = self.peak_plot = self.valley_plot = None

    def iniciar(self):
        self.fig, self.ax = plt.subplots(figsize=(14, 6))
        self.ax.plot(self.x, self.y, "k", lw=1, label="Respiração")
        self.ax.set(xlabel=self.time_col, ylabel=self.resp_col)
        self.ax.grid(True, alpha=0.3)
        self._atualizar_marcadores()
        self._atualizar_titulo()
        self.fig.canvas.mpl_connect("button_press_event", self._evento_clique)
        self.fig.canvas.mpl_connect("key_press_event", self._evento_tecla)
        plt.tight_layout()
        plt.show()
        return self.df

    def _atualizar_marcadores(self):
        for plot in [self.peak_plot, self.valley_plot]:
            if plot is not None: plot.remove()
        
        for col, color, marker in [(self.peak_col, "red", "^"), (self.valley_col, "blue", "v")]:
            mask = self.df.loc[self.idx, col].to_numpy() == 1
            plot = self.ax.scatter(self.x[mask], self.y[mask], marker=marker, color=color, s=70, zorder=5)
            if col == self.peak_col: self.peak_plot = plot
            else: self.valley_plot = plot
        
        self.ax.legend(*self.ax.get_legend_handles_labels(), loc="upper right")
        self.fig.canvas.draw_idle()

    def _atualizar_titulo(self):
        nomes = {"pico": "ADICIONAR PICO", "vale": "ADICIONAR VALE", "remover": "REMOVER MARCADOR"}
        self.ax.set_title(f"Editor de respiração | Modo: {nomes[self.modo]} | 1=Pico | 2=Vale | 3=Remover | Tolerância: {self.click_tolerance}px")
        self.fig.canvas.draw_idle()

    def _evento_tecla(self, event):
        if event.key in ["1", "2", "3"]:
            self.modo = {"1": "pico", "2": "vale", "3": "remover"}[event.key]
            self._atualizar_titulo()

    def _evento_clique(self, event):
        if event.inaxes != self.ax or event.x is None or event.y is None: return
        if self.modo == "pico": self._adicionar_marcador(event.x, event.y, "pico")
        elif self.modo == "vale": self._adicionar_marcador(event.x, event.y, "vale")
        elif self.modo == "remover": self._remover_marcador(event.x, event.y)

    def _ponto_curva_mais_proximo(self, mouse_x, mouse_y):
        pts = self.ax.transData.transform(np.column_stack((self.x, self.y)))
        dist = np.linalg.norm(pts - np.array([mouse_x, mouse_y]), axis=1)
        pos = np.argmin(dist)
        return (self.idx[pos], dist[pos]) if dist[pos] <= self.click_tolerance else (None, dist[pos])

    def _adicionar_marcador(self, mouse_x, mouse_y, tipo):
        idx, dist = self._ponto_curva_mais_proximo(mouse_x, mouse_y)
        if idx is None: return
        self.df.loc[idx, self.peak_col if tipo == "pico" else self.valley_col] = 1
        self.df.loc[idx, self.valley_col if tipo == "pico" else self.peak_col] = 0
        self._atualizar_marcadores()

    def _remover_marcador(self, mouse_x, mouse_y):
        markers = np.concatenate([self.df.index[self.df[c] == 1].to_numpy() for c in [self.peak_col, self.valley_col]])
        if len(markers) == 0: return
        
        pts = self.ax.transData.transform(np.column_stack((self.df.loc[markers, self.time_col], self.df.loc[markers, self.resp_col])))
        dist = np.linalg.norm(pts - np.array([mouse_x, mouse_y]), axis=1)
        pos = np.argmin(dist)
        
        if dist[pos] <= self.click_tolerance:
            self.df.loc[markers[pos], [self.peak_col, self.valley_col]] = 0
            self._atualizar_marcadores()

def editar_respiracao(caminho_csv, time_col="Time", resp_col="resp", 
                      peak_col="resp_peaks", valley_col="resp_valleys", 
                      click_tolerance=15, caminho_saida=None):
    df = pd.read_csv(caminho_csv)
    df_editado = EditorRespiracao(df, time_col, resp_col, peak_col, valley_col, click_tolerance).iniciar()
    
    if caminho_saida is None:
        base, ext = os.path.splitext(caminho_csv)
        caminho_saida = f"{base}_edited{ext}"
    
    df_editado.to_csv(caminho_saida, index=False)
    print(f"\nEdição concluída.\nArquivo original:\n{caminho_csv}\n\nArquivo editado:\n{caminho_saida}")
    return df_editado