# LABORATORIO-3
import numpy as np
import matplotlib.pyplot as plt
import scipy.io.wavfile as wav
import scipy.signal as signal
from sklearn.decomposition import FastICA, PCA
from scipy.linalg import svd
from scipy.signal import wiener

# Cargar archivos de audio
micL_path = 'micL.wav'
micS_path = 'micS.wav'
micSh_path = 'micSh.wav'
noise_path = 'Sonido_ambiente.wav'

# Leer archivos de audio
sample_rate_L, data_L = wav.read(micL_path)
sample_rate_S, data_S = wav.read(micS_path)
sample_rate_Sh, data_Sh = wav.read(micSh_path)
sample_rate_Noise, data_Noise = wav.read(noise_path)

# Guardar en diccionarios para facilitar el acceso
audio_data = {"micL": data_L, "micS": data_S, "micSh": data_Sh}
sample_rates = {"micL": sample_rate_L, "micS": sample_rate_S, "micSh": sample_rate_Sh}

# Calcular la Relación Señal-Ruido (SNR)
def calculate_snr(signal, noise):
    signal_power = np.mean(signal ** 2)
    noise_power = np.mean(noise ** 2)
    return 10 * np.log10(signal_power / noise_power)

snr_values = {key: calculate_snr(data, data_Noise[:len(data)]) for key, data in audio_data.items()}
print("Valores de SNR:", snr_values)

# Visualización de las formas de onda y espectrogramas con subplots
fig = plt.figure(figsize=(15, 15))
for i, (key, data) in enumerate(audio_data.items()):
    sr = sample_rates[key]
    time = np.linspace(0, len(data) / sr, num=len(data))

    # Forma de onda
    plt.subplot(len(audio_data), 2, i * 2 + 1)
    plt.plot(time, data)
    plt.title(f"Forma de onda - {key}")
    plt.xlabel("Tiempo (s)")
    plt.ylabel("Amplitud")

    # Espectrograma
    f, t, Sxx = signal.spectrogram(data, sr)
    plt.subplot(len(audio_data), 2, i * 2 + 2)
    plt.imshow(10 * np.log10(Sxx), aspect='auto', origin='lower',
               extent=[t.min(), t.max(), f.min(), f.max()])
    plt.title(f"Espectrograma - {key}")
    plt.xlabel("Tiempo (s)")
    plt.ylabel("Frecuencia (Hz)")

plt.tight_layout()
plt.show()

# Aplicar Beamforming con SVD
min_length = min(len(data_L), len(data_S), len(data_Sh))
signals_matrix = np.vstack([data_L[:min_length], data_S[:min_length], data_Sh[:min_length]]).T

U, S, Vt = svd(signals_matrix, full_matrices=False)
beamformed_signals = U[:, :2] @ np.diag(S[:2])

# Aplicar Filtro de Wiener para reducción de ruido
denoised_signals = np.apply_along_axis(lambda x: wiener(x, mysize=29), 0, beamformed_signals)

# Visualización de las señales procesadas con subplots
plt.figure(figsize=(12, 6))
for i in range(2):
    plt.subplot(2, 1, i + 1)
    plt.plot(denoised_signals[:, i])
    plt.title(f"Señal filtrada con Wiener - Componente {i+1}")
    plt.xlabel("Muestras")
    plt.ylabel("Amplitud")

plt.tight_layout()
plt.show()

# Aplicar PCA para estabilizar las señales
pca = PCA(n_components=2, whiten=True, random_state=42)
pca_signals = pca.fit_transform(denoised_signals)

# Aplicar ICA para separación de fuentes
ica = FastICA(n_components=2, max_iter=3000, tol=0.0001, random_state=42)
enhanced_signals = ica.fit_transform(pca_signals)

# Visualización de las señales separadas con subplots
plt.figure(figsize=(12, 6))
for i in range(2):
    plt.subplot(2, 1, i + 1)
    plt.plot(enhanced_signals[:, i])
    plt.title(f"Señal Separada por ICA - Componente {i+1}")
    plt.xlabel("Muestras")
    plt.ylabel("Amplitud")

plt.tight_layout()
plt.show()

# Guardar las señales separadas
output_files = {}
for i in range(2):
    output_file = f"enhanced_voice_{i+1}.wav"
    wav.write(output_file, sample_rate_L, enhanced_signals[:, i].astype(np.int16))
    output_files[f"Voz Separada Mejorada {i+1}"] = output_file

# Mostrar los archivos generados
output_files
