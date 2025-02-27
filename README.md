# LABORATORIO #3

***Luz Marina Valderrama-5600741***

***Shesly Nicole Colorado - 5600756***

***Samuel Esteban Fonseca Luna - 5600808***

Este código aborda el problema del cóctel, donde se mezclan múltiples fuentes de audio (como voces en un ambiente ruidoso) y se necesita separar la voz principal o fuente de interés del ruido de fondo y otras voces, donde en este caso grabamos 3 audios de los tres integrantes del grupo hablablando al mismo tiempo.

Para lograrlo, se utiliza una técnica de procesamiento de señales en 4 fases:

-  *Beamforming:* Enfocar la fuente principal de voz utilizando múltiples micrófonos.
-  *ICA (FastICA):* Separar las fuentes mezcladas en componentes independientes.
 - *Normalización:* Ajustar la amplitud de las señales para una mejor calidad de audio.
-  *Comparación de SNR (Signal-to-Noise Ratio):* Evaluar la calidad de la señal filtrada en cada etapa.
 - *Selección y Guardado del Mejor Audio Filtrado:* Selecciona la señal con el mejor SNR y la guarda como .wav para escuchar y evaluar la calidad de la voz separada.


# Librerías Necesarias
    import numpy as np
    import matplotlib.pyplot as plt
    import scipy.io.wavfile as wav
    from sklearn.decomposition import FastICA
    from scipy.linalg import svd
    import os

 `numpy:` Para operaciones matemáticas y manipulación de matrices.
 `matplotlib.pyplot:` Para generar gráficas de SNR y análisis de señales.
 `scipy.io.wavfile:` Para leer y guardar archivos de audio .wav.
 `sklearn.decomposition.FastICA:` Para separación de fuentes mediante ICA.
 `scipy.linalg.svd:` Para Beamforming utilizando SVD (Singular Value Decomposition).
 `os:` Para manejo de directorios y guardar archivos.

# Definir carpeta de salida
    output_dir = r"C:\Users\Lenovo\.spyder-py3\jejeje"
    if not os.path.exists(output_dir):
    os.makedirs(output_dir)

Se define el directorio de salida donde se guardarán:
- Audios filtrados y señales separadas.
- Resultados finales como gráficos y archivos .wav.
La variable output_dir contiene la ruta completa de la carpeta de salida.

 `os.path.exists(output_dir):` Verifica si la carpeta ya existe en el sistema de archivos.
 `os.makedirs(output_dir):` Si la carpeta no existe, la crea automáticamente.
Este enfoque previene errores si la carpeta ya existe y asegura que los archivos de salida se guarden correctamente.
    
# Cargar archivos de audio
    micL_path = 'micL.wav'
    micS_path = 'micS.wav'
    micSh_path = 'micSh.wav'
    noise_path = 'Sonido_ambiente.wav'


`micL.wav:` Señal de micrófono izquierdo (Luz).

`micS.wav:` Señal de micrófono central (Samuel).

`micSh.wav:` Señal de micrófono derecho (Shesly).

`Sonido_ambiente.wav:` Ruido de fondo o sonido ambiente.

**¿Por qué se utilizan múltiples micrófonos?:**

Se utilizan múltiples micrófonos para:
- Capturar la voz principal desde diferentes posiciones.
- Ayudar en la separación de fuentes utilizando diferencias espaciales.
- Mejorar la direccionalidad y reducir el ruido de fondo mediante Beamforming.
  
**¿Qué es el archivo de ruido ambiente?:**

Es una grabación del ruido de fondo sin ninguna fuente de voz principal. 

Se utiliza para calcular el SNR y comparar la calidad de las señales filtradas.

También se utiliza como referencia de ruido para evaluar la efectividad de la separación de fuentes.

# Leer archivos de audio
    sample_rate_L, data_L = wav.read(micL_path)
    sample_rate_S, data_S = wav.read(micS_path)
    sample_rate_Sh, data_Sh = wav.read(micSh_path)
    sample_rate_Noise, data_Noise = wav.read(noise_path)

Se leen los archivos de audio utilizando `wav.read()` de `scipy.io.wavfile`.

Cada archivo se descompone en dos partes:

`sample_rate_X:` La frecuencia de muestreo (en Hz), que indica cuántas muestras por segundo tiene la señal de audio.

`data_X:` La señal de audio en sí, almacenada como un arreglo NumPy.

**¿Por qué se utiliza .wav como formato de archivo?:**

.wav es un formato de audio sin pérdida (lossless), lo que significa que:
- No comprime ni modifica la señal de audio original.
- Permite un procesamiento preciso sin pérdida de calidad.

# Recortar todas las señales al mismo tamaño mínimo
    min_length = min(len(data_L), len(data_S), len(data_Sh), len(data_Noise))
    data_L, data_S, data_Sh, data_Noise = data_L[:min_length], data_S[:min_length], data_Sh[:min_length], data_Noise[:min_length]

1. Encuentra la Longitud Mínima:

`min_length = min(len(...)):` Encuentra la longitud más corta entre todas las señales `(data_L, data_S, data_Sh, data_Noise)`.

Esto asegura que ninguna señal sea más larga que las demás.

2. Recorta Todas las Señales:

`Utiliza slicing (:)` para recortar cada señal hasta la longitud mínima.

De esta manera, todas las señales terminan con exactamente el mismo número de muestras.

Este enfoque previene errores en etapas posteriores que requieren procesar las señales en paralelo.

**¿Por qué se utiliza `min()` en lugar de `max()`?**

Si se utilizara `max()`, algunas señales serían más cortas, lo que provocaría:
- Errores de Indexación al intentar acceder a índices inexistentes.
- Pérdida de sincronización en las señales durante el procesamiento conjunto.
- Distorsiones al aplicar algoritmos como ICA y Beamforming.

# Aplicar Beamforming con SVD
    signals_matrix = np.column_stack((data_L, data_S, data_Sh))
    U, S, Vt = svd(signals_matrix, full_matrices=False)
    beamformed_signals = U[:, :3] @ np.diag(S[:3])

**¿Qué es Beamforming y por qué es importante?**

Beamforming es una técnica utilizada para enfocar una fuente de sonido utilizando múltiples micrófonos. 

Funciona aprovechando las diferencias de tiempo de llegada y diferencias de amplitud de la señal en cada micrófono.

Permite:
- Enfocar la voz principal y reducir el ruido de fondo.
- Mejorar la direccionalidad al enfocarse en una fuente específica.
- Atenuar señales no deseadas provenientes de otras direcciones.
  
**¿Por qué se utiliza SVD para Beamforming?**

SVD (Singular Value Decomposition) descompone una matriz en componentes principales, revelando:
- Direcciones dominantes de la señal.
- Energía y contribución de cada fuente en la mezcla.
SVD se utiliza aquí para:
- Identificar las direcciones principales de llegada de la señal de voz.
- Separar las señales mezcladas en componentes con máxima energía.

`np.column_stack(...)` concatena las señales de los 3 micrófonos en una sola matriz.

La matriz resultante `(signals_matrix)`

`svd()` descompone la matriz de señales en 3 componentes:

`U:` Matriz de direcciones principales de la señal. Cada columna representa una dirección en el espacio de mezclas.

`S:` Vector de valores singulares. Representa la energía o intensidad de cada componente de la señal.

`Vt:` Matriz de componentes fuente. Cada fila representa una fuente independiente en el espacio original.

`full_matrices=False` optimiza el cálculo, manteniendo solo las componentes necesarias.

El producto de matrices `U[:, :3] @ np.diag(S[:3])` representa:
- La combinación óptima de señales para enfocar la fuente principal.
- Una señal que maximiza la relación señal/ruido (SNR).
- Una señal que atenúa componentes de ruido provenientes de otras direcciones.
  
# Aplicar ICA
    ica = FastICA(n_components=3, max_iter=3000, tol=0.0001, random_state=42)
    enhanced_signals = ica.fit_transform(beamformed_signals)

**¿Qué es ICA (Independent Component Analysis)?**

ICA es un método de separación de fuentes ciegas que asume que:
- Las señales mezcladas son combinaciones lineales de fuentes independientes.
- Las fuentes originales son estadísticamente independientes entre sí.
- Cada fuente tiene una distribución no gaussiana.
Objetivo de ICA:
- Separar señales mezcladas en fuentes independientes.
- En este contexto, extraer la voz principal y reducir el ruido de fondo.

**¿Por qué se utiliza FastICA en lugar de ICA estándar?**

FastICA es una versión rápida y optimizada de ICA que:
- Utiliza un algoritmo iterativo de máxima verosimilitud.
- Maximiza la no gaussianidad de las señales separadas, usando kurtosis o entropía negada.
- Es computacionalmente más eficiente y converge más rápido que ICA estándar.
- Es ideal para señales de voz, que tienden a tener una distribución no gaussiana.

**¿Cómo Funciona ICA?**

ICA utiliza un modelo de mezcla lineal: *X=A⋅S*

Donde:

- X: Señal observada (mezcla de fuentes).
- A: Matriz de mezcla desconocida.
- S: Fuentes independientes originales (voz, ruido, interferencias).

ICA busca una matriz de separación W tal que: *S=W⋅X*
- W se calcula maximizando la independencia estadística.
- Utiliza no gaussianidad como criterio de independencia.

# Normalizar señales antes de guardar
    def normalize_signal(signal):
    signal = signal - np.mean(signal)
    max_val = np.max(np.abs(signal))
     if max_val > 0:
        signal = signal / max_val
        signal = (signal * 32767).astype(np.int16)
     return signal

    normalized_signals = np.apply_along_axis(normalize_signal, 0, enhanced_signals)

**¿Por Qué es Necesaria la Normalización?**

Después de aplicar ICA, las señales separadas pueden tener:
- Amplitud muy alta o muy baja, dependiendo de la mezcla original.
- Desbalance en la energía debido a la separación independiente.
- Distorsión o saturación al guardarlas como .wav.
  
La normalización asegura que todas las señales tengan:
- Amplitud adecuada y balanceada.
- Rango de valores apropiado para guardar el archivo de audio.
- Calidad de voz consistente y sin distorsión.

# Guardar señales normalizadas en archivos .wav
    output_files_normalized = {}
    for i in range(3):
     output_file = os.path.join(output_dir, f"enhanced_voice_final_{i+1}.wav")
     wav.write(output_file, sample_rate_L, normalized_signals[:, i])
     output_files_normalized[f"Voz Separada {i+1}"] = output_file

    print("Archivos generados:", output_files_normalized)


 
# Función para calcular SNR
    def calculate_snr(signal, noise):
     signal_power = np.mean(signal ** 2)
     noise_power = np.mean(noise ** 2)
      if noise_power > 0:
        return 10 * np.log10(signal_power / noise_power)
      else:
         return -np.inf  # Manejo de división por cero

 

# Cálculo de SNR para las señales originales
    snr_original = {
      "MicL": calculate_snr(data_L, data_Noise),
      "MicS": calculate_snr(data_S, data_Noise),
      "MicSh": calculate_snr(data_Sh, data_Noise)
    }



# Cálculo de SNR para las señales después de Beamforming
    snr_beamforming = {}
    for i in range(beamformed_signals.shape[1]):
      snr_beamforming[f"Beamforming {i+1}"] = calculate_snr(beamformed_signals[:, i], data_Noise)



# Cálculo de SNR para las señales después de ICA
    snr_ica = {}
    for i in range(enhanced_signals.shape[1]):
      snr_ica[f"ICA {i+1}"] = calculate_snr(enhanced_signals[:, i], data_Noise)

 

# Cálculo de SNR para las señales Normalizadas
 snr_normalized = {}
    for i in range(normalized_signals.shape[1]):
      snr_normalized[f"Normalizado {i+1}"] = calculate_snr(normalized_signals[:, i], data_Noise)

      

# Cálculo de Potencia del Ruido Ambiente
    potencia_ruido_ambiente = 10 * np.log10(np.mean(data_Noise ** 2))

# Imprimir Potencia del Ruido Ambiente
    print("\n=== Potencia del Ruido Ambiente ===")
    print(f"Potencia de Ruido: {potencia_ruido_ambiente:.2f} dB")

# Imprimir resultados en consola
    print("\n=== SNR de las Señales ===")
    print("Original:", snr_original)
    print("Después de Beamforming:", snr_beamforming)
    print("Después de ICA:", snr_ica)
    print("Después de Normalización:", snr_normalized)

# === Selección y Guardado del Audio Filtrado Final ===

# Función para seleccionar la señal con el mejor SNR
    def select_best_snr_signal(snr_dict, signals_matrix):
      best_snr_key = max(snr_dict, key=snr_dict.get)
      best_index = int(best_snr_key.split()[-1]) - 1  # Extraer el índice de la mejor señal
      best_signal = signals_matrix[:, best_index]
      print(f"\n=== Mejor Señal Filtrada ===")
      print(f"Señal Seleccionada: {best_snr_key} con SNR de {snr_dict[best_snr_key]:.2f} dB")
      return best_signal

# Seleccionar la mejor señal normalizada
    best_filtered_signal = select_best_snr_signal(snr_normalized, normalized_signals)

# Guardar el archivo de audio filtrado final
    output_file_filtered = os.path.join(output_dir, "filtered_voice_final.wav")
    wav.write(output_file_filtered, sample_rate_L, best_filtered_signal.astype(np.int16))

    print(f"\n=== Archivo de Audio Filtrado Guardado ===")
    print(f" Archivo Generado: {output_file_filtered}")

# Graficar SNR en diferentes etapas
    def plot_snr_comparison(snr_dicts, titles, potencia_ruido):
     plt.figure(figsize=(14, 12))
      for i, (snr_dict, title) in enumerate(zip(snr_dicts, titles)):
         labels = list(snr_dict.keys())
         values = list(snr_dict.values())
        
         plt.subplot(3, 2, i+1)
         plt.bar(labels, values, color='skyblue', label="SNR Señales")
         plt.axhline(y=potencia_ruido, color='red', linestyle='--', label="Potencia Ruido Ambiente")
         plt.title(title)
         plt.xlabel("Señales")
         plt.ylabel("SNR (dB)")
         plt.xticks(rotation=45)
         plt.grid(True, linestyle='--', alpha=0.7)
         plt.legend()
    
      plt.tight_layout()
      plt.show()

    plot_snr_comparison(
      [snr_original, snr_beamforming, snr_ica, snr_normalized],
      ["Original", "Beamforming", "ICA", "Normalización"],
      potencia_ruido_ambiente
    )







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
signals_matrix = np.column_stack((data_L[:min_length], data_S[:min_length], data_Sh[:min_length]))
print("Dimensiones correctas de signals_matrix:", signals_matrix.shape)

# Visualizar señales originales alineadas antes de aplicar SVD
plt.figure(figsize=(15, 10))
for i in range(3):
    plt.subplot(3, 1, i + 1)
    plt.plot(signals_matrix[:, i])
    plt.title(f"Señal Original Alineada - Mic {i+1}")
    plt.xlabel("Muestras")
    plt.ylabel("Amplitud")
plt.tight_layout()
plt.show()

# Aplicar SVD correctamente y mostrar las dimensiones de U, S y Vt
U, S, Vt = svd(signals_matrix, full_matrices=False)
print("Dimensiones de U:", U.shape)
print("Dimensiones de S:", S.shape)
print("Dimensiones de Vt:", Vt.shape)

# Verificar los valores singulares
print("Valores singulares (S):", S[:10])  # Mostrar solo los primeros 10 valores singulares

# Reconstruir beamformed_signals con más componentes
beamformed_signals = U[:, :2] @ np.diag(S[:2])
print("Dimensiones de beamformed_signals (ajustado):", beamformed_signals.shape)

# Visualizar las señales después de Beamforming
plt.figure(figsize=(15, 10))
for i in range(2):
    plt.subplot(2, 1, i + 1)
    plt.plot(beamformed_signals[:, i])
    plt.title(f"Beamformed Signal - Componente {i+1}")
    plt.xlabel("Muestras")
    plt.ylabel("Amplitud")
plt.tight_layout()
plt.show()

# Aplicar Filtro de Wiener para reducción de ruido
denoised_signals = np.apply_along_axis(lambda x: wiener(x, mysize=29), 0, beamformed_signals)

# Visualización de las señales procesadas con subplots
plt.figure(figsize=(12, 6))
for i in range(2):
    plt.subplot(2, 1, i + 1)
    plt.plot(denoised_signals[:, i])
    plt.title(f"Señal Filtrada con Wiener - Componente {i+1}")
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

# Función para normalizar señales
def normalize_signal(signal):
    signal = signal - np.mean(signal)  # Eliminar DC
    max_val = np.max(np.abs(signal))
    if max_val > 0:
        signal = signal / max_val  # Normalizar a [-1, 1]
        signal = (signal * 32767).astype(np.int16)  # Escalar a int16
    return signal

# Normalizar las señales antes de guardarlas
normalized_signals = np.apply_along_axis(normalize_signal, 0, enhanced_signals)

# Guardar las señales separadas normalizadas
output_files_normalized = {}
for i in range(2):
    output_file = f"enhanced_voice_final_{i+1}.wav"
    wav.write(output_file, sample_rate_L, normalized_signals[:, i])
    output_files_normalized[f"Voz Separada Final Normalizada {i+1}"] = output_file

# Mostrar los archivos generados
print("Archivos de audio generados:", output_files_normalized)
