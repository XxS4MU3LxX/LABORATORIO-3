no# LABORATORIO #3

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

# ANÁLISIS TEMPORAL Y ESPECTRAL (FFT)
    def plot_waveform_and_fft(signal, sample_rate, title):
      N = len(signal)
      T = 1.0 / sample_rate
      time = np.linspace(0.0, N * T, N)
      freq = np.fft.fftfreq(N, T)[:N // 2]
      fft_spectrum = np.fft.fft(signal)
      magnitude = 2.0 / N * np.abs(fft_spectrum[:N // 2])

 Analiza temporal y espectralmente las señales capturadas por los micrófonos ( MicL, MicS, MicSh).
 
Gráfica la forma de onda de la señal en el dominio del tiempo.

Calcula la FFT y gráfica el espectro de frecuencias en el dominio de la frecuencia.

En audio , la FFT permite:
- Ver qué frecuencias están presentes en una señal.
- Analizar el contenido espectral de una señal de voz o ruido.
- Identificar picos de frecuencia que corresponden a tonos o ruidos específicos.
- 
En este código, la FFT se usa para:
- Analizar el contenido espectral de las señales capturadas por los micrófonos.
- Comparar cómo cambia el espectro después de cada etapa de procesamiento:
- Original (capturada por los micrófonos)
- Beamforming (dirigida hacia una fuente)
- ICA (fuentes separadas)
- Normalización (filtrado final)
    
   # Forma de Onda en el Tiempo
      plt.figure(figsize=(14, 6))
      plt.subplot(1, 2, 1)
      plt.plot(time, signal, color='blue')
      plt.title(f"Forma de Onda - {title}")
      plt.xlabel("Tiempo (s)")
      plt.ylabel("Amplitud")
      plt.grid()

  Gráfica la forma de onda de la señal en el dominio del tiempo.

   # Espectro de Frecuencia (FFT)
      plt.subplot(1, 2, 2)
      plt.plot(freq, magnitude, color='red')
      plt.title(f"Espectro de Frecuencia (FFT) - {title}")
      plt.xlabel("Frecuencia (Hz)")
      plt.ylabel("Magnitud")
      plt.grid()
      plt.tight_layout()
      plt.show()

Gráfica el espectro de frecuencias de la señal en el dominio de la frecuencia.

![image](https://github.com/user-attachments/assets/c96ee8f5-ffd9-43d2-b207-df168a2d9def)
![image](https://github.com/user-attachments/assets/4045945e-8ad3-44f1-ad5e-df0dbdbe0719)
![image](https://github.com/user-attachments/assets/c081f693-df75-453a-b4a4-ed5f47180959)
![image](https://github.com/user-attachments/assets/453bf208-867a-4f10-98e9-0cea5a1e2fc8)
![image](https://github.com/user-attachments/assets/4b68dbb9-ec3f-454c-8b35-ab9a43df5e47)
![image](https://github.com/user-attachments/assets/50203e82-3df8-43d6-a9a4-68318de62e4a)

# Análisis Temporal y Espectral para cada micrófono
     plot_waveform_and_fft(data_L, sample_rate_L, "Micrófono L")
     plot_waveform_and_fft(data_S, sample_rate_S, "Micrófono S")
     plot_waveform_and_fft(data_Sh, sample_rate_Sh, "Micrófono Sh")

Se llama a la función `plot_waveform_and_fft()` para analizar las señales capturadas por:

`Micrófono L: ( data_L)`

`Micrófono S: ( data_S)`

`Micrófono Sh: ( data_Sh)`

Permite ver cómo varía la señal en el tiempo y frecuencia.

Identifique patrones o componentes de ruido que afecten la calidad de la señal.

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

Guarde las señales normalizadas (filtradas) como archivos de audio .wav.

Genera 3 archivos de audio correspondientes a las 3 señales separadas después de aplicar ICA y Normalización.

Facilita la reproducción, análisis posterior y comparación de las señales filtradas.

 
# ANÁLISIS ESPECTRAL PARA SEÑALES FILTRADAS (NORMALIZADAS)

    print("\n=== Análisis FFT de Señales Filtradas ===")
    plot_waveform_and_fft(normalized_signals[:, 0], sample_rate_L, "Normalizado 1 (Filtrado)")
    plot_waveform_and_fft(normalized_signals[:, 1], sample_rate_L, "Normalizado 2 (Filtrado)")
    plot_waveform_and_fft(normalized_signals[:, 2], sample_rate_L, "Normalizado 3 (Filtrado)")

Analiza espectralmente las 3 señales filtradas después de aplicar:
- Formación de haces
- ICA (Análisis de Componentes Independientes)
- Normalización (Filtrado final)
- Aplica la FFT a cada señal filtrada para obtener el espectro de frecuencias.
- 
Gráfica la forma de onda y el espectro de frecuencias de:

`Normalizado 1 (Filtrado)`

`Normalizado 2 (Filtrado)`

`Normalizado 3 (Filtrado)`

Permite comparar cómo cambia el espectro después de cada etapa de procesamiento.

Ayuda a evaluar la efectividad del filtrado.

# CÁLCULO DE SNR Y POTENCIA DE RUIDO

    def calculate_snr(signal, noise):
      signal_power = np.mean(signal ** 2)
      noise_power = np.mean(noise ** 2)
        if noise_power > 0:
          return 10 * np.log10(signal_power / noise_power)
        else:
          return -np.inf  # Manejo de división por cero

Cuantifica la calidad de la señal.
 
Calcula el SNR (relación señal-ruido) en decibeles (dB).
 
SNR mide la relación entre la potencia de la señal y la potencia del ruido:
![image](https://github.com/user-attachments/assets/b6ef11be-0290-4ddc-9b53-0d1731ffb269)

P_signal : Potencia de la señal.

P_noise : Potencia del ruido.

# Potencia del Ruido Ambiente
    potencia_ruido_ambiente = 10 * np.log10(np.mean(data_Noise ** 2))
    print("\n=== Potencia del Ruido Ambiente ===")
    print(f"Potencia de Ruido: {potencia_ruido_ambiente:.2f} dB")

Calcula la potencia del ruido ambiente en dB.

data_Noise es el ruido ambiente capturado.

Se calcula como la potencia media en dB:

![image](https://github.com/user-attachments/assets/67cf95a5-68a2-4e0f-afe1-05018bdf45ca)


# SNR de las señales en cada etapa
    snr_original = { 
      "MicL": calculate_snr(data_L, data_Noise), 
      "MicS": calculate_snr(data_S, data_Noise), 
      "MicSh": calculate_snr(data_Sh, data_Noise) 
    }

    snr_beamforming = { 
      f"Beamforming {i+1}": calculate_snr(beamformed_signals[:, i], data_Noise) 
      for i in range(beamformed_signals.shape[1]) 
    }

    snr_ica = { 
       f"ICA {i+1}": calculate_snr(enhanced_signals[:, i], data_Noise) 
       for i in range(enhanced_signals.shape[1]) 
    }

    snr_normalized = { 
      f"Normalizado {i+1}": calculate_snr(normalized_signals[:, i], data_Noise) 
      for i in range(normalized_signals.shape[1]) 
    }

- Calcula el SNR de las señales originales capturadas por los 3 micrófonos.

- Calcula el SNR de las 3 señales resultantes después de Beamforming.

- Calcula el SNR de las 3 señales obtenidas después de aplicar ICA (FastICA).

- Calcula el SNR de las 3 señales después de la Normalización.

- Cada señal se compara con el ruido ambiente `(data_Noise)` para calcular el SNR en dB.

# Impresión Completa de SNR
    print("\n=== SNR de las Señales ===")
    print("Original:", snr_original)
    print("Después de Beamforming:", snr_beamforming)
    print("Después de ICA:", snr_ica)
    print("Después de Normalización:", snr_normalized)

Imprime los valores de SNR calculados en cada etapa del procesamiento.

# SELECCIÓN DEL MEJOR SNR

    def select_best_snr_signal(snr_dict):
      best_snr_key = max(snr_dict, key=snr_dict.get)
      best_snr = snr_dict[best_snr_key]
      print(f"\n=== Mejor Señal Filtrada ===")
      print(f"Señal Seleccionada: {best_snr_key} con SNR de {best_snr:.2f} dB")

    select_best_snr_signal(snr_normalized)

Selecciona la señal filtrada con el mejor SNR.

Imprime la señal seleccionada y su valor de SNR.

Se utiliza para determinar cuál de las 3 señales normalizadas tiene la mejor calidad.
    
![image](https://github.com/user-attachments/assets/cd097f5d-019d-471c-a3a0-e112955fe93c)

# CONCLUSIONES
1. ¿Cómo afecta la posición relativa de los micrófonos y las fuentes sonoras en la efectividad de la separación de señales?
La posición de los micrófonos y las fuentes sonoras es clave para la separación de señales porque determina cómo cada micrófono captura la combinación de las voces. Si los micrófonos están demasiado cerca unos de otros o alineados con las fuentes, las señales captadas serán muy similares, dificultando la separación. En cambio, si están bien distribuidos en el espacio y a diferentes distancias de las fuentes, se pueden obtener diferencias en la amplitud y fase de las señales, facilitando el uso de técnicas como el Análisis de Componentes Independientes (ICA) o Beamforming para separar cada voz.


2. ¿Qué mejoras implementaría en la metodología para obtener mejores resultados?
Para mejorar los resultados, se podrían implementar las siguientes mejoras:

Aumentar el número de micrófonos: Más micrófonos permiten una mejor captura espacial de las señales, facilitando la separación.

Optimizar la ubicación de los micrófonos: Distribuirlos estratégicamente para maximizar las diferencias en la captura de cada fuente.

Aplicar técnicas avanzadas de procesamiento de señales: Usar filtros adaptativos, técnicas de reducción de ruido y métodos avanzados de separación de fuentes como Deep Learning.

Calibrar el sistema antes de la grabación: Medir el ruido ambiental y ajustar la ganancia de los micrófonos para mejorar la calidad de las grabaciones.

Incluir pruebas con diferentes niveles de ruido: Para evaluar la robustez del sistema ante condiciones más realistas.

