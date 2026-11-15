import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, freqz

# ==========================================
# 1. Definición de la señal de entrada
# ==========================================
fs = 1000.0       # Frecuencia de muestreo (1 kHz)
t = np.arange(0, 1.0, 1.0/fs)  # Vector de tiempo (1 segundo)

# Señal compuesta: 50 Hz, 150 Hz y 300 Hz
f1, f2, f3 = 50.0, 150.0, 300.0
senal_pura = np.sin(2 * np.pi * f1 * t) + 0.5 * np.sin(2 * np.pi * f2 * t) + 0.2 * np.sin(2 * np.pi * f3 * t)

# Añadir ruido blanco aleatorio
ruido = np.random.normal(0, 0.3, len(t))
senal_ruidosa = senal_pura + ruido

# ==========================================
# 2. Diseño de los filtros (Butterworth IIR)
# ==========================================
orden = 4
nyq = 0.5 * fs  # Frecuencia de Nyquist

# Filtro Pasa Bajos (Corte en 80 Hz para dejar pasar solo los 50 Hz)
corte_bajo = 80.0 / nyq
b_low, a_low = butter(orden, corte_bajo, btype='low')

# Filtro Pasa Altos (Corte en 250 Hz para dejar pasar solo los 300 Hz y ruido de alta frec.)
corte_alto = 250.0 / nyq
b_high, a_high = butter(orden, corte_alto, btype='high')

# Filtro Pasa Bandas (Paso entre 100 Hz y 200 Hz para aislar los 150 Hz)
corte_banda = [100.0 / nyq, 200.0 / nyq]
b_band, a_band = butter(orden, corte_banda, btype='band')

# ==========================================
# 3. Aplicación del filtro a la señal
# ==========================================
# Se usa filtfilt para aplicar el filtro hacia adelante y hacia atrás (fase cero)
senal_filtrada_low = filtfilt(b_low, a_low, senal_ruidosa)
senal_filtrada_high = filtfilt(b_high, a_high, senal_ruidosa)
senal_filtrada_band = filtfilt(b_band, a_band, senal_ruidosa)

# ==========================================
# 4. Visualización de los resultados
# ==========================================
plt.figure(figsize=(12, 10))

# Señal Original
plt.subplot(4, 1, 1)
plt.plot(t, senal_ruidosa, label='Señal con Ruido', color='lightgray')
plt.plot(t, senal_pura, label='Señal Original Pura', color='black', alpha=0.8)
plt.title('Señal de Entrada (Original vs Ruidosa)')
plt.xlabel('Tiempo [s]')
plt.ylabel('Amplitud')
plt.legend(loc='upper right')

# Señal Filtrada Pasa Bajos
plt.subplot(4, 1, 2)
plt.plot(t, senal_filtrada_low, color='blue', label='Filtro Pasa Bajos (< 80 Hz)')
plt.title('Señal tras Filtro Pasa Bajos (Aísla componente de 50 Hz)')
plt.xlabel('Tiempo [s]')
plt.ylabel('Amplitud')
plt.legend(loc='upper right')

# Señal Filtrada Pasa Bandas
plt.subplot(4, 1, 3)
plt.plot(t, senal_filtrada_band, color='green', label='Filtro Pasa Bandas (100-200 Hz)')
plt.title('Señal tras Filtro Pasa Bandas (Aísla componente de 150 Hz)')
plt.xlabel('Tiempo [s]')
plt.ylabel('Amplitud')
plt.legend(loc='upper right')

# Señal Filtrada Pasa Altos
plt.subplot(4, 1, 4)
plt.plot(t, senal_filtrada_high, color='red', label='Filtro Pasa Altos (> 250 Hz)')
plt.title('Señal tras Filtro Pasa Altos (Aísla componente de 300 Hz + Ruido)')
plt.xlabel('Tiempo [s]')
plt.ylabel('Amplitud')
plt.legend(loc='upper right')

plt.tight_layout()
plt.show()