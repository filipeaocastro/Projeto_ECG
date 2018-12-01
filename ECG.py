# -*- coding: utf-8 -*-
"""
Created on Tue Oct 23 16:18:20 2018

@author: Filipe Augusto
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import csv
from scipy.signal import welch
from scipy.fftpack import fft
import sys
import wfdb
import array as arr
from scipy.signal import butter, lfilter, detrend, fftconvolve
from detect_peaks import detect_peaks
import dspplot


def get_fft_values(y_values, T, N, f_s):
    f_values = np.linspace(0.0, 1.0/(2.0*T), N//2)
    fft_values_ = fft(y_values)
    fft_values = 2.0/N * np.abs(fft_values_[0:N//2])
    return f_values, fft_values

def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y

def find_angular_freq(lin_freq):
    ang_freq = lin_freq * 2 * np.pi
    return ang_freq

def plot_fft(): # Plota a FFT do sinal original e do filtrado
    #Plotando a FFT
    plt.figure(1)
    plt.subplot(211)
    f_values, fft_values = get_fft_values(signal[:,1], T, n_samples, fs)
    plt.plot(f_values, fft_values, linestyle='-', color='blue', linewidth = 1, label = 'Raw ECG FFT')
    plt.legend(loc='upper right')
    #plt.ylim(ymax = 0.04, ymin = -0.003)
    
    plt.subplot(212)
    f_values, fft_values = get_fft_values(signal_filtered, T, n_samples, fs)
    plt.plot(f_values, fft_values, linestyle='-', color='blue', linewidth = 1, label = 'Filtered ECG FFT')
    plt.legend(loc='upper right')
    plt.show()
    
 # Plota até dois sinais
def plot_signals(sig1, sig2 = 0, both = False, lb1 = '', lb2 = '', mean = False, fig = 2):
    plt.figure(fig)
    m1 = np.mean( np.abs(sig1))
    m2 = np.mean( np.abs(sig2))
    if both == False:
        plt.plot(signal[:,0], sig1, color = 'orange', linewidth = 1, label = 'ECG ' + lb1)
        plt.legend(loc='upper right')
        plt.xlabel("seconds")
        plt.ylabel("mV")
        if mean:
            plt.axhline(m1, color = 'red', linewidth = 1.2, linestyle = '--')
        
    else:
        plt.subplot(211)
        plt.plot(signal[:,0], sig1, color = 'orange', linewidth = 1, label = 'ECG ' + lb1)
        plt.legend(loc='upper right')
        plt.xlabel("seconds")
        plt.ylabel("mV")
        if mean:
            plt.axhline(m1, color = 'red', linewidth = 1.2, linestyle = '--')
        
        plt.subplot(212)
        plt.plot(signal[:,0], sig2, color = 'blue', linewidth = 1, label = 'ECG ' + lb2)
        plt.xlabel("seconds")
        plt.ylabel("mV")
        plt.legend(loc='upper right')
        if mean:
            plt.axhline(m2, color = 'red', linewidth = 1.2, linestyle = '--')
    plt.show()

def read_signal(name, time = 0): # Lê os aquivos do WFDB
    data = wfdb.rdsamp(name, sampto = 100)
    fs = data[1]["fs"]
    T = 1 / fs
    if(time != 0):
        n_samples = fs * time
        data = wfdb.rdsamp(name, sampto = n_samples)
    else:
        data = wfdb.rdsamp(name)
        n_samples = data[1]["sig_len"]
    t_n = n_samples * T
    t = np.arange(0, t_n, T)
    
    if len(t) < n_samples:
        t = np.arange(0, t_n + T, T)
    if len(t) > n_samples:
        t = np.arange(0, t_n - T, T)
    x = np.array(data[0])
    signal = np.array((t, x[:,0], x[:,1]))
    signal = np.transpose(signal, (1, 0))
    return signal, fs, T, n_samples

def QRS_onset_end (diff, TH, millis, PK, length, onset = True):
    fdiff = diff.astype(float)
    time = fs * millis/1000
    time = int(time)
    QRS_border = np.zeros(length)
    for i, j, k in zip(PK, TH, range(0, len(QRS_border))):
        var = np.zeros(2)
        aux1 = 0
        aux0 = 1
        while True:
            if onset:
                idx0 = i - aux0
                idx1 = i - aux1
            else:
                idx1 = i + aux0
                idx0 = i + aux1
                
            var[0] = fdiff[idx0]
            var[1] = fdiff[idx1]
            sub0 = var[0] - j
            sub1 = var[1] - j
            if onset:
                cross = (sub0 >= 0 and sub1 < 0)
                comp_dif = abs(sub0) < abs(sub1)
            else:
                cross = (sub1 >= 0 and sub0 < 0)
                comp_dif = abs(sub0) < abs(sub1)
            if cross:
                if abs(sub0) < abs(sub1):
                    QRS_border[k] = idx0
                else:
                    QRS_border[k] = idx1
                break
            if aux0 >= time:
                QRS_border[k] = idx0
                break
            aux1 += 1
            aux0 += 1
    QRS_border = QRS_border.astype(int)
    return QRS_border

def detectPK(signal_filtered, diff, QRS_index):
    
    samples = int(fs * 0.150) # Pega o número de amostras em 150 ms
    # Cria vetores para armazenar o PKa e PKb
    PKa = np.zeros(len(R_peaks), dtype = int)
    PKb = np.zeros(len(R_peaks), dtype = int)
    for i in range(0, len(QRS_index[:,2]) ):
        
        for k in range(QRS_index[i, 2], QRS_index[i, 2] - samples, -1):        
            pico = np.zeros(1)
            pico = detect_peaks( (diff[k-2], diff[k-1], diff[k]) )
            if pico != 0:
                PKa[i] = k-1
                break    
        
        for j in range(QRS_index[i, 2], QRS_index[i, 2] + samples):
            pico = np.zeros(1)
            pico = detect_peaks( (diff[j], diff[j+1], diff[j+2]) )
            if pico != 0:
                PKb[i] = j+1
                break  
            
        idx = QRS_index[i, 2]
        for j in range(PKb[i], QRS_index[i, 2] + samples, 1):
            if(j >= len(signal_filtered)):
                break
            if signal_filtered[j] < signal_filtered[idx] :
                idx = j
        QRS_index[i, 3] = idx - 1
            
        idx = QRS_index[i, 2]
        for k in range(PKa[i], QRS_index[i, 2] - samples, -1):
            if(j >= len(signal_filtered)):
                break
            if signal_filtered[k] < signal_filtered[idx]:
                idx = k
        QRS_index[i, 1] = idx 
    return PKa, PKb, QRS_index[:, 1], QRS_index[:, 2]

PKa, PKb, QRS_index[:, 1], QRS_index[:, 2] = detectPK(signal_filtered, diff, QRS_index)
        
# Lê o sinal e as anotações
sig_name = '100'
signal, fs, T, n_samples = read_signal(sig_name, time = 100)
ann = wfdb.rdann(sig_name, 'atr', sampto = n_samples)
ann = np.array(ann)

# Define as frequências de corte do filtro
lowcut = 0.5
highcut = 50

# Filtra o sinal
signal_filtered = butter_bandpass_filter(signal[:,1], lowcut, highcut, fs, order = 2)
signal_detrended = detrend(signal_filtered)

# Faz o TEO do sinal ECG
signal_TEO = np.zeros(shape = len(signal_filtered))
for i in range(1, len(signal[:,1]) - 1):
    signal_TEO[i] = np.power(signal_filtered[i], 2) - signal_filtered[i+1] * signal_filtered[i-1]

# Cria a janela de Bartlett
window = np.bartlett(9)

# Convolui o sinal TEO com a janela Bartlett
signal_TEO_windowed = fftconvolve(signal_TEO, window, mode='same')

# Define o threshold de detecção de pico como média do sinal + desvio padrão
th = np.mean(signal_TEO_windowed) + np.std(signal_TEO_windowed)

# Detecta os picos R
R_peaks_init = detect_peaks(signal_TEO_windowed, mph = th) - 1

# Faz a derivada do sinal
diff = np.diff(signal_filtered)
diff_mean = np.mean(diff) - np.std(diff)

# ******************* EXCLUIR ONDAS R QUE ESTEJAM A MENOS DE 0.5 s UMA DA OUTRA *******************
R_peaks = []

for i in range(0, len(R_peaks_init) - 1):
    dist = signal[R_peaks_init[i+1], 0] - signal[R_peaks_init[i], 0]
    if(dist >= 0.4):
        R_peaks.append(R_peaks_init[i])
        

# Cria uma matriz vazia para armazenar os índices dos QRS e armazena as ondas R
QRS_index = np.zeros((len(R_peaks), 5), dtype = int)
QRS_index[:, 2] = R_peaks[:]

samples = int(fs * 0.150) # Pega o número de amostras em 150 ms

# Cria vetores para armazenar o PKa e PKb
PKa = np.zeros(len(R_peaks), dtype = int)
PKb = np.zeros(len(R_peaks), dtype = int)

# Preenche a metriz QRS_index, PKa e PKb
for i in range(0, len(QRS_index[:,2]) ):
    
    for k in range(QRS_index[i, 2], QRS_index[i, 2] - samples, -1):        
        pico = np.zeros(1)
        pico = detect_peaks( (diff[k-2], diff[k-1], diff[k]) )
        if pico != 0:
            PKa[i] = k-1
            break    
    
    for j in range(QRS_index[i, 2], QRS_index[i, 2] + samples):
        pico = np.zeros(1)
        pico = detect_peaks( (diff[j], diff[j+1], diff[j+2]) )
        if pico != 0:
            PKb[i] = j+1
            break  
        
    idx = QRS_index[i, 2]
    for j in range(PKb[i], QRS_index[i, 2] + samples, 1):
        if(j >= len(signal_filtered)):
            break
        if signal_filtered[j] < signal_filtered[idx] :
            idx = j
    QRS_index[i, 3] = idx - 1
        
    idx = QRS_index[i, 2]
    for k in range(PKa[i], QRS_index[i, 2] - samples, -1):
        if(j >= len(signal_filtered)):
            break
        if signal_filtered[k] < signal_filtered[idx]:
            idx = k
    QRS_index[i, 1] = idx 
        
# Fazer um loop para achar o primeiro ponto de mínimo antes e depois do Qp e Sp, respectivamente
PKQ = np.zeros(len(QRS_index[:, 1]))
count = 0
for i in QRS_index[:, 1]:
    aux = 20
    PKQ_init = detect_peaks([4, 4, 4], valley = True)
    #while PKQ_init.size == 0:
    wdw_min = i - aux
    PKQ_init = detect_peaks(diff[wdw_min:i], valley = True)
    #aux += 1
    
    PKQ[count] = PKQ_init[len(PKQ_init)-1] + i - aux
    count += 1

PKS = np.zeros(len(QRS_index[:, 3]))
count = 0    
for i in QRS_index[:, 3]:
    aux = 20
    PKS_init = detect_peaks([4, 4, 4], valley = True)
    wdw_max = i + aux
    PKS_init = detect_peaks(diff[i:wdw_max], valley = True)
    PKS[count] = PKS_init[0] + i
    count += 1
    
# Definindo um Threshold de acordo com K
PKQ = PKQ.astype(int)
PKS = PKS.astype(int)
KQ = 0.94
KS = 0.995
fdiff = diff.astype(float)
THQ = np.zeros(len(QRS_index[:, 3]))
THS = np.zeros(len(QRS_index[:, 3]))

if min(fdiff) < 0:
    fdiff_positive = fdiff - min(fdiff)
for i, j in zip(PKQ, range(0, len(PKQ))):
    num = fdiff_positive[i]
    THQ[j] = num / KQ
for i, j in zip(PKS, range(0, len(PKS))):
    num = fdiff_positive[i]
    THS[j] = num / KS
if min(fdiff) < 0:
    THQ = THQ + min(fdiff)
    THS = THS + min(fdiff)
    
# Verificando o ponto que cruza o Threshold
QRS_onset = QRS_onset_end (diff, THQ, 25, PKQ, len(QRS_index[:, 2]), onset = True)
QRS_end = QRS_onset_end (diff, THS, 15, PKS, len(QRS_index[:, 2]), onset = False)

QRS_index[:, 0] = QRS_onset - 1
QRS_index[:, 4] = QRS_end - 1

QRS_data = pd.DataFrame(data = QRS_index, columns = ['Q_onset', 'Q_peak', 'R_peak', 'S_peak', 'S_end'])


QRS_times = np.zeros((len(R_peaks) - 1, 4), float)
 #Encontrando os intervalos Q-R
 for i in QRS_index[[1,:], 1]:



# *************** PLOTs **********************

# Plota FFT do sinal original e filtrado
plot_fft()

# Plota o sinal original e filtrado
plot_signals(signal_filtered, signal[:, 1], lb1 = 'filtrado', lb2 = 'raw', mean = False, both = True, fig = 2 )

# Plota o TEO original do sinal e o TEO convoluído com a janela de Bartlett
plot_signals(signal_TEO, signal_TEO_windowed, lb1 = 'TEO ECG', lb2 = 'TEO ECG + Window', both = True, fig = 3)

# Plota o TEO convoluído com a média para detecção de pico e os picos
plt.figure(4)
plot_signals(signal_TEO_windowed, lb1 = 'TEO output convolved with Bartlett Window', fig = 4)
plt.axhline(th, color = 'red', linewidth = 2)
plt.scatter(signal[R_peaks, 0], signal_TEO_windowed[R_peaks], color = 'red')

# Plota o sinal derivado com o QRS detectado
plt.figure(5)
plt.plot(signal[:-1, 0], diff, label = 'Sinal Derivado')
plt.plot(signal[R_peaks, 0], diff[R_peaks], 'ro', color = 'black', label = 'R')
plt.plot(signal[PKb, 0], diff[PKb], 'ro', color = 'green', label = 'S')
plt.plot(signal[PKa, 0], diff[PKa], 'ro', color = 'red', label = 'Q')
plt.legend(loc='upper right')

# Plota sinal filtrado com QRS detectado
plt.figure(6)
plt.clf()
plt.plot(signal[:, 0], signal_filtered, color = 'blue', label = 'ECG Filtrado')
plt.plot(signal[QRS_index[:, 2] , 0], signal_filtered[QRS_index[:, 2]], 'ro', color = 'black', label = 'R')
plt.plot(signal[ QRS_index[:, 3] , 0], signal_filtered[QRS_index[:, 3]], 'ro', color = 'green', label = 'S')
plt.plot(signal[ QRS_index[:, 1] , 0], signal_filtered[QRS_index[:, 1]], 'ro', color = 'red', label = 'Q')
plt.legend(loc='upper right')
plt.xlabel("seconds")
plt.ylabel("mV")

plt.figure(7)
plt.plot(signal[:, 0], signal_TEO_windowed, linewidth = 2, color = 'red')
plt.xlabel("Time (s)")
plt.ylabel("TEO output (mV²)")

# Plota o sinal derivado com o PKQ, PKS, limiares, início e fim do QRS
plt.figure(8)
plt.plot(signal[:-1, 0], diff, label = 'Sinal Derivado')
plt.plot(signal[PKQ, 0], diff[PKQ], 'ro', color = 'black', label = 'PKQ')
plt.plot(signal[PKQ, 0], THQ, 'ro', color = 'red', label = 'THQ')
plt.plot(signal[PKS, 0], diff[PKS], 'ro', color = 'green', label = 'PKS')
plt.plot(signal[PKS, 0], THS, 'ro', color = 'pink', label = 'THS')
plt.plot(signal[QRS_onset, 0], fdiff[QRS_onset], 'ro', color = 'grey', label = 'QRS onset')
plt.plot(signal[QRS_end, 0], fdiff[QRS_end], 'ro', color = 'yellow', label = 'QRS end')
plt.legend(loc='upper right')

# Plota o sinal filtrado com o PKQ, PKS, início e fim do QRS
plt.figure(9)
plt.plot(signal[:, 0], signal_filtered, label = 'Sinal Filtrado')
plt.plot(signal[PKQ-1, 0], signal_filtered[PKQ-1], 'ro', color = 'black', label = 'PKQ')
plt.plot(signal[PKS-1, 0], signal_filtered[PKS-1], 'ro', color = 'green', label = 'PKS')
plt.plot(signal[QRS_onset-1, 0], signal_filtered[QRS_onset-1], 'ro', color = 'grey', label = 'QRS_onset')
plt.plot(signal[QRS_end-1, 0], signal_filtered[QRS_end-1], 'ro', color = 'yellow', label = 'QRS_end')
plt.legend(loc='upper right')

plt.figure(10)
plt.plot(signal[:, 0], signal[:, 1], label = 'Sinal Raw')
plt.plot(signal[PKQ-1, 0], signal[PKQ-1, 1], 'ro', color = 'black', label = 'PKQ')
plt.plot(signal[PKS-1, 0], signal[PKS-1, 1], 'ro', color = 'green', label = 'PKS')
plt.plot(signal[QRS_onset-1, 0], signal[QRS_onset-1, 1], 'ro', color = 'grey', label = 'QRS_onset')
plt.plot(signal[QRS_end-1, 0], signal[QRS_end-1, 1], 'ro', color = 'yellow', label = 'QRS_onset')
plt.legend(loc='upper right')


plt.figure(11)
plt.plot(signal[:, 0], signal_filtered, label = 'Filtered Signal')
plt.plot(signal[QRS_data['Q_onset'], 0], signal_filtered[QRS_data['Q_onset']], 'ro', color = 'yellow', 
         label = 'Q onset')
plt.plot(signal[QRS_data['Q_peak'], 0], signal_filtered[QRS_data['Q_peak']], 'ro', color = 'green', 
         label = 'Q peak')
plt.plot(signal[QRS_data['R_peak'], 0], signal_filtered[QRS_data['R_peak']], 'ro', color = 'black', 
         label = 'R peak')
plt.plot(signal[QRS_data['S_peak'], 0], signal_filtered[QRS_data['S_peak']], 'ro', color = 'red', 
         label = 'S peak')
plt.plot(signal[QRS_data['S_end'], 0], signal_filtered[QRS_data['S_end']], 'ro', color = 'orange', 
         label = 'S end')
plt.legend(loc='upper right')


# Falta:
# [X] Arrumar um jeito melhor de pegar o sinal
# [X] Pré processar o sinal (tá com muito ruído)
# [ ] Feature extraction
#   [X] TEO
#   [X] Fazer convolução com a janela
#   [X] Definir Threshold p/ os picos
#   [X] Detectar as ondas R
#   [X] Detectar os picos Q e S
#   [X] Remover falsas ondas R
#   [X] Detectar início e fim do QRS
#   [ ] Medir o tempo entre Q-R
#   [ ] Medir o tempo entre R-S
#   [ ] Medir o tempo entre Q-S
#   [ ] Medir o tempo entre R-R (anterior)
# [ ] Classificar