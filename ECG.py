# -*- coding: utf-8 -*-
"""
Created on Tue Oct 23 16:18:20 2018

@author: Filipe Augusto
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.fftpack import fft
import wfdb
from scipy.signal import butter, lfilter, detrend, fftconvolve
from detect_peaks import detect_peaks
from sklearn.utils import shuffle   

#import dspplot


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

def QRS_onset_end (diff, TH, millis, PK, length, fs, onset = True):
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
            else:
                cross = (sub1 >= 0 and sub0 < 0)
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

def detectPK(signal_filtered, diff, QRS_index, fs, R_peaks):
    
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
            if j + 2 >= len(diff):
                PKb[i] = j+1
                break
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

def firstAnn(QRS_index, a_idx):
    
    aux = 0
    for i in range(0, len(QRS_index[:, 2])):
        dif = abs(QRS_index[0, 2] - a_idx[i])
        if(dif <= 5):
            aux = i
            break
    return int(aux)

def NOVOmisDetect(QRS_index, a_idx):
    
    misdecs_idx = []    # Indexes de ondas R (nas anotações) não detectadas corretamente
    discard = []        # Indexes do QRS_index que devem ser descartados
    beat_num = []       # Vetor p/ armazenar o número de cada beat
    sync_beats = []     # Vetor p/ armazenar os beats detectados corretamente
    misdecs = 0.0         # Quantidade de misdetections
    
    
    for i in range(0, len(QRS_index[:, 2])):
        isCorr, corr_idx = correspond(QRS_index, a_idx, i)
        if isCorr:
            sync_beats.append(i)
            beat_num.append(corr_idx)
        else:
            discard.append(i)
    
    
    for i in range(0, len(a_idx)):
        aux = 0
        for j in beat_num:
            if a_idx[i] == a_idx[j]:
                aux += 1
                break
        if aux == 0:
            misdecs_idx.append(i)
    misdecs = len(misdecs_idx) / len(a_idx)
    
    return misdecs_idx, discard, beat_num, sync_beats, misdecs

def misDetect(QRS_index, a_idx):
    
    misdecs = 0         # Quantidade de misdetections
    misdecs_idx = []    # Indexes de ondas R (nas anotações) não detectadas corretamente
    discard = []        # Indexes do QRS_index que devem ser descartados
    beat_num = []       # Vetor p/ armazenar o número de cada beat
    first = firstAnn(QRS_index, a_idx)
    a_idx1 = a_idx[first:]
    idx_a = int(0)
    i = int(0)
    while i < len(QRS_index[:, 2]):
        dif = abs(QRS_index[i, 2] - a_idx1[idx_a])
        if(dif > 6):
            misdecs += 1
            misdecs_idx.append(int(idx_a))
            print([i, idx_a])
            if(correspond(QRS_index, a_idx1, i)):
                i -= 1
            else:
                discard.append(int(i))
        else:
            beat_num.append(int(idx_a))
        if idx_a == (len(a_idx1) - 1):
            break
        idx_a += 1
        i += 1
    return misdecs, misdecs_idx, discard, beat_num
        
def correspond(QRS_index, a_idx1, idx):
    aux = 0
    corr_idx = int(0)
    corr = False
    for i in range(idx, len(a_idx1[:])):
        dif = abs(QRS_index[idx, 2] - a_idx1[i])
        if(dif <= 6):
           # print(i)
            aux += 1
            corr_idx = int(i)
            break
    if(aux > 0):
        corr = True
    return corr, corr_idx

def janelaSinal(signal_TEO_windowed, fs):
    samples = fs * 40
    leng = (len(signal_TEO_windowed) // samples) + 1
    signal_windows = np.zeros((samples, leng), dtype = int)
    
    c = 0
    r = 0
    for i in range(0, len(signal_TEO_windowed)):
        if ((i % samples) == 0) and i >= samples:
            c += 1
            r = 0
        signal_windows[r, c] = i
        r += 1
    if signal_windows[:, -1].all() == False:
        signal_windows = np.delete(signal_windows, len(signal_windows[1, :]) - 1, axis = 1)
    return signal_windows

def ftExtraction(sig_name, returnAll = False):
    
    signal, fs, T, n_samples = read_signal(sig_name, time = 1800)
    ann = wfdb.rdann(sig_name, 'atr', sampto = n_samples)
    a1 = np.array( [ann.symbol, ann.sample])
    a1 = np.transpose(a1)
    ann_del = []
    for i in range(0, len(a1) - 1):
        if ((a1[i, 0] == '~') or (a1[i, 0] == '+') or (a1[i, 0] == 'x') or (a1[i, 0] == '!') or (a1[i, 0] == '|')
        or (a1[i, 0] == '^') or (a1[i, 0] == '=') or (a1[i, 0] == '"') or (a1[i, 0] == '~')):
            ann_del.append(i)
    a1 = np.delete(a1, ann_del, axis = 0)        
    
            
                
    
    a_idx = a1[:, 1].astype(int)
    a_sym = a1[:, 0].astype(str)
    
    
    # Define as frequências de corte do filtro
    lowcut = 0.5
    highcut = 50
    
    # Filtra o sinal
    signal_filtered = butter_bandpass_filter(signal[:,1], lowcut, highcut, fs, order = 2)
    #signal_detrended = detrend(signal_filtered)
    
    # Faz o TEO do sinal ECG
    signal_TEO = np.zeros(shape = len(signal_filtered))
    for i in range(1, len(signal[:,1]) - 1):
        signal_TEO[i] = np.power(signal_filtered[i], 2) - signal_filtered[i+1] * signal_filtered[i-1]
    
    # Cria a janela de Bartlett
    window = np.bartlett(9)
    
    # Convolui o sinal TEO com a janela Bartlett
    signal_TEO_windowed = fftconvolve(signal_TEO, window, mode='same')
    
    # Define o threshold de detecção de pico como média do sinal + desvio padrão
    signal_windows = janelaSinal(signal_TEO_windowed, fs)
    hist = []
    for i in np.transpose(signal_windows):
        hist.append(np.histogram(signal_TEO_windowed[i[0]:i[-1]]))
    
    th = []
    for i in np.transpose(signal_windows):
        th.append(np.mean(signal_TEO_windowed[i[0]:i[-1]]) + 1.5 * np.std(signal_TEO_windowed[i[0]:i[-1]]))
    
    '''
    for i in hist:
        aux = i[1]
        #sub = aux[3] - aux[2]
        #sub = aux[2] + (sub / 2)
        sub = aux[2]
        th.append(sub)
    '''
    
    # Detecta os picos R
    R_peaks_init = []
    aux = 0
    for i, j in zip(np.transpose(signal_windows), range(0, len(th))):
        R_peaks_vect = detect_peaks(signal_TEO_windowed[i[0]:i[-1]], mph = th[j])
        for k in R_peaks_vect:
            R_peaks_init.append(k + aux)
        aux = i[-1]
    
    #R_peaks_init = detect_peaks(signal_TEO_windowed, mph = th) #- 1
    
    # Faz a derivada do sinal
    diff = np.diff(signal_filtered)
    #diff_mean = np.mean(diff) - np.std(diff)
    
    # ******************* EXCLUIR ONDAS R QUE ESTEJAM A MENOS DE 0.5 s UMA DA OUTRA *******************
        
    R_peaks, cont = rmBeats(R_peaks_init, signal_TEO_windowed, signal)
    while cont != 0:
        R_peaks, cont = rmBeats(R_peaks, signal_TEO_windowed, signal)
        
    R_peaks.append(R_peaks_init[len(R_peaks_init) - 1])
            
    
    # Cria uma matriz vazia para armazenar os índices dos QRS e armazena as ondas R
    QRS_index = np.zeros((len(R_peaks), 6), dtype = int)
    QRS_index[:, 2] = R_peaks[:]
    
    #samples = int(fs * 0.150) # Pega o número de amostras em 150 ms
    
    
    PKa, PKb, QRS_index[:, 1], QRS_index[:, 2] = detectPK(signal_filtered, diff, QRS_index, fs, R_peaks)
    QRS_index = np.delete(QRS_index, 0, 0)
    QRS_index = np.delete(QRS_index, len(QRS_index[:, 1]) - 1, 0)
    
       
    # Fazer um loop para achar o primeiro ponto de mínimo antes e depois do Qp e Sp, respectivamente
    #aux = 0,06 ms
    aux = int(fs * 0.5)
    
    N = 5
    cumsum, movav_diff = [0], []
    
    for i, x in enumerate(diff, 1):
        cumsum.append(cumsum[i-1] + x)
        if i>=N:
            moving_ave = (cumsum[i] - cumsum[i-N])/N
            #can do stuff with moving_ave here
            movav_diff.append(moving_ave)
    
    
    PKQ = np.zeros(len(QRS_index[:, 1]))
    count = 0
    for i in QRS_index[:, 1]:
        #aux = 20
        PKQ_init = detect_peaks([4, 4, 4], valley = True)
        i = i - N
        #while PKQ_init.size == 0:
        wdw_min = i - aux
        PKQ_init = detect_peaks(movav_diff[wdw_min:i], valley = True)
        #aux += 1
        
        PKQ[count] = PKQ_init[len(PKQ_init)-1] + i - aux
        count += 1
    
    PKS = np.zeros(len(QRS_index[:, 3]))
    count = 0    
    for i in QRS_index[:, 3]:
        #aux = 40
        i = i-N
        PKS_init = detect_peaks([4, 4, 4], valley = True)
        wdw_max = i + aux
        PKS_init = detect_peaks(movav_diff[i:wdw_max], valley = True)
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
    QRS_onset = QRS_onset_end (diff, THQ, 25, PKQ, len(QRS_index[:, 2]), fs, onset = True)
    QRS_end = QRS_onset_end (diff, THS, 15, PKS, len(QRS_index[:, 2]), fs, onset = False)
    
    QRS_index[:, 0] = QRS_onset - 1
    QRS_index[:, 4] = QRS_end - 1
    
    #misDect, annMisDect_idx, discard, beat_num = misDetect(QRS_index, a_idx)
    misdecs_idx, discard, beat_num, sync_beats, misdecs = NOVOmisDetect(QRS_index, a_idx)
    
    QRS_index = np.delete(QRS_index, discard, 0)
    QRS_index[:, 5] = np.transpose(np.array(beat_num))
    
    QRS_data = pd.DataFrame(data = QRS_index, columns = ['Q_onset', 'Q_peak', 'R_peak', 'S_peak', 'S_end',
                                                        'Beat_nº'])
    
    QRS_times = np.zeros((len(QRS_index[:, 2]), 8), float)
    
    # Encontrando os intervalos Q-R
    for i, j, k in zip(QRS_index[:, 2], QRS_index[:, 1], range(0, len(QRS_index[:, 2]))):
        QRS_times[k, 0] = signal[i, 0] - signal[j, 0]
        
    # Encontrando os intervalos Q-S
    for i, j, k in zip(QRS_index[:, 3], QRS_index[:, 1], range(0, len(QRS_index[:, 2]))):
        QRS_times[k, 1] = signal[i, 0] - signal[j, 0]
        
    # Encontrando os intervalos R-R
    for i, j in zip(QRS_index[:, 2], range(0, len(QRS_index[:, 2]))):
        if(j == 0):
            QRS_times[j, 2] = signal[QRS_index[j+1, 2], 0] - signal[i, 0]
            QRS_times[j, 2] /= QRS_index[j+1, 5] - QRS_index[j, 5]
        else:  
           QRS_times[j, 2] = signal[i, 0] - signal[QRS_index[j-1, 2], 0]
           QRS_times[j, 2] /= QRS_index[j, 5] - QRS_index[j-1, 5]
    
    # Encontrando os intervalos R-S
    for i, j, k in zip(QRS_index[:, 3], QRS_index[:, 2], range(0, len(QRS_index[:, 2]))):
        QRS_times[k, 3] = signal[i, 0] - signal[j, 0]
    
    # Encontrando o intervalo Q_onset - S_end
    for i, j, k in zip(QRS_index[:, 4], QRS_index[:, 0], range(0, len(QRS_index[:, 2]))):
        QRS_times[k, 4] = signal[i, 0] - signal[j, 0]
        
    for i, j, k, l in zip(QRS_index[:, 1], QRS_index[:, 2], QRS_index[:, 3], range(0, len(QRS_index[:, 2]))):
        QRS_times[l, 5] = signal_filtered[i]
        QRS_times[l, 6] = signal_filtered[j]
        QRS_times[l, 7] = signal_filtered[k]
        
    #ann_sym = pd.DataFrame(data = np.transpose(a_sym[1:]), columns = ['Ann'])
    #firstAnnIdx = firstAnn(QRS_index, a_idx)
    #ann_sym = a_sym[firstAnnIdx:-1]
    ann_sym = np.delete(a_sym, misdecs_idx, 0)
    ann_sym = pd.DataFrame(data = np.transpose(ann_sym), columns = ['Ann'])
    ds_beat = pd.DataFrame(data = np.transpose(beat_num), columns = ['Beat_nº'])
    vect = [2, 4, 6, 7]
         
    
    #QRS_timesData = pd.DataFrame(data = QRS_times, columns = ['Q-R', 'Q-S', 'R-R', 'R-S', 'QRS_len', 'Q', 'R', 'S'])
    QRS_timesData = pd.DataFrame(data = QRS_times[:, vect], columns = ['R-R', 'QRS_len', 'R', 'S'])
    QRS_timesData = QRS_timesData.join(ann_sym)
    QRS_timesData = QRS_timesData.join(ds_beat)
    
    if returnAll:
        return (QRS_timesData, signal, signal_filtered, signal_TEO, signal_TEO_windowed, signal_windows, th, R_peaks,
                diff, PKb, PKa, QRS_index, PKQ, THQ, PKS, THS, QRS_onset, QRS_end, QRS_data, movav_diff,
                discard)
    else:
        return QRS_timesData
    
    
    
    
    # ******************************* CLASSIFICATION ***************************
def classifica(QRS_timesData):
    X = QRS_timesData.iloc[:, 0:4].values
    y = QRS_timesData.iloc[:, 4].values
    
    from sklearn.preprocessing import LabelEncoder
    labelencoder_y = LabelEncoder()
    y = labelencoder_y.fit_transform(y.astype(str))
    
    from sklearn.cross_validation import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.7, random_state = 0)
    
    # Feature Scaling
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)
    
    # Fitting the classifier to the Training set
    from sklearn.svm import SVC
    classifier = SVC(kernel = 'rbf')
    classifier.fit(X_train, y_train)
    classifier.score(X_test, y_test)
    
    # Predicting the Test set results
    y_pred = classifier.predict(X_test)
    
    # Making the Confusion Matrix
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_test, y_pred)
    
    from sklearn.metrics import accuracy_score
    acc_s = accuracy_score(y_test, y_pred)
    
    from sklearn.metrics import classification_report
    report = classification_report(y_test, y_pred)
    
    
    return cm, acc_s, report

def classificaEsp(X_train, X_test, y_train, y_test):

    # Feature Scaling
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)
    
    # Fitting the classifier to the Training set
    from sklearn.svm import SVC
    classifier = SVC(kernel = 'rbf')
    classifier.fit(X_train, y_train)
    classifier.score(X_test, y_test)
    
    # Predicting the Test set results
    y_pred = classifier.predict(X_test)
    
    # Making the Confusion Matrix
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_test, y_pred)
    
    from sklearn.metrics import accuracy_score
    acc_s = accuracy_score(y_test, y_pred)
    
    from sklearn.metrics import classification_report
    report = classification_report(y_test, y_pred)
    
    
    return cm, acc_s, report

def rmBeats(R_peaks_init, signal_TEO_windowed, signal):
    R_peaks = []
    i_aux = 0
    while i_aux < (len(R_peaks_init) -1):
        dist = signal[R_peaks_init[i_aux+1], 0] - signal[R_peaks_init[i_aux], 0]
        if(dist >= 0.35):
            R_peaks.append(R_peaks_init[i_aux])
        else:
            sub = signal_TEO_windowed[R_peaks_init[i_aux]] - signal_TEO_windowed[R_peaks_init[i_aux+1]]
            if sub > 0:
                R_peaks.append(R_peaks_init[i_aux])   
            else:
                R_peaks.append(R_peaks_init[i_aux+1])
            i_aux += 1  
        i_aux += 1        
    i_aux = 0
    cont = 0
    while i_aux < (len(R_peaks) - 1):
        dist = signal[R_peaks[i_aux+1], 0] - signal[R_peaks[i_aux], 0]
        if dist <= 0.35:
            cont += 1
        i_aux += 1
    
    return R_peaks, cont

def makeSet(df, length, toLength = True):
    if toLength:
        newSet = pd.DataFrame(columns = ['R-R', 'QRS_len', 'R', 'S', 'Ann', 'Beat_nº'])
        newSet = newSet.append(df.iloc[:length, :])
    else:
        newSet = pd.DataFrame(columns = ['R-R', 'QRS_len', 'R', 'S', 'Ann', 'Beat_nº'])
        newSet = newSet.append(df.iloc[length:, :])
    
    return newSet

def splitSet(df1, df2, df3, X_set = True):
    
    newSet = pd.DataFrame()
    
    if X_set:
        newSet = newSet.append(df1.iloc[:, 0:4])
        newSet = newSet.append(df2.iloc[:, 0:4])
        newSet = newSet.append(df3.iloc[:, 0:4])
        newSet = np.array(newSet.values)
    else:
        newSet = np.array(df1.iloc[:, 4].values)
        newSet = np.concatenate([newSet, df2.iloc[:, 4].values])
        newSet = np.concatenate([newSet, df3.iloc[:, 4].values])
        from sklearn.preprocessing import LabelEncoder
        labelencoder_y = LabelEncoder()
        newSet = labelencoder_y.fit_transform(newSet.astype(str))
    
    return newSet


#  ****************************** FAZER UM VETOR DE POSIÇÕES ÚTEIS DE ANOTAÇÕES *************************
    
N_beats = pd.DataFrame(columns = ['R-R', 'QRS_len', 'R', 'S', 'Ann', 'Beat_nº'])
R_beats = pd.DataFrame(columns = ['R-R', 'QRS_len', 'R', 'S', 'Ann', 'Beat_nº'])
L_beats = pd.DataFrame(columns = ['R-R', 'QRS_len', 'R', 'S', 'Ann', 'Beat_nº'])


names = ['111', '214', '118', '212']   
# Lê o sinal e as anotações
for sig_name in names:
    QRS_timesData1 = ftExtraction(sig_name)
    for i in range(0, len(QRS_timesData1.iloc[:, 1])):
        if QRS_timesData1.iloc[i, 4] == 'N':
            N_beats = N_beats.append(QRS_timesData1.iloc[i, :])
        if QRS_timesData1.iloc[i, 4] == 'R':
            R_beats = R_beats.append(QRS_timesData1.iloc[i, :])
        if QRS_timesData1.iloc[i, 4] == 'L':
            L_beats = L_beats.append(QRS_timesData1.iloc[i, :])
            
    
N_beats = shuffle(N_beats).reset_index(drop = True)
L_beats = shuffle(L_beats).reset_index(drop = True)
R_beats = shuffle(R_beats).reset_index(drop = True)

train_N = makeSet(N_beats, 150)
train_R = makeSet(R_beats, 200)
train_L = makeSet(L_beats, 200)
X_trainSet = splitSet(train_N, train_R, train_L, X_set = True)
y_trainSet = splitSet(train_N, train_R, train_L, X_set = False)

test_N = makeSet(N_beats, 150, toLength = False)
test_R = makeSet(R_beats, 200, toLength = False)
test_L = makeSet(L_beats, 200, toLength = False)
X_testSet = splitSet(test_N, test_R, test_L, X_set = True)
y_testSet = splitSet(test_N, test_R, test_L, X_set = False)

cm, acc_s, report = classificaEsp(X_trainSet, X_testSet, y_trainSet, y_testSet)
            
all_beats = pd.DataFrame()
all_beats = all_beats.append(N_beats)
all_beats = all_beats.append(R_beats)
all_beats = all_beats.append(L_beats)
df = shuffle(all_beats).reset_index(drop = True)    
cm, acc, rep = classifica(QRS_timesData1)


sig_name = '101'
(QRS_timesData, signal, signal_filtered, signal_TEO, signal_TEO_windowed, signal_windows, th,
 R_peaks, diff, PKb, PKa, QRS_index, PKQ, THQ, PKS, THS, QRS_onset, QRS_end, QRS_data, movav_diff, 
 discard) = ftExtraction(sig_name, returnAll = True)

signal, fs, T, n_samples = read_signal(sig_name, time = 1800)
ann = wfdb.rdann(sig_name, 'atr', sampto = n_samples)
a = ann.symbol
a1 = np.array( [ann.symbol, ann.sample])
a1 = np.transpose(a1)
ann_del = []
for i in range(0, len(a1) - 1):
    if ((a1[i, 0] == '~') or (a1[i, 0] == '+') or (a1[i, 0] == 'x') or (a1[i, 0] == '!') or (a1[i, 0] == '|')
    or (a1[i, 0] == '^') or (a1[i, 0] == '=') or (a1[i, 0] == '"') or (a1[i, 0] == '~')):
        ann_del.append(i)
a1 = np.delete(a1, ann_del, axis = 0)        

        
            

a_idx = a1[:, 1].astype(int)
a_sym = a1[:, 0].astype(str)


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
signal_windows = janelaSinal(signal_TEO_windowed, fs)
hist = []
for i in np.transpose(signal_windows):
    hist.append(np.histogram(signal_TEO_windowed[i[0]:i[-1]]))

th = []
for i in np.transpose(signal_windows):
    th.append(np.mean(signal_TEO_windowed[i[0]:i[-1]]) + 1.5 * np.std(signal_TEO_windowed[i[0]:i[-1]]))

'''
for i in hist:
    aux = i[1]
    #sub = aux[3] - aux[2]
    #sub = aux[2] + (sub / 2)
    sub = aux[2]
    th.append(sub)
'''

# Detecta os picos R
R_peaks_init = []
aux = 0
for i, j in zip(np.transpose(signal_windows), range(0, len(th))):
    R_peaks_vect = detect_peaks(signal_TEO_windowed[i[0]:i[-1]], mph = th[j])
    for k in R_peaks_vect:
        R_peaks_init.append(k + aux)
    aux = i[-1]

#R_peaks_init = detect_peaks(signal_TEO_windowed, mph = th) #- 1

# Faz a derivada do sinal
diff = np.diff(signal_filtered)
diff_mean = np.mean(diff) - np.std(diff)

# ******************* EXCLUIR ONDAS R QUE ESTEJAM A MENOS DE 0.5 s UMA DA OUTRA *******************
    
R_peaks, cont = rmBeats(R_peaks_init, signal_TEO_windowed, signal)
while cont != 0:
    R_peaks, cont = rmBeats(R_peaks, signal_TEO_windowed, signal)
    
R_peaks.append(R_peaks_init[len(R_peaks_init) - 1])
        

# Cria uma matriz vazia para armazenar os índices dos QRS e armazena as ondas R
QRS_index = np.zeros((len(R_peaks), 6), dtype = int)
QRS_index[:, 2] = R_peaks[:]

samples = int(fs * 0.150) # Pega o número de amostras em 150 ms


PKa, PKb, QRS_index[:, 1], QRS_index[:, 2] = detectPK(signal_filtered, diff, QRS_index, fs, R_peaks)
QRS_index = np.delete(QRS_index, 0, 0)
QRS_index = np.delete(QRS_index, len(QRS_index[:, 1]) - 1, 0)

   
# Fazer um loop para achar o primeiro ponto de mínimo antes e depois do Qp e Sp, respectivamente
#aux = 0,06 ms
aux = int(fs * 0.5)

N = 5
cumsum, movav_diff = [0], []

for i, x in enumerate(diff, 1):
    cumsum.append(cumsum[i-1] + x)
    if i>=N:
        moving_ave = (cumsum[i] - cumsum[i-N])/N
        #can do stuff with moving_ave here
        movav_diff.append(moving_ave)


PKQ = np.zeros(len(QRS_index[:, 1]))
count = 0
for i in QRS_index[:, 1]:
    #aux = 20
    PKQ_init = detect_peaks([4, 4, 4], valley = True)
    i = i - N
    #while PKQ_init.size == 0:
    wdw_min = i - aux
    PKQ_init = detect_peaks(movav_diff[wdw_min:i], valley = True)
    #aux += 1
    
    PKQ[count] = PKQ_init[len(PKQ_init)-1] + i - aux
    count += 1

PKS = np.zeros(len(QRS_index[:, 3]))
count = 0    
for i in QRS_index[:, 3]:
    #aux = 40
    i = i-N
    PKS_init = detect_peaks([4, 4, 4], valley = True)
    wdw_max = i + aux
    PKS_init = detect_peaks(movav_diff[i:wdw_max], valley = True)
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
QRS_onset = QRS_onset_end (diff, THQ, 25, PKQ, len(QRS_index[:, 2]), fs, onset = True)
QRS_end = QRS_onset_end (diff, THS, 15, PKS, len(QRS_index[:, 2]), fs, onset = False)

QRS_index[:, 0] = QRS_onset - 1
QRS_index[:, 4] = QRS_end - 1

#misDect, annMisDect_idx, discard, beat_num = misDetect(QRS_index, a_idx)
misdecs_idx, discard, beat_num, sync_beats, misdecs = NOVOmisDetect(QRS_index, a_idx)

QRS_index = np.delete(QRS_index, discard, 0)
QRS_index[:, 5] = np.transpose(np.array(beat_num))

QRS_data = pd.DataFrame(data = QRS_index, columns = ['Q_onset', 'Q_peak', 'R_peak', 'S_peak', 'S_end',
                                                     'Beat_nº'])

QRS_times = np.zeros((len(QRS_index[:, 2]), 8), float)

# Encontrando os intervalos Q-R
for i, j, k in zip(QRS_index[:, 2], QRS_index[:, 1], range(0, len(QRS_index[:, 2]))):
    QRS_times[k, 0] = signal[i, 0] - signal[j, 0]
    
# Encontrando os intervalos Q-S
for i, j, k in zip(QRS_index[:, 3], QRS_index[:, 1], range(0, len(QRS_index[:, 2]))):
    QRS_times[k, 1] = signal[i, 0] - signal[j, 0]
    
# Encontrando os intervalos R-R
for i, j in zip(QRS_index[:, 2], range(0, len(QRS_index[:, 2]))):
    if(j == 0):
        QRS_times[j, 2] = signal[QRS_index[j+1, 2], 0] - signal[i, 0]
        QRS_times[j, 2] /= QRS_index[j+1, 5] - QRS_index[j, 5]
    else:  
       QRS_times[j, 2] = signal[i, 0] - signal[QRS_index[j-1, 2], 0]
       QRS_times[j, 2] /= QRS_index[j, 5] - QRS_index[j-1, 5]

# Encontrando os intervalos R-S
for i, j, k in zip(QRS_index[:, 3], QRS_index[:, 2], range(0, len(QRS_index[:, 2]))):
    QRS_times[k, 3] = signal[i, 0] - signal[j, 0]

# Encontrando o intervalo Q_onset - S_end
for i, j, k in zip(QRS_index[:, 4], QRS_index[:, 0], range(0, len(QRS_index[:, 2]))):
    QRS_times[k, 4] = signal[i, 0] - signal[j, 0]
    
for i, j, k, l in zip(QRS_index[:, 1], QRS_index[:, 2], QRS_index[:, 3], range(0, len(QRS_index[:, 2]))):
    QRS_times[l, 5] = signal_filtered[i]
    QRS_times[l, 6] = signal_filtered[j]
    QRS_times[l, 7] = signal_filtered[k]
    
#ann_sym = pd.DataFrame(data = np.transpose(a_sym[1:]), columns = ['Ann'])
firstAnnIdx = firstAnn(QRS_index, a_idx)
ann_sym = a_sym[firstAnnIdx:-1]
ann_sym = np.delete(ann_sym, annMisDect_idx, 0)
ann_sym = pd.DataFrame(data = np.transpose(ann_sym), columns = ['Ann'])
ds_beat = pd.DataFrame(data = np.transpose(beat_num), columns = ['Beat_nº'])
vect = [2, 4, 6, 7]
     

#QRS_timesData = pd.DataFrame(data = QRS_times, columns = ['Q-R', 'Q-S', 'R-R', 'R-S', 'QRS_len', 'Q', 'R', 'S'])
QRS_timesData = pd.DataFrame(data = QRS_times[:, vect], columns = ['R-R', 'QRS_len', 'R', 'S'])
QRS_timesData = QRS_timesData.join(ann_sym)
QRS_timesData = QRS_timesData.join(ds_beat)




# ******************************* CLASSIFICATION ***************************

X = QRS_timesData.iloc[:, 0:4].values
y = QRS_timesData.iloc[:, 4].values

from sklearn.preprocessing import LabelEncoder
labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(y.astype(str))

from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Fitting the classifier to the Training set
from sklearn.svm import SVC
classifier = SVC(kernel = 'rbf')
classifier.fit(X_train, y_train)
classifier.score(X_test, y_test)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

from sklearn.metrics import accuracy_score
acc_s = accuracy_score(y_test, y_pred)

from sklearn.metrics import classification_report
report = classification_report(y_test, y_pred)
# *************** PLOTs **********************



# Plota FFT do sinal original e filtrado
plot_fft()

# Plota o sinal original e filtrado
plot_signals(signal_filtered, signal[:, 1], lb1 = 'filtrado', lb2 = 'raw', mean = False, both = True, 
             fig = 2 )

# Plota o TEO original do sinal e o TEO convoluído com a janela de Bartlett
plot_signals(signal_TEO, signal_TEO_windowed, lb1 = 'TEO ECG', lb2 = 'TEO ECG + Window', both = True, fig = 3)

# Plota o TEO convoluído com a média para detecção de pico e os picos
plt.figure(4)
plot_signals(signal_TEO_windowed, lb1 = 'TEO output convolved with Bartlett Window', fig = 4)
for i, j in zip(np.transpose(signal_windows), range(0, len(th))):
    plt.plot(signal[i[0]:i[-1], 0], np.repeat(th[j], len(i) - 1), color = 'red')
#plt.axhline(th, color = 'red', linewidth = 2)
plt.scatter(signal[R_peaks, 0], signal_TEO_windowed[R_peaks], color = 'black')

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
#plt.plot(signal[ QRS_index[:, 3] , 0], signal_filtered[QRS_index[:, 3]], 'ro', color = 'green', label = 'S')
#plt.plot(signal[ QRS_index[:, 1] , 0], signal_filtered[QRS_index[:, 1]], 'ro', color = 'red', label = 'Q')
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


plt.figure(8)
plt.plot(signal[:-1, 0], diff, label = 'Sinal Derivado')
plt.plot(signal[:-5, 0], movav_diff, label = 'Sinal Derivado')


plt.plot(signal[QRS_index[discard, 2], 0], signal_filtered[QRS_index[discard, 2]], 'ro', color = 'yellow', 
         label = 'detecções descartadas')
plt.legend(loc='upper right')



# Falta:
# [X] Arrumar um jeito melhor de pegar o sinal
# [X] Pré processar o sinal
# [ ] Feature extraction
#   [X] TEO
#   [X] Fazer convolução com a janela
#   [X] Janelar o sinal TEO
#   [X] Definir Threshold p/ os picos p/ cada janela
#   [X] Detectar as ondas R
#   [X] Detectar os picos Q e S
#   [X] Remover falsas ondas R
#   [X] Detectar início e fim do QRS
#   [X] Medir o tempo entre Q-R
#   [X] Medir o tempo entre R-S
#   [X] Medir o tempo entre Q-S
#   [X] Medir o tempo entre R-R (anterior)
#   [X] Medir o tempo de início e fim do QRS
#   [X] Verificar quantos e quais ondas R não foram classificadas corretamente
# [X] Pegar as anotações do sinal
# [ ] Separar os beats por tipo
# [ ] Classificar (por tipo)
# [ ] Analisar resultados por tipo
# [ ] Calcular acurácia
# [ ] Calcular sensibilidade
# [ ] Calcular precisão