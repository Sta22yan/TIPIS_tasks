import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, ifft, fftfreq
from scipy.signal import butter, filtfilt, square, hilbert

class Program:
    def __init__(self):
        self.samplingRate = 1000
        self.duration = 2
        self.time = np.linspace(0, self.duration, self.samplingRate, endpoint=False)

        self.carrierFreq = 10
        self.modulationFreq = 1

        self.minCutoff = 17
        self.maxCutoff = 26

    def getHarmonicSignal(self):
        return np.sin(2 * np.pi * self.carrierFreq * self.time)

    def getModulatingSignal(self):
        return np.where(square(2 * np.pi * self.modulationFreq * self.time) > 0, 1, 0)
    
    def getAmplitudeModulationSignal(self, harmonicSignal, modulatingSignal):
        return (1 + modulatingSignal) * harmonicSignal

    def getFrequencyModulationSignal(self, modulatingSignal):
        return np.sin(2 * np.pi * self.carrierFreq * self.time + 50 * np.pi * np.cumsum(modulatingSignal) / self.samplingRate)

    def getPhaseModulationSignal(self, modulatingSignal):
        return np.sin(2 * np.pi * self.carrierFreq * self.time + 0.5 * np.pi * modulatingSignal)

    def getSpectr(self, signal):
        spectr = fft(signal)
        freq = fftfreq(len(signal), 1 / self.samplingRate)
        return freq, np.abs(spectr)

    def getSynthesizedSignal(self, spectr, freq):
        filtered = np.where((freq > self.minCutoff) & (freq < self.maxCutoff), spectr, 0)
        return ifft(filtered).real

    def getFilteredSignal(self, signal):
        nyq = 0.5 * self.samplingRate
        normalCutoff = self.maxCutoff / nyq
        b, a = butter(5, normalCutoff, btype='low', analog=False)
        return filtfilt(b, a, signal)
        
    def getComparedSignal(self, filteredSignal):
        envelope = np.abs(hilbert(filteredSignal))

        meanEnvelope = np.mean(envelope)

        return (envelope > meanEnvelope).astype(float)

    def showPlots(self, harmonicSignal, modulatingSignal, signalAM, freqAM, spectrAM, signalFM, freqFM, spectrFM, signalPM, freqPM, spectrPM, synthesizedSignal, filteredSignal):
        plt.figure(figsize=(10, 5))
        plt.plot(self.time, harmonicSignal, label="Гармонический сигнал")
        plt.plot(self.time, modulatingSignal, label="Модулирующий сигнал")
        plt.xlabel("Время (с)")
        plt.ylabel("Амплитуда")
        plt.legend(loc='right', bbox_to_anchor=(1.1, 1.1), ncol=1)
        plt.grid()
        plt.show()

        plt.figure(figsize=(10, 5))
        plt.plot(self.time, signalAM, label="Амплитудная модуляция")
        plt.xlabel("Время (с)")
        plt.ylabel("Амплитуда")
        plt.legend(loc='right', bbox_to_anchor=(1.1, 1.1), ncol=1)
        plt.grid()
        plt.show()

        plt.figure(figsize=(10, 5))
        plt.plot(freqAM, spectrAM, label="Спектр амплитудной модуляции")
        plt.xlim(0, 100)
        plt.xlabel("Частота (Гц)")
        plt.ylabel("Амплитуда")
        plt.legend(loc='right', bbox_to_anchor=(1.1, 1.1), ncol=1)
        plt.grid()
        plt.show()

        plt.figure(figsize=(10, 5))
        plt.plot(self.time, signalFM, label="Частотная модуляция")
        plt.xlabel("Время (с)")
        plt.ylabel("Амплитуда")
        plt.legend(loc='right', bbox_to_anchor=(1.1, 1.1), ncol=1)
        plt.grid()
        plt.show()

        plt.figure(figsize=(10, 5))
        plt.plot(freqFM, spectrFM, label="Спектр частотной модуляции")
        plt.xlim(0, 100)
        plt.xlabel("Частота (Гц)")
        plt.ylabel("Амплитуда")
        plt.legend(loc='right', bbox_to_anchor=(1.1, 1.1), ncol=1)
        plt.grid()
        plt.show()

        plt.figure(figsize=(10, 5))
        plt.plot(self.time, signalPM, label="Фазовая модуляция")
        plt.xlabel("Время (с)")
        plt.ylabel("Амплитуда")
        plt.legend(loc='right', bbox_to_anchor=(1.1, 1.1), ncol=1)
        plt.grid()
        plt.show()

        plt.figure(figsize=(10, 5))
        plt.plot(freqPM, spectrPM, label="Спектр фазовой модуляции")
        plt.xlim(0, 100)
        plt.xlabel("Частота (Гц)")
        plt.ylabel("Амплитуда")
        plt.legend(loc='right', bbox_to_anchor=(1.1, 1.1), ncol=1)
        plt.grid()
        plt.show()

        plt.figure(figsize=(10, 5))
        plt.plot(self.time, synthesizedSignal, label="Синтезированный сигнал")
        plt.plot(self.time, filteredSignal, label="Отфильтрованный сигнал")
        plt.xlabel("Время (с)")
        plt.ylabel("Амплитуда")
        plt.legend(loc='right', bbox_to_anchor=(1.1, 1.1), ncol=1)
        plt.grid()
        plt.show()

        plt.figure(figsize=(10, 5))
        plt.plot(self.time, self.getComparedSignal(filteredSignal), label="Итоговый сигнал")
        plt.xlabel("Время (с)")
        plt.ylabel("Амплитуда")
        plt.legend(loc='right', bbox_to_anchor=(1.1, 1.1), ncol=1)
        plt.grid()
        plt.show()
        
    def start(self):
        harmonicSignal = self.getHarmonicSignal()
        modulatingSignal = self.getModulatingSignal()
        
        signalAM = self.getAmplitudeModulationSignal(harmonicSignal, modulatingSignal)
        signalFM = self.getFrequencyModulationSignal(modulatingSignal)
        signalPM = self.getPhaseModulationSignal(modulatingSignal)
        
        freqAM, spectrAM = self.getSpectr(signalAM)
        freqFM, spectrFM = self.getSpectr(signalFM)
        freqPM, spectrPM = self.getSpectr(signalPM)
        
        synthesizedSignal = self.getSynthesizedSignal(spectrAM, freqAM)
        filteredSignal = self.getFilteredSignal(synthesizedSignal)

        self.showPlots(harmonicSignal, modulatingSignal, signalAM, freqAM, spectrAM, signalFM, freqFM, spectrFM, signalPM, freqPM, spectrPM, synthesizedSignal, filteredSignal)

if __name__ == "__main__":
    program = Program()
    program.start()

