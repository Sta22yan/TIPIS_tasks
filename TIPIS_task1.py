import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import fft, fftfreq

class Program:
    def __init__(self):
        self.samplingRate = 1000
        self.duration = 2
        self.time = np.linspace(0, self.duration, self.samplingRate * self.duration, endpoint=False)

    def getHarmonicSignal(self, freq):
        return np.sin(2 * np.pi * freq * self.time)

    def getDigitalSignal(self, freq, amplitude=1):
        return amplitude * (np.sign(np.sin(2 * np.pi * freq * self.time)) + 1) / 2

    def getSpectr(self, signal):
        yf = fft(signal)
        xf = fftfreq(len(signal), 1 / self.samplingRate)
        return xf[:len(signal) // 2], np.abs(yf[:len(signal) // 2])

    def showPlots(self, freq, harmonicSignal, digitalSignal, xfHarmonic, yfHarmonic, xfDigital, yfDigital, num):
        fig = plt.figure(num)
        
        plt.subplot(2, 2, 1)
        plt.plot(self.time, harmonicSignal)
        plt.title(f"Гармонический сигнал {freq} Гц")
        plt.xlabel("Время (сек)")
        plt.ylabel("Амплитуда")
        plt.grid(True)

        plt.subplot(2, 2, 2)
        plt.plot(xfHarmonic, yfHarmonic)
        plt.title(f"Спектр гармонического сигнала {freq} Гц")
        plt.xlabel("Частота (Гц)")
        plt.ylabel("Амплитуда")
        plt.xlim(0, 50)
        plt.grid(True)

        plt.subplot(2, 2, 3)
        plt.plot(self.time, digitalSignal)
        plt.title(f"Цифровой сигнал {freq} Гц")
        plt.xlabel("Время (сек)")
        plt.ylabel("Амплитуда")
        plt.grid(True)

        plt.subplot(2, 2, 4)
        plt.plot(xfDigital, yfDigital)
        plt.title(f"Спектр цифрового сигнала {freq} Гц")
        plt.xlabel("Частота (Гц)")
        plt.ylabel("Амплитуда")
        plt.grid(True)

        plt.tight_layout()

    def start(self, freq):
        for num, fq in enumerate(freq):
            harmonicSignal = self.getHarmonicSignal(fq)
            digitalSignal = self.getDigitalSignal(fq)

            xfHarmonic, yfHarmonic = self.getSpectr(harmonicSignal)
            xfDigital, yfDigital = self.getSpectr(digitalSignal)
            yfHarmonic[0] = 0
            yfDigital[0] = 0

            self.showPlots(fq, harmonicSignal, digitalSignal, xfHarmonic, yfHarmonic, xfDigital, yfDigital, num)

        plt.show()


if __name__ == "__main__":
    freq = [1, 2, 4, 8]
    program = Program()
    program.start(freq)
