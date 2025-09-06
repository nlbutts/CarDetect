"""
Example main for RP2040 (Pi Pico 2) that uses GP18-GP21 for SPI to talk to an
Infineon radar module. Adjust register addresses and commands for your specific
radar model.
"""
from machine import Pin, SPI, ADC
import time
import math


def fft_inplace(x):
    # pure-python in-place radix-2 Cooley-Tukey FFT
    n = len(x)
    # bit-reversal permutation
    j = 0
    for i in range(1, n):
        bit = n >> 1
        while j & bit:
            j ^= bit
            bit >>= 1
        j ^= bit
        if i < j:
            x[i], x[j] = x[j], x[i]
    # Cooley-Tukey
    m = 2
    while m <= n:
        theta = -2.0 * math.pi / m
        wm = complex(math.cos(theta), math.sin(theta))
        for k in range(0, n, m):
            w = 1+0j
            half = m // 2
            for j in range(half):
                t = w * x[k + j + half]
                u = x[k + j]
                x[k + j] = u + t
                x[k + j + half] = u - t
                w *= wm
        m <<= 1
    return x

def main():
    # Simple loop: read a status register and blink an LED if data-ready
    led = Pin(25, Pin.OUT)
    debug = Pin(15, Pin.OUT)

    # ADC on GP27
    adc = ADC(Pin(27))

    # Sampling params
    fs = 10000                 # Hz
    N = 2048                   # FFT size (must be power of two)
    interval_us = int(1_000_000 / fs)  # microseconds per sample

    print("Starting ADC capture: fs={} Hz, N={}".format(fs, N))

    try:
        while True:
            led.toggle()
            # capture N samples at fs using ticks_us for timing
            samples = []
            t0 = time.ticks_us()
            for i in range(N):
                target = t0 + i * interval_us
                # wait until target time (ticks_diff handles wrap)
                while time.ticks_diff(time.ticks_us(), target) < 0:
                    pass
                # read 16-bit ADC value
                raw = adc.read_u16()
                # center around zero and normalize to roughly [-1, 1]
                #samples.append((raw - 32768) / 32768.0)
                samples.append(raw)
                debug.toggle()

            # send whole buffer as a single CSV line prefixed with "DATA,"
            # (easier to parse on host and avoids per-line overhead)
            try:
                print("DATA," + ",".join("{:.6f}".format(s) for s in samples))
            except Exception:
                # fall back to simple prints if join fails
                for s in samples:
                    print("{:.6f}".format(s))

            # # prepare complex array for FFT
            # xc = [complex(s, 0) for s in samples]
            # fft_inplace(xc)

            # # compute magnitudes for positive frequencies
            # half = N // 2
            # mags = [abs(xc[k]) / N for k in range(half)]

            # # find peak frequency
            # peak_idx = 0
            # peak_val = 0.0
            # for i, m in enumerate(mags):
            #     if m > peak_val:
            #         peak_val = m
            #         peak_idx = i
            # # convert bin to frequency
            # peak_freq = peak_idx * (fs / N)

            # # toggle LED and print peak
            # led.toggle()
            # print("Peak: {:.1f} Hz, amplitude: {:.4f}".format(peak_freq, peak_val))

            # small pause before next window (optional)
            time.sleep_ms(10)

    except KeyboardInterrupt:
        print("Stopped")

if __name__ == "__main__":
    main()
