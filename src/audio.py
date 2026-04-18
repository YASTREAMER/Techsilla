import pyaudio
import wave
from scipy.io import wavfile
import noisereduce as nr
import numpy as np


def Recording(
    CHUNK=1024 * 4, FORMAT=pyaudio.paInt16, CHANNELS=1, RATE=44100, RECORD_SECONDS=15
):
    WAVE_OUTPUT_FILENAME = "Audio/test.wav"
    p = pyaudio.PyAudio()

    stream = p.open(
        format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK
    )

    print("* recording")

    frames = []

    for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
        data = stream.read(CHUNK)
        frames.append(data)

    print("* done recording")

    stream.stop_stream()
    stream.close()
    p.terminate()

    wf = wave.open(WAVE_OUTPUT_FILENAME, "wb")
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(p.get_sample_size(FORMAT))
    wf.setframerate(RATE)
    wf.writeframes(b"".join(frames))
    wf.close()

    Denoising(WAVE_OUTPUT_FILENAME)


def Denoising(AudioFile):
    rate, data = wavfile.read(AudioFile)

    # if stereo, convert to mono by averaging channels
    if data.ndim > 1:
        data = np.mean(data, axis=1)

    # perform noise reduction
    reduced_noise = nr.reduce_noise(y=data, sr=rate)

    # save output
    wavfile.write("Audio/ouput.wav", rate, reduced_noise.astype(np.int16))
