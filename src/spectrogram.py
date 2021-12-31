import tensorflow as tf
import tensorflow_io as tfio

from src.constants import SR

NFFT = 1024
WINDOW = 1024
STRIDE = 512 # Hop length
MELS = 128
FMAX = SR / 2
TOP_DB = 80

def scale_tensor(tensor: tf.Tensor) -> tf.Tensor:
    minimum = tf.reduce_min(tensor)
    maximum = tf.reduce_max(tensor)
    
    scaled = (tensor - minimum) / (maximum - minimum)
    
    return (scaled - 0.5) * 2

def compute_mel_spectrogram(audio: tf.Tensor) -> tf.Tensor:
    spectrogram = tfio.audio.spectrogram(
        audio, nfft=NFFT, window=WINDOW, stride=STRIDE
    )
    
    mel_spectrogram = tfio.audio.melscale(
        spectrogram, rate=SR, mels=MELS, fmin=0, fmax=FMAX
    )

    dbscale_mel_spectrogram = tfio.audio.dbscale(
        mel_spectrogram, top_db=TOP_DB
    )
    
    return scale_tensor(dbscale_mel_spectrogram)