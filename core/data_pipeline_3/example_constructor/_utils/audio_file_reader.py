from pathlib import Path
import numpy as np
import torchaudio
import torchaudio.functional as audio_functional



class AudioFileReader():
    
    def __init__(self, sampling_rate:int) -> None:
        self.sampling_rate = sampling_rate
        return
    
    def read(self, path:Path) -> np.ndarray:
        
        try:
            waveform, original_sample_rate = torchaudio.load(str(path))
            waveform = waveform.mean(dim=0)     # reduce dim to 1 for mono
        except:
            raise ValueError(f"could not decode audio file: {path}")
        
        if waveform.numel() == 0 or waveform.shape[-1] == 0:
            raise ValueError(f"audio file contains no samples: {path}")
        
        if original_sample_rate != self.sampling_rate:
            waveform = audio_functional.resample(
                waveform,
                orig_freq=original_sample_rate,
                new_freq=self.sampling_rate,
            )

        return waveform.numpy()
