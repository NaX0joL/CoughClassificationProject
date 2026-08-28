import os
import sys

print(os.getcwd())
sys.path.append(os.getcwd())

from train_mfcc_annotated_segments_mlp import main as mfcc_annotated_segments_mlp
from train_mfcc_annotated_segments_lenet import main as mfcc_annotated_segments_lenet
from train_mfcc_annotated_segments_patchtst import main as mfcc_annotated_segments_patchtst
from train_mfcc_annotated_segments_transformer import main as mfcc_annotated_segments_transformer

from train_log_mel_spectrogram_annotated_segments_mlp import main as log_mel_spectrogram_annotated_segments_mlp
from train_log_mel_spectrogram_annotated_segments_lenet import main as log_mel_spectrogram_annotated_segments_lenet
from train_log_mel_spectrogram_annotated_segments_patchtst import main as log_mel_spectrogram_annotated_segments_patchtst
from train_log_mel_spectrogram_annotated_segments_transformer import main as log_mel_spectrogram_annotated_segments_transformer

from train_mfcc_sliding_windows_mlp import main as mfcc_sliding_windows_mlp
from train_mfcc_sliding_windows_lenet import main as mfcc_sliding_windows_lenet
from train_mfcc_sliding_windows_transformer import main as mfcc_sliding_windows_transformer



def main():
    
    script_queue = [
        # mfcc_annotated_segments_mlp,
        # mfcc_annotated_segments_lenet,
        # mfcc_annotated_segments_patchtst,
        # mfcc_annotated_segments_transformer,
        
        # log_mel_spectrogram_annotated_segments_mlp,
        # log_mel_spectrogram_annotated_segments_lenet,
        # log_mel_spectrogram_annotated_segments_patchtst,
        # log_mel_spectrogram_annotated_segments_transformer,
        
        # mfcc_sliding_windows_mlp,
        # mfcc_sliding_windows_lenet,
        mfcc_sliding_windows_transformer,
    ]
    
    for script in script_queue:
        script()
        print()
    
    return



if __name__ == "__main__":
    main()
    print("DONE!")
