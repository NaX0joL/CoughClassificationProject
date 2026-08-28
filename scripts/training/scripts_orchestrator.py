import os
import sys

print(os.getcwd())
sys.path.append(os.getcwd())

from train_mfcc_mlp import main as mfcc_mlp
from train_mfcc_lenet import main as mfcc_lenet
from train_mfcc_patchtst import main as mfcc_patchtst
from train_mfcc_transformer import main as mfcc_transformer

from train_melband_mlp import main as melband_mlp
from train_melband_lenet import main as melband_lenet
from train_melband_patchtst import main as melband_patchtst
from train_melband_transformer import main as melband_transformer

from train_mfcc_sliding_window_mlp import main as mfcc_sliding_window_mlp
from train_mfcc_sliding_window_lenet import main as mfcc_sliding_window_lenet
from train_mfcc_sliding_window_transformer import main as mfcc_sliding_window_transformer



def main():
    
    script_queue = [
        # mfcc_mlp,
        # mfcc_lenet,
        # mfcc_patchtst,
        # mfcc_transformer,
        
        # melband_mlp,
        # melband_lenet,
        # melband_patchtst,
        # melband_transformer,
        
        # mfcc_sliding_window_mlp,
        # mfcc_sliding_window_lenet,
        mfcc_sliding_window_transformer,
    ]
    
    for script in script_queue:
        script()
        print()
    
    return



if __name__ == "__main__":
    main()
    print("DONE!")