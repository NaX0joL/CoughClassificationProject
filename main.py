from core.data_pipeline.source_reader import ElderlyCoughAudioSourceReader
from core.data_pipeline.preprocessing.segmentation import CoughSegmenter
from core.data_pipeline.preprocessing.transform import MFCC
from core.data_pipeline.preprocessing.padding import Padder



def main():
    print("Hello from coughclassificationproject!")
    
    source_series = ElderlyCoughAudioSourceReader().get_source_series() 
    examples = CoughSegmenter(kept_metadata_key=["patient_id", "cough_audio"]).segment(source_series)
    mfcc_examples = MFCC().transform(examples)
    padded_examples = Padder(target_length=820, padding_type="random", random_seed=42).pad(mfcc_examples)
    
    max_len = -1
    for things in mfcc_examples:
        # print(things.value.shape)
        # print(things.label)
        # print(things.metadata)
        
        max_len = max(max_len, things.value.shape[0])
    print(max_len)
    
    return



if __name__ == "__main__":
    main()
