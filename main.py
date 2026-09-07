
from core.data_pipeline_3.source_reader.elderly_cough_audio.source_reader import SourceReader 



def main():
    data = SourceReader().get_source_data()
    print(len(data))
    return



if __name__ == "__main__":
    main()
    print("DONE!")
