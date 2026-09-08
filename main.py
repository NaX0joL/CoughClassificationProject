
from core.data_pipeline_3.pipeline import DataPipeline
from core.data_pipeline_3.source_reader.elderly_cough_audio.source_reader import SourceReader
from core.data_pipeline_3.partitioner import Partitioner



def main():
    pipeline = DataPipeline(
        source_reader=SourceReader(),
        partitioner=Partitioner(),
    )
    pipeline.get_data_module()
    return



if __name__ == "__main__":
    main()
    print("DONE!")
