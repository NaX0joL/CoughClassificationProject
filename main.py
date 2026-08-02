from core.data_pipeline.source_reader import ElderlyCoughAudioSourceReader


def main():
    print("Hello from coughclassificationproject!")
    records = ElderlyCoughAudioSourceReader().get_source_data()
    
    for row in records:
        print(row, "\n")
    
    return


if __name__ == "__main__":
    main()
