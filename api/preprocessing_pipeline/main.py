import pickle

import pandas as pd

from api.preprocessing_pipeline.pipeline import PreprocessingPipeline


def read_data_frame(path: str = '../../data/cars_train.csv') -> pd.DataFrame:
    return pd.read_csv(path)


def save_pipeline(pipeline: PreprocessingPipeline):
    pickle.dump(pipeline, open('../../binaries/preprocessing_pipeline.pickle', 'wb'))


def main():
    df: pd.DataFrame = read_data_frame()
    pipeline: PreprocessingPipeline = PreprocessingPipeline()
    pipeline.fit(df)
    save_pipeline(pipeline)


if __name__ == '__main__':
    main()
