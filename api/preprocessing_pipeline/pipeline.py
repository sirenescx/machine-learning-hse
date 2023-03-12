import numpy as np
import pandas as pd
from category_encoders.cat_boost import CatBoostEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder, PolynomialFeatures


class PreprocessingPipeline:
    FEATURES_FOR_CATBOOST_ENCODING: list[str] = ['fuel', 'owner', 'seller_type', 'transmission']
    FEATURES_FOR_ONE_HOT_ENCODING: list[str] = ['seats']
    NUMERIC_FEATURES: list[str] = ['year', 'km_driven', 'mileage', 'engine', 'max_power']
    TARGET: str = 'selling_price'

    preprocessing_pipeline: Pipeline
    medians: list = None

    def __init__(self):
        transformer = ColumnTransformer(
            [
                (
                    'cbe',
                    CatBoostEncoder(),
                    self.FEATURES_FOR_CATBOOST_ENCODING
                ),
                (
                    'ohe',
                    OneHotEncoder(drop='first', sparse_output=False),
                    self.FEATURES_FOR_ONE_HOT_ENCODING
                ),
                (
                    'scaler',
                    StandardScaler(),
                    self.NUMERIC_FEATURES
                ),
            ],
            remainder='passthrough',
            verbose_feature_names_out=False
        ).set_output(transform='pandas')
        extra_scaler = ColumnTransformer(
            [
                (
                    'scaler',
                    StandardScaler(),
                    self.FEATURES_FOR_CATBOOST_ENCODING
                ),
            ],
            remainder='passthrough',
            verbose_feature_names_out=False
        ).set_output(transform='pandas')

        self.preprocessing_pipeline: Pipeline = Pipeline([
            ('transformer', transformer),
            ('extra_scaler', extra_scaler),
            ('polynomial_features', PolynomialFeatures(2))
        ]).set_output(transform='pandas')

    def drop_duplicates(self, df: pd.DataFrame, has_target: True) -> pd.DataFrame:
        if has_target:
            df = df.iloc[df.drop(columns=[self.TARGET]).drop_duplicates(keep='first').index]
        else:
            df = df.iloc[df.drop_duplicates(keep='first').index]
        df.reset_index(drop=True)
        return df

    def remove_units_from_mileage(self, measurement: object) -> float:
        if type(measurement) == str:
            measurement = str(measurement)
            measurement_split: list[str] = measurement.split()
            if len(measurement_split) > 1:
                value: float = float(measurement_split[0])
                units: str = measurement_split[1]
                if units == 'km/kg':
                    return value * 1.40
                return value
        return float(measurement)

    def remove_units_from_engine(self, measurement: object) -> float:
        if type(measurement) == str:
            measurement = str(measurement)
            measurement_split: list[str] = measurement.split()
            if len(measurement_split) > 1:
                return float(measurement_split[0])
        return float(measurement)

    def remove_units_from_max_power(self, measurement: object) -> float:
        if type(measurement) == str:
            measurement = str(measurement)
            measurement_split: list[str] = measurement.split()
            if len(measurement_split) > 1:
                return float(measurement_split[0])
            # to fix ' bhp' case
            if measurement.isnumeric():
                return float(measurement)
            return np.NaN
        return float(measurement)

    def clear_units(self, df: pd.DataFrame) -> pd.DataFrame:
        df.mileage = df.mileage.apply(self.remove_units_from_mileage)
        df.engine = df.engine.apply(self.remove_units_from_engine)
        df.max_power = df.max_power.apply(self.remove_units_from_max_power)
        df = df.drop(columns=['torque', 'name'])
        return df

    def fill_na(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.medians is None:
            self.medians = df.median()
        df = df.fillna(self.medians)
        return df

    def cast_to_int(self, df: pd.DataFrame) -> pd.DataFrame:
        df.engine = df.engine.astype(int)
        df.seats = df.seats.astype(int)
        return df

    def split_to_X_and_y(self, df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        has_target: bool = self.TARGET in list(df.columns)
        df = self.drop_duplicates(df.copy(), has_target)
        df = self.clear_units(df.copy())
        df = self.fill_na(df.copy())
        df = self.cast_to_int(df.copy())
        if has_target:
            X: pd.DataFrame = df.drop(columns=self.TARGET)
            y: pd.DataFrame = df.selling_price
        else:
            return df, None
        return X, y

    def transform(self, df: pd.DataFrame):
        X, _ = self.split_to_X_and_y(df.copy())
        X = self.preprocessing_pipeline.transform(X.copy())
        return X

    def fit(self, df: pd.DataFrame, transform: bool = False):
        X, y = self.split_to_X_and_y(df.copy())
        self.preprocessing_pipeline.fit(X.copy(), y.copy())

        if transform:
            X = self.preprocessing_pipeline.transform(X.copy())
            X[self.TARGET] = y
            X.to_csv('data/cars_train_transformed.csv')

        return self
