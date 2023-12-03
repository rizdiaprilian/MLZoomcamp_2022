from pydantic import BaseModel, validator
import pandera as pa
from pandera.typing import DataFrame, Series

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import DictVectorizer


class DataSets(pa.DataFrameModel):
    seniority: Series[int]
    home: Series[str]
    time: Series[int]
    age: Series[int]
    marital: Series[str]
    records: Series[str]
    job: Series[str]
    expenses: Series[int]
    income: Series[float]
    assets: Series[float]
    debt: Series[float]
    amount: Series[int]
    price: Series[int]

class ProcessConfig(BaseModel):
    test_size: float = 0.4

    @validator("test_size")
    def must_be_non_negative(cls, v):
        if v < 0:
            raise ValueError(f"{v} must be non-negative.")
        return v

def loading_data(
        raw_file: str = "CreditScoring.csv"
) -> pd.DataFrame:
    df = pd.read_csv(raw_file)
    df.columns = df.columns.str.lower()

    return df

def preprocessing(df: pd.DataFrame) -> pd.DataFrame:
    status_values = {
        1: 'ok',
        2: 'default',
        0: 'unk'
    }

    home_values = {
        1: 'rent',
        2: 'owner',
        3: 'private',
        4: 'ignore',
        5: 'parents',
        6: 'other',
        0: 'unk'
    }

    marital_values = {
        1: 'single',
        2: 'married',
        3: 'widow',
        4: 'separated',
        5: 'divorced',
        0: 'unk'
    }

    records_values = {
        1: 'no',
        2: 'yes',
        0: 'unk'
    }

    job_values = {
        1: 'fixed',
        2: 'partime',
        3: 'freelance',
        4: 'others',
        0: 'unk'
    }

    df.marital = df.marital.map(marital_values)
    df.home = df.home.map(home_values)
    df.records = df.records.map(records_values)
    df.status = df.status.map(status_values)
    df.job = df.job.map(job_values)

    for c in ['income', 'assets', 'debt']:
        df[c] = df[c].replace(to_replace=99999999, value=np.nan)
    
    df = df[df.status != 'unk'].reset_index(drop=True)

    return df

@pa.check_types
def prepare(
    df: pd.DataFrame, 
    test_size: float = 0.3
        ) -> DataFrame[DataSets]:
    df_full_train, df_test = train_test_split(df, test_size=test_size, random_state=11)
    df_train, df_val = train_test_split(df_full_train, test_size=0.25, random_state=11)

    df_train = df_train.fillna(0).reset_index(drop=True)
    df_val = df_val.fillna(0).reset_index(drop=True)
    df_test = df_test.fillna(0).reset_index(drop=True)

    y_train = (df_train.status == 'default').astype('int').values
    y_val = (df_val.status == 'default').astype('int').values
    y_test = (df_test.status == 'default').astype('int').values

    del df_train['status']
    del df_val['status']
    del df_test['status']

    return df_train, df_val

@pa.check_types
def transform_data(
        df_train: pd.DataFrame, 
        df_val: pd.DataFrame):
    dv = DictVectorizer(sparse=False)

    # Filling missing values with 0
    dict_train = df_train.to_dict(orient='records')
    dict_val = df_val.to_dict(orient='records')

    X_train = dv.fit_transform(dict_train)
    X_val = dv.transform(dict_val)
    
    return X_train, X_val

def main(
        process_config: ProcessConfig = ProcessConfig()
):
    df = loading_data()
    df = preprocessing(df)
    df_train, df_val = prepare(df, process_config.test_size)

    X_train, X_val = transform_data(df_train, df_val)


if __name__ =="__main__":
    main(process_config=ProcessConfig(test_size=0.2))