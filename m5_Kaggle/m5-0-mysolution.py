import numpy as np
import pandas as pd
import os
import psutil

# Função para verificar o uso de memória
def get_memory_usage():
    return np.round(psutil.Process(os.getpid()).memory_info()[0] / 2.0 ** 30, 2)

# Função para reduzir o uso de memória
def reduce_mem_usage(df):
    numerics = ['int8', 'int16', 'int32', 'float16', 'float32', 'float64']
    for col in df.columns:
        col_type = df[col].dtype
        if col_type in numerics:
            c_min, c_max = df[col].min(), df[col].max()
            if str(col_type).startswith('int'):
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
    return df

# Função para mesclar dataframes
def merge_by_concat(df1, df2, merge_on):
    merged = df1[merge_on].merge(df2, on=merge_on, how='left')
    return pd.concat([df1, merged.drop(columns=merge_on)], axis=1)

TARGET = 'sales'         # Our main target
END_TRAIN = 1913+28         # Last day in train set
MAIN_INDEX = ['id','d']  # We can identify item by these columns

# Carregar dados
train_df = pd.read_csv('Data/sales_train_evaluation.csv')
prices_df = pd.read_csv('Data/sell_prices.csv')
calendar_df = pd.read_csv('Data/calendar.csv')

#FIltrando loja para testes 
stores = ["CA_1", "CA_2"]
train_df = train_df[train_df['store_id'].isin(stores)]

# Filtrando Categoria
cat_selected = "HOBBIES"
train_df = train_df[train_df['cat_id'].isin([cat_selected])]

index_columns = ['id','item_id','dept_id','cat_id','store_id','state_id']
# Transformando os dados
grid_df = pd.melt(train_df, id_vars=index_columns,
                   var_name='d', value_name='sales')

index_columns = ['id','item_id','dept_id','cat_id','store_id','state_id']

add_grid = pd.DataFrame()
for i in range(1,29):
    temp_df = train_df[index_columns]
    temp_df = temp_df.drop_duplicates()
    temp_df['d'] = 'd_'+ str(END_TRAIN+i)
    temp_df[TARGET] = np.nan
    add_grid = pd.concat([add_grid,temp_df])

del train_df, temp_df

grid_df = pd.concat([grid_df,add_grid])
del add_grid