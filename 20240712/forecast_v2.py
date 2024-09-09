#ライブラリのimport
import os
import pandas as pd
import warnings
import numpy as np
import pandas as pd
#%matplotlib inline#Jupyter Notebook 専用のマジックコマンド。メンテ用で利用
import matplotlib.pyplot as plt
import shap
import seaborn as sns
import matplotlib as mpl
from dateutil.relativedelta import relativedelta
from IPython.display import display, clear_output
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from matplotlib.gridspec import GridSpec
from datetime import datetime
from datetime import timedelta
from PIL import Image
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error, max_error, mean_absolute_error
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, date, time
import pickle
from sklearn.preprocessing import StandardScaler

from read_v2 import read_data, process_Activedata

def show_forecast(product,start_datetime):

    #st.header(product)
    #st.header(start_datetime)

    #!-----------------------------------------------------------------------
    #!自動ラックの在庫データ
    #!-----------------------------------------------------------------------
    file_path = '中間成果物/在庫推移MBデータ_統合済&特定日時抽出済.csv'
    zaiko_df = pd.read_csv(file_path, encoding='shift_jis')
    # 品番列の空白を削除
    zaiko_df['品番'] = zaiko_df['品番'].str.strip()
    # '計測日時'をdatatime型に変換
    zaiko_df['計測日時'] = pd.to_datetime(zaiko_df['計測日時'], errors='coerce')
    zaiko_df = zaiko_df.rename(columns={'計測日時': '日時'})

    zaiko_df = zaiko_df[zaiko_df['品番'] == product]

    # 2. Plotlyを使って棒グラフを作成
    fig = px.bar(zaiko_df, x='日時', y='在庫数（箱）', title='Sample Bar Chart')

    # 3. StreamlitでPlotlyのグラフを表示
    st.plotly_chart(fig)

    activedata = process_Activedata()
    activedata = activedata[activedata['品番'] == product]
    #!稼働時間で割る　休憩時間の考慮が必要？
    activedata['日量数（箱数）'] = activedata['日量数（箱数）'] / 16.5
    activedata = activedata.set_index('日付').resample('H').ffill().reset_index()

    fig2 = px.bar(activedata, x='日付', y='日量数（箱数）', title='Sample Bar Chart')

    st.plotly_chart(fig2)
    st.dataframe(activedata)

    specific_time = start_datetime
    zaiko_df = zaiko_df[zaiko_df['日時'] == specific_time]

    fig3 = px.bar(zaiko_df, x='日時', y='在庫数（箱）', title='Sample Bar Chart')

    st.plotly_chart(fig3)
    st.dataframe(zaiko_df)

    # Extract relevant columns from the inventory dataframe
    zaiko_df['日時'] = pd.to_datetime(zaiko_df['日時'])
    zaiko_extracted = zaiko_df[['日時', '在庫数（箱）']]

    # Extract relevant columns from the consumption inventory dataframe
    activedata['日付'] = pd.to_datetime(activedata['日付'])
    activedata_extracted = activedata[['日付', '日量数（箱数）']]

    # Find the matching start time in activedata_df
    start_time = zaiko_extracted.iloc[0]['日時']
    end_time = start_time + pd.Timedelta(hours=20)
    filtered_activedata = activedata_extracted[(activedata_extracted['日付'] >= start_time) & (activedata_extracted['日付'] < end_time)]

    # Calculate the inventory after each hour
    inventory_after_consumption_corrected = []
    current_inventory = zaiko_extracted.iloc[0]['在庫数（箱）']

    for i, row in filtered_activedata.iterrows():
        inventory_after_consumption_corrected.append({
            '日時': row['日付'],
            '在庫数（箱）': current_inventory
        })
        if i != 0:  # Skip subtraction for the first timestamp
            current_inventory -= row['日量数（箱数）']


    inventory_df_corrected = pd.DataFrame(inventory_after_consumption_corrected)
    st.dataframe(inventory_df_corrected)

    fig3 = px.bar(inventory_df_corrected, x='日時', y='在庫数（箱）', title='在庫予測')

    st.plotly_chart(fig3)





