# app.py
import streamlit as st
import pandas as pd
import analysis_v2 # analysis_v1.pyが同じディレクトリにある前提
import sys
from datetime import datetime, time as dt_time
import pickle

# 分析用の各ステップの実行フラグを保存する関数
def save_flag(step1_flag, step2_flag, step3_flag, filename='flag.pkl'):
    with open(filename, 'wb') as file:
        pickle.dump((step1_flag, step2_flag, step3_flag), file)
        
# 分析用の各ステップの実行フラグを読み込む関数
def load_flag(filename='flag.pkl'):
    with open(filename, 'rb') as file:
        step1_flag, step2_flag, step3_flag = pickle.load(file)
        print(f"Model and data loaded from {filename}")
        return step1_flag, step2_flag, step3_flag
        
# `rf_model`と`X`と'data'を保存する関数
def save_model_and_data(rf_model, X, data, filename='model_and_data.pkl'):
    with open(filename, 'wb') as file:
        pickle.dump((rf_model, X, data), file)
        print(f"Model and data saved to {filename}")
        
# `rf_model`と`X`と'data'を読み込む関数
def load_model_and_data(filename='model_and_data.pkl'):
    with open(filename, 'rb') as file:
        rf_model, X, data = pickle.load(file)
        print(f"Model and data loaded from {filename}")
        return rf_model, X, data


def main():
    
    st.sidebar.title("ナビゲーション")
    page = st.sidebar.radio("ページ選択", ["🏠 ホーム", "📊 分析","📖 マニュアル"])
    
    # 折り返し線を追加
    st.sidebar.markdown("---")

    if page == "🏠 ホーム":
    
        #アプリ立ち上げ時に分析ページの実行フラグを初期化
        step1_flag = 0
        step2_flag = 0
        step3_flag = 0
                
        # 分析用の各ステップの実行フラグを保存
        save_flag(step1_flag, step2_flag, step3_flag)
        
        st.title("🤖 AI在庫分析アプリ 0.01")
        st.write("このアプリは、AIを使って在庫の分析を行うためのツールです。外部のDBからデータをインポートし、リアルタイムで在庫の分析を行います")
        
    elif page == "📊 分析":
        analysis_page()
        
    elif page == "📖 マニュアル":
        st.title("マニュアル")

def analysis_page():

    # 分析用の各ステップの実行フラグを読み込む
    step1_flag, step2_flag, step3_flag = load_flag()
    
    # 確認用
    st.sidebar.info(f"{step1_flag}")
    st.sidebar.info(f"{step2_flag}")
    st.sidebar.info(f"{step3_flag}")

    st.sidebar.title("STEP1：データ選択")

    # フォーム作成
    with st.sidebar.form(key='my_form'):
    
        #---<ToDo>---
        #変更ふ必要
        file_path = '中間成果物/所在管理MBデータ_統合済&特定日時抽出済.csv'
        df = pd.read_csv(file_path, encoding='shift_jis')

        # 品番リスト
        df['品番'] = df['品番'].str.strip()
        unique_hinban_list = df['品番'].unique()

        # サイドバーに品番選択ボックスを作成
        product = st.selectbox("品番を選択してください", unique_hinban_list)
        
        # 「適用」ボタンをフォーム内に追加
        submit_button_step1 = st.form_submit_button(label='適用')

    # 適用ボタンが押されたときの処理
    if submit_button_step1 == True:

        st.sidebar.info(f"新たに選択された品番: {product}")
        
        # analysis_v1.pyの中で定義されたshow_analysis関数を呼び出す
        # 学習
        data, rf_model, X = analysis_v2.show_analysis(product)

        # モデルとデータを保存
        save_model_and_data(rf_model, X, data)
        
        #実行フラグを更新する
        step1_flag = 1
        step2_flag = 0
        step3_flag = 0

        # モデルとデータを保存
        save_flag(step1_flag, step2_flag, step3_flag)


    # 適用ボタンが押されなかったときの処理
    else:
        
        # まだ一度もSTEP1が実行されていない時
        if step1_flag == 0:
            st.sidebar.warning("品番を選択してください")

        #1度はボタン押されている
        elif step1_flag == 1:
            st.sidebar.info(f"過去に選択された品番: {product}")
            
            # 保存したモデルとデータを読み込む
            rf_model, X, data = load_model_and_data()
        
        
    #--------------------------------------------------------------------------------
        
    # サイドバーで開始日と終了日を選択
    st.sidebar.title("STEP2：データ確認")
    
    # ---<ToDo>---
    # データの最小日時と最大日時を取得
    data = pd.read_csv("一時保存データ.csv",encoding='shift_jis')
    data['日時'] = pd.to_datetime(data['日時'], errors='coerce')
    min_datetime = data['日時'].min()
    max_datetime = data['日時'].max()
    
    #確認用
    #print(min_datetime,max_datetime)
    
    default_values = {
        'start_date': min_datetime.date(),
        'end_date': max_datetime.date(),
        'start_time': datetime.strptime("00:00", "%H:%M").time(),  # 0:00として初期化
        'end_time': datetime.strptime("23:00", "%H:%M").time(),  # 23:00として初期化
        'button_clicked': False
    }
    
    for key, value in default_values.items():
        if key not in st.session_state:
            st.session_state[key] = value
    
    # サイドバーにフォームの作成
    with st.sidebar.form(key='filter_form'):
        st.session_state.start_date = st.date_input("開始日", st.session_state.start_date)
        st.session_state.end_date = st.date_input("終了日", st.session_state.end_date)
        start_time_hours = st.slider("開始時間", 0, 23, st.session_state.start_time.hour, format="%02d:00")
        end_time_hours = st.slider("終了時間", 0, 23, st.session_state.end_time.hour, format="%02d:00")
    
        # 時間を更新
        st.session_state.start_time = dt_time(start_time_hours, 0)
        st.session_state.end_time = dt_time(end_time_hours, 0)
    
        # フォームの送信ボタン
        submit_button_step2 = st.form_submit_button(label='適用')
        
    data = data.reset_index(drop=True)
    
    # 開始日時と終了日時を結合
    start_datetime = datetime.combine(st.session_state.start_date, st.session_state.start_time)
    end_datetime = datetime.combine(st.session_state.end_date, st.session_state.end_time)
    
    print(start_datetime, end_datetime)

    # start_datetimeとend_datetimeに対応するインデックスを見つける
    start_index = data.index[data['日時'] == start_datetime].tolist()
    end_index = data.index[data['日時'] == end_datetime].tolist()
    
    # フォームが送信された場合の処理
    if submit_button_step2:
        
        if start_index == [] or end_index == []:
            st.sidebar.error("非稼動日を選択しています。")
            step2_flag = 2
            
            
        else:
            st.sidebar.info(f"開始日時: {start_datetime}, インデックス: {start_index}")
            st.sidebar.info(f"終了日時: {end_datetime}, インデックス: {end_index}")
            min_datetime, max_datetime, bar_df, df2 = analysis_v2.step2(data, rf_model, X, start_index, end_index)
            step2_flag = 1

            # モデルとデータを保存
            save_flag(step1_flag, step2_flag, step3_flag)
            
            
    else:

        if step2_flag == 0:
            st.sidebar.warning("開始日、終了日、開始時間、終了時間を選択し、実行ボタンを押してください。")
            min_datetime = min_datetime.to_pydatetime()
            max_datetime = max_datetime.to_pydatetime()
            
        elif step2_flag == 1:
            st.sidebar.info(f"開始日時1: {start_datetime}, インデックス: {start_index}")
            st.sidebar.info(f"終了日時1: {end_datetime}, インデックス: {end_index}")
            min_datetime, max_datetime, bar_df, df2 = analysis_v2.step2(data, rf_model, X, start_index, end_index)
            step2_flag = 1

            # モデルとデータを保存
            save_flag(step1_flag, step2_flag, step3_flag)
            
        
    #--------------------------------------------------------------------------------
    
    # サイドバーに日時選択スライダーを表示
    st.sidebar.title("STEP3：AIデータ分析")
    
    # フォーム作成
    with st.sidebar.form("date_selector_form"):
        selected_datetime = st.slider(
            "要因分析の結果を表示する日時を選択してください",
            min_value=min_datetime,
            max_value=max_datetime,
            value=min_datetime,
            format="YYYY-MM-DD HH",
            step=pd.Timedelta(hours=1)
        )
        submit_button_step3 = st.form_submit_button("適用")
        
    if submit_button_step3:
        step3_flag = 1
        st.sidebar.info(f"選択された日時: {selected_datetime}")
        
        analysis_v2.step3(bar_df, df2, selected_datetime)
        
        # モデルとデータを保存
        save_flag(step1_flag, step2_flag, step3_flag)
    
    elif step3_flag == 0:
        st.sidebar.warning("日時を選択してください")

if __name__ == "__main__":

    print("test")
    
    main()
