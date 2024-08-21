#ライブラリのimport
import streamlit as st
import pandas as pd
from datetime import datetime, time as dt_time
import pickle
import time
import analysis_v2 # analysis_v2.pyが同じディレクトリにある前提

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
        
# 中間結果変数を保存する関数
def save_model_and_data(rf_model, X, data,product, filename='model_and_data.pkl'):
    with open(filename, 'wb') as file:
        pickle.dump((rf_model, X, data, product), file)
        print(f"Model and data saved to {filename}")
        
# 中間結果変数を読み込む関数
def load_model_and_data(filename='model_and_data.pkl'):
    with open(filename, 'rb') as file:
        rf_model, X, data,product = pickle.load(file)
        print(f"Model and data loaded from {filename}")
        return rf_model, X, data,product

# 品番情報を表示する関数
def display_hinban_info(hinban):
    file_path = '中間成果物/所在管理MBデータ_統合済&特定日時抽出済.csv'
    df = pd.read_csv(file_path, encoding='shift_jis')
    df['品番'] = df['品番'].str.strip()
    filtered_df = df[df['品番'] == hinban]# 品番を抽出
    filtered_df = pd.DataFrame(filtered_df)
    filtered_df = filtered_df.reset_index(drop=True)
    product = filtered_df.loc[0]

    # タイトル表示
    st.header('品番情報')
    
    value1 = str(product['品番'])
    value2 = str(product['品名'])
    value3 = str(product['仕入先名'])
    value4 = str(product['収容数'])
    # 3つの列を作成
    col1, col2, col3, col4 = st.columns(4)
    col1.metric(label="品番", value=value1)
    col2.metric(label="品名", value=value2)
    col3.metric(label="仕入先名", value=value3)
    col4.metric(label="収容数", value=value4)
    #差分表示一例
    #col3.metric(label="仕入先名", value="15 mph", delta="1 mph")

def analysis_page():

    # カスタムCSSを適用して画面サイズを設定する
    st.markdown(
        """
        <style>
        .main .block-container {
            max-width: 70%;
            margin-left: auto;
            margin-right: auto;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # 分析用の各ステップの実行フラグを読み込む
    step1_flag, step2_flag, step3_flag = load_flag()
    
    # 確認用
    # フラグ状態どうなっている？
    #st.sidebar.success(f"{step1_flag}")
    #st.sidebar.success(f"{step2_flag}")
    #st.sidebar.success(f"{step3_flag}")

    st.sidebar.write("## 🔥各ステップを順番に実行してください🔥")

    st.sidebar.title("ステップ１：品番選択")

    # フォーム作成
    with st.sidebar.form(key='my_form'):
    
        #---<ToDo>---
        #変更必要
        #file_path = '中間成果物/所在管理MBデータ_統合済&特定日時抽出済.csv'#こっちは文字化けでエラーになる
        file_path = '中間成果物/所在管理MBデータ_統合済&特定日時抽出済.csv'
        df = pd.read_csv(file_path, encoding='shift_jis')

        # 品番リスト
        df['品番'] = df['品番'].str.strip()
        unique_hinban_list = df['品番'].unique()

        # サイドバーに品番選択ボックスを作成
        product = st.selectbox("品番を選択してください", unique_hinban_list)
        
        # 「適用」ボタンをフォーム内に追加
        submit_button_step1 = st.form_submit_button(label='登録する')

    # 適用ボタンが押されたときの処理
    if submit_button_step1 == True:

        st.sidebar.success(f"新たに選択された品番: {product}")
        
        # analysis_v1.pyの中で定義されたshow_analysis関数を呼び出す
        # 学習
        data, rf_model, X= analysis_v2.show_analysis(product)

        # モデルとデータを保存
        save_model_and_data(rf_model, X, data, product)
        
        #実行フラグを更新する
        step1_flag = 1
        step2_flag = 0
        step3_flag = 0

        # モデルとデータを保存
        save_flag(step1_flag, step2_flag, step3_flag)
        
        display_hinban_info(product)

    # 適用ボタンが押されなかったときの処理
    else:
        
        # まだ一度もSTEP1が実行されていない時
        if step1_flag == 0:
            st.sidebar.warning("品番を選択してください")

        #1度はボタン押されている
        elif step1_flag == 1:
            st.sidebar.success(f"過去に選択された品番: {product}")
            
            # 保存したモデルとデータを読み込む
            rf_model, X, data, product = load_model_and_data()

            display_hinban_info(product)
        
    #--------------------------------------------------------------------------------
        
    # タイトル
    st.sidebar.title("ステップ２：在庫確認")
    
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
    
    #スライドバーで選択するバージョン
    # # サイドバーにフォームの作成
    # with st.sidebar.form(key='filter_form'):
    #     st.session_state.start_date = st.date_input("開始日", st.session_state.start_date)
    #     st.session_state.end_date = st.date_input("終了日", st.session_state.end_date)
    #     start_time_hours = st.slider("開始時間", 0, 23, st.session_state.start_time.hour, format="%02d:00")
    #     end_time_hours = st.slider("終了時間", 0, 23, st.session_state.end_time.hour, format="%02d:00")
    
    #     # 時間を更新
    #     st.session_state.start_time = dt_time(start_time_hours, 0)
    #     st.session_state.end_time = dt_time(end_time_hours, 0)
    
    #     # フォームの送信ボタン
    #     submit_button_step2 = st.form_submit_button(label='適用')

    # 時間の選択肢をリストとして用意
    hours_options = [f"{i:02d}:00" for i in range(24)]

    # サイドバーにフォームの作成
    with st.sidebar.form(key='filter_form'):
        st.session_state.start_date = st.date_input("開始日", st.session_state.start_date)
        st.session_state.end_date = st.date_input("終了日", st.session_state.end_date)

        # 開始時間の設定
        if st.session_state.start_date.weekday() == 0:  # 月曜であるかどうかを確認
            start_time_hours_str = "08:00"
        else:
            start_time_hours_str = "00:00"

        end_time_hours_str = "23:00"
        
        #start_time_hours_str = st.selectbox("開始時間", hours_options, index=st.session_state.start_time.hour)
        #end_time_hours_str = st.selectbox("終了時間", hours_options, index=st.session_state.end_time.hour)

        #st.header(start_time_hours_str)
        #st.header(end_time_hours_str)
        
        # 時間を更新
        st.session_state.start_time = dt_time(int(start_time_hours_str.split(":")[0]), 0)
        st.session_state.end_time = dt_time(int(end_time_hours_str.split(":")[0]), 0)
        
        # フォームの送信ボタン
        submit_button_step2 = st.form_submit_button(label='登録する')
        
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
            st.sidebar.success(f"開始日時: {start_datetime}")
            st.sidebar.success(f"終了日時: {end_datetime}")
            #st.sidebar.success(f"開始日時: {start_datetime}, インデックス: {start_index}")
            #st.sidebar.success(f"終了日時: {end_datetime}, インデックス: {end_index}")
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
            st.sidebar.success(f"開始日時: {start_datetime}")
            st.sidebar.success(f"終了日時: {end_datetime}")
            min_datetime, max_datetime, bar_df, df2 = analysis_v2.step2(data, rf_model, X, start_index, end_index)
            step2_flag = 1

            # モデルとデータを保存
            save_flag(step1_flag, step2_flag, step3_flag)
            
        
    #--------------------------------------------------------------------------------
    
    # サイドバーに日時選択スライダーを表示
    st.sidebar.title("ステップ３：要因分析")
    
    # スライドバーで表示するよう
    # # フォーム作成
    # with st.sidebar.form("date_selector_form"):
    #     selected_datetime = st.slider(
    #         "要因分析の結果を表示する日時を選択してください",
    #         min_value=min_datetime,
    #         max_value=max_datetime,
    #         value=min_datetime,
    #         format="YYYY-MM-DD HH",
    #         step=pd.Timedelta(hours=1)
    #     )
    #     submit_button_step3 = st.form_submit_button("登録する")

    # 日時の選択肢を生成
    datetime_range = pd.date_range(min_datetime, max_datetime, freq='H')
    datetime_options = [dt.strftime("%Y-%m-%d %H:%M") for dt in datetime_range]

    # フォーム作成
    with st.sidebar.form("date_selector_form"):
        # 日時選択用セレクトボックス
        selected_datetime = st.selectbox(
            "要因分析の結果を表示する日時を選択してください",
            datetime_options
        )
        submit_button_step3 = st.form_submit_button("登録する")

        
    if submit_button_step3:
        step3_flag = 1
        st.sidebar.success(f"選択された日時: {selected_datetime}")
        
        analysis_v2.step3(bar_df, df2, selected_datetime)
        
        # モデルとデータを保存
        save_flag(step1_flag, step2_flag, step3_flag)
    
    elif step3_flag == 0:
        st.sidebar.warning("日時を選択してください")

def main():
    
    #スライドバーの設定
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

        # プログレスバーの初期化
        #progress_bar = st.progress(0)

        # ステータス表示のためのテキスト
        #status_text = st.empty()

        # 進捗状況の更新
        #for i in range(101):
            # プログレスバーの進捗を更新
            #progress_bar.progress(i)
            
            # ステータス表示の更新
            #status_text.text(f'Progress: {i}%')
            
            # 処理の遅延をシミュレート
            #time.sleep(0.1)  # 0.1秒間の遅延

        # 処理完了後のメッセージ
        #st.success("Processing complete!")

#本スクリプトが直接実行されたときに実行
if __name__ == "__main__":

    print("プログラムを開始します")
    
    main()
