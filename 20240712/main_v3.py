#メイン用

#! ライブラリのimport
import streamlit as st
import pandas as pd
from datetime import datetime, time as dt_time
from datetime import datetime, timedelta
import pickle
import matplotlib.pyplot as plt
import plotly.express as px
import fitz  # PyMuPDF
#! 自作ライブラリのimport
#データ読み取り用
from read_v3 import read_data, process_Activedata
import analysis_v3 # analysis_v3.pyが同じディレクトリにある前提
import forecast_v3

#! 要因分析用の各ステップの実行フラグを保存する関数
def save_flag_analysis(step1_flag, step2_flag, step3_flag, filename='temp/flag_analysis.pkl'):
    with open(filename, 'wb') as file:
        pickle.dump((step1_flag, step2_flag, step3_flag), file)
        
#! 要因分析用の各ステップの実行フラグを読み込む関数
def load_flag_analysis(filename='temp/flag_analysis.pkl'):
    with open(filename, 'rb') as file:
        step1_flag, step2_flag, step3_flag = pickle.load(file)
        print(f"Model and data loaded from {filename}")
        return step1_flag, step2_flag, step3_flag
    
#! 予測用の各ステップの実行フラグを保存する関数
def save_flag_predict(step1_flag, step2_flag, step3_flag, filename='temp/flag_predict.pkl'):
    with open(filename, 'wb') as file:
        pickle.dump((step1_flag, step2_flag, step3_flag), file)
        
#! 予測用の各ステップの実行フラグを読み込む関数
def load_flag_predict(filename='temp/flag_predict.pkl'):
    with open(filename, 'rb') as file:
        step1_flag, step2_flag, step3_flag = pickle.load(file)
        print(f"Model and data loaded from {filename}")
        return step1_flag, step2_flag, step3_flag
        
#! 中間結果変数を保存する関数
def save_model_and_data(rf_model, rf_model2, rf_model3, X, data,product, filename='temp/model_and_data.pkl'):
    with open(filename, 'wb') as file:
        pickle.dump((rf_model, rf_model2, rf_model3, X, data, product), file)
        print(f"Model and data saved to {filename}")
        
#! 中間結果変数を読み込む関数
def load_model_and_data(filename='temp/model_and_data.pkl'):
    with open(filename, 'rb') as file:
        rf_model, rf_model2, rf_model3, X, data,product = pickle.load(file)
        print(f"Model and data loaded from {filename}")
        return rf_model, rf_model2, rf_model3, X, data,product

#! ユニークな品番リスト「品番_整備室」を作成する関数（Activeデータを活用)
def create_hinban_info():

    file_path = 'temp/activedata.csv'
    df = pd.read_csv(file_path, encoding='shift_jis')

    #! ユニークな品番リストを作成
    df['品番'] = df['品番'].str.strip()
    unique_hinban_list = df['品番'].unique()

    #! '品番' ごとに '整備室' のユニークな値を集める
    hinban_seibishitsu_df = df.groupby('品番')['整備室'].unique().reset_index()

    #! '整備室' のユニークな値を行ごとに展開
    hinban_seibishitsu_df = hinban_seibishitsu_df.explode('整備室')

    #!　ユニークな '品番_整備室' 列を作成
    hinban_seibishitsu_df['品番_整備室'] = hinban_seibishitsu_df.apply(lambda row: f"{row['品番']}_{row['整備室']}", axis=1)

    return hinban_seibishitsu_df

#! 品番情報を表示する関数
def display_hinban_info(hinban):

    #file_path = '中間成果物/所在管理MBデータ_統合済&特定日時抽出済.csv'
    #file_path = '中間成果物/所在管理MBデータ_統合済&特定日時抽出済.csv'#こっちは文字化けでエラーになる
    #df = pd.read_csv(file_path, encoding='shift_jis')
    #df = process_Activedata()
    file_path = 'temp/activedata.csv'
    df = pd.read_csv(file_path, encoding='shift_jis')
    df['品番'] = df['品番'].str.strip()
    hinban = hinban.split('_')[0]#整備室情報削除
    filtered_df = df[df['品番'] == hinban]# 品番を抽出
    filtered_df = pd.DataFrame(filtered_df)
    filtered_df = filtered_df.reset_index(drop=True)
    product = filtered_df.loc[0]

    # タイトル表示
    st.header('品番情報')
    
    value1 = str(product['品番'])
    value2 = str(product['品名'])
    value3 = str(product['仕入先名/工場名'])
    value4 = str(product['収容数'])
    value5 = str(product['整備室'])
    
    # 5つの列で表示
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric(label="品番", value=value1)
    col2.metric(label="品名", value=value2)
    col3.metric(label="仕入先名", value=value3)
    col4.metric(label="収容数", value=value4)
    col5.metric(label="整備室", value=value5)
    
    #差分表示一例
    #col3.metric(label="仕入先名", value="15 mph", delta="1 mph")

#! カスタムCSS
def apply_custom_css():
    """
    カスタムCSSを適用して、画面サイズを設定する関数。
    """
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

#-----------------------------------------------------------------------------------------------------------------------------------

#! 予測ページ
def forecast_page():

    # ページタイトル
    st.title("在庫リミット計算")
    st.info("📌 **この画面では、数時間先の在庫を計算することができます。実行する際は左側のサイドバーで各種設定を行ってください。**")

    # 折り返し線を追加
    st.markdown("---")

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
    step1_flag_predict, step2_flag_predict, step3_flag_predict = load_flag_predict()
 
    # サイドバートップメッセージ
    st.sidebar.write("## 🔥各ステップを順番に実行してください🔥")

    #!-------------------------------------------------------------------------------
    #! 予測ページのステップ1のサイドバータイトル
    #!-------------------------------------------------------------------------------
    st.sidebar.title("ステップ１：品番選択")

    #! フォーム作成
    with st.sidebar.form(key='my_form'):
    
        hinban_seibishitsu_df = create_hinban_info()

        # サイドバーに品番選択ボックスを作成
        product = st.selectbox("品番を選択してください", hinban_seibishitsu_df['品番_整備室'])
        
        # 「適用」ボタンをフォーム内に追加
        submit_button_step1 = st.form_submit_button(label='登録する')

    #! 適用ボタンが押されたときの処理
    if submit_button_step1 == True:

        st.sidebar.success(f"新たに選択された品番: {product}")
        
        # モデルとデータを保存
        save_model_and_data(None, None,None, None, None, product)
        
        #実行フラグを更新する
        step1_flag_predict = 1
        step2_flag_predict = 0
        step3_flag_predict = 0

        # モデルとデータを保存
        save_flag_predict(step1_flag_predict, step2_flag_predict, step3_flag_predict)
        
        #!　品番情報を表示
        display_hinban_info(product)

        # 折り返し線を追加
        st.markdown("---")


    #! 適用ボタンが押されなかったときの処理
    else:
        
        #! まだ一度もSTEP1が実行されていない時
        if step1_flag_predict == 0:
            st.sidebar.warning("品番を選択してください")

        #! 1度はボタン押されている
        elif step1_flag_predict == 1:
            st.sidebar.success(f"過去に選択された品番: {product}")
            
            #! 品番情報表示
            display_hinban_info(product)

            # 折り返し線を追加
            st.markdown("---")

    
    #!-------------------------------------------------------------------------------
    #! 予測ページのステップ2のサイドバータイトル
    #!-------------------------------------------------------------------------------
    st.sidebar.title("ステップ２：日時選択")

    # max_datetimeは現在の実行時刻
    max_datetime = datetime.now()

    # min_datetimeは1年前の日付
    min_datetime = max_datetime - timedelta(days=365)
    
    default_values = {
        'start_date': max_datetime.date(),
        'start_time': datetime.strptime("00:00", "%H:%M").time(),  # 0:00として初期化
        'end_time': datetime.strptime("23:00", "%H:%M").time(),  # 23:00として初期化
        'button_clicked': False
    }
    
    for key, value in default_values.items():
        if key not in st.session_state:
            st.session_state[key] = value
    
    with st.sidebar.form(key='filter_form'):

        # 開始日
        st.session_state.start_date = st.date_input("開始日", st.session_state.start_date)
        
        # 開始時間の選択肢をセレクトボックスで提供
        # サイドバーにフォームの作成
        hours = [f"{i:02d}:00" for i in range(24)]
        start_time_str = st.selectbox("開始時間", hours, index=st.session_state.start_time.hour)
        
        # 選択された時間をdt_timeオブジェクトに変換
        start_time_hours = int(start_time_str.split(":")[0])

        # 時間を更新
        st.session_state.start_time = dt_time(start_time_hours, 0)

        # フォームの送信ボタン
        submit_button_step2 = st.form_submit_button(label='登録する')
    
    # 開始日時と終了日時を結合
    start_datetime = datetime.combine(st.session_state.start_date, st.session_state.start_time)
    
    # ボタンを押された時
    if submit_button_step2:

        if (step1_flag_predict == 1):

            st.sidebar.success(f"開始日時: {start_datetime}")
            step2_flag_predict = 1

            # モデルとデータを保存
            save_flag_predict(step1_flag_predict, step2_flag_predict, step3_flag_predict)

        else:
            st.sidebar.error("順番にステップを実行ください")

    # ボタンを押されなかった時       
    else:

        if step2_flag_predict == 0:
            st.sidebar.warning("開始日、開始時間を選択し、登録するボタンを押してください。")
            min_datetime = min_datetime
            #min_datetime = min_datetime.to_pydatetime()
            
        elif step2_flag_predict == 1:
            st.sidebar.success(f"開始日時: {start_datetime}")
            min_datetime = start_datetime
            step2_flag_predict = 1

            # モデルとデータを保存
            save_flag_predict(step1_flag_predict, step2_flag_predict, step3_flag_predict)

    #!-------------------------------------------------------------------------------
    #! 予測ページのステップ3のサイドバータイトル
    #!-------------------------------------------------------------------------------
    st.sidebar.title("ステップ３：在庫数入力")

    # フォーム作成
    with st.sidebar.form("date_selector_form"):
        # 日時選択用セレクトボックス
        selected_zaiko = st.selectbox("組立ラインの在庫数（箱）を入力してください",list(range(0,10)))
        submit_button_step3 = st.form_submit_button("登録する")

    # ボタンが押された時
    if submit_button_step3:
        step3_flag_predict = 1

        if (step1_flag_predict == 1) and (step2_flag_predict == 1):

            st.sidebar.success(f"入力された在庫数: {selected_zaiko}")#、在庫数（箱）：{int(zaikosu)}")
            #rf_model, X, data, product = load_model_and_data()
            forecast_v3.show_forecast(product,start_datetime,selected_zaiko)
            
            step3_flag_predict = 0
            
            # モデルとデータを保存
            save_flag_predict(step1_flag_predict, step2_flag_predict, step3_flag_predict)

        else:
            st.sidebar.error("順番にステップを実行ください")

    # ボタンが押されなかった時
    else:
        # STEP1が未達の時
        if (step1_flag_predict == 0) or (step2_flag_predict == 0):
            st.sidebar.warning("在庫数を入力してください")
        
        # STEP2が未達の時
        elif step2_flag_predict == 1:
            st.sidebar.warning("在庫数を入力してください")

#-----------------------------------------------------------------------------------------------------------------------------------
#! 要因分析ページ            
def analysis_page():

    st.title("在庫変動要因分析")
    st.info("📌 **この画面では、在庫変動の要因分析を行うことができます。実行する際は左側のサイドバーで各種設定を行ってください。**")

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
    step1_flag_analysis, step2_flag_analysis, step3_flag_analysis = load_flag_analysis()
    
    # 確認用
    # フラグ状態どうなっている？
    #st.sidebar.success(f"{step1_flag_analysis}")
    #st.sidebar.success(f"{step3_flag_analysis}")
    #st.sidebar.success(f"{step3_flag_analysis}")

    st.sidebar.write("## 🔥各ステップを順番に実行してください🔥")

    #! ステップ１
    st.sidebar.title("ステップ１：品番選択")

    # フォーム作成
    with st.sidebar.form(key='my_form'):
    
        #!　ユニークな '品番_整備室' 列を作成
        hinban_seibishitsu_df = create_hinban_info()

        #! サイドバーに品番選択ボックスを作成
        product = st.selectbox("品番を選択してください", hinban_seibishitsu_df['品番_整備室'])
        
        # 「登録する」ボタンをフォーム内に追加
        submit_button_step1 = st.form_submit_button(label='登録する')

    # 適用ボタンが押されたときの処理
    if submit_button_step1 == True:

        st.sidebar.success(f"新たに選択された品番: {product}")
        
        # analysis_v1.pyの中で定義されたshow_analysis関数を呼び出す
        # 学習
        data, rf_model, rf_model2, rf_model3, X= analysis_v3.show_analysis(product)
        #data, rf_model2, X= analysis_v3.show_analysis(product, '2024-05-01-00', '2024-08-31-00')
        #data, rf_model3, X= analysis_v3.show_analysis(product, '2024-05-01-00', '2024-08-31-00')

        #! モデルとデータを保存
        #save_model_and_data(rf_model, X, data, product)
        save_model_and_data(rf_model, rf_model2, rf_model3, X, data, product, filename='temp/model_and_data.pkl')
        
        #実行フラグを更新する
        step1_flag_analysis = 1
        step3_flag_analysis = 0
        step3_flag_analysis = 0

        #! フラグを保存
        save_flag_analysis(step1_flag_analysis, step2_flag_analysis, step3_flag_analysis)
        
        display_hinban_info(product)

    # 適用ボタンが押されなかったときの処理
    else:
        
        # まだ一度もSTEP1が実行されていない時
        if step1_flag_analysis == 0:
            st.sidebar.warning("品番を選択し、「登録する」ボタンをてください")

        #1度はボタン押されている
        elif step1_flag_analysis == 1:
            st.sidebar.success(f"過去に選択された品番: {product}")
            
            #! 保存したモデルとデータを読み込む
            rf_model, rf_model2, rf_model3, X, data, product = load_model_and_data()

            display_hinban_info(product)
        
    #--------------------------------------------------------------------------------
        
    #! ステップ２
    st.sidebar.title("ステップ２：在庫確認")
    
    # ---<ToDo>---
    # データの最小日時と最大日時を取得
    data = pd.read_csv("temp/一時保存データ.csv",encoding='shift_jis')
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
            step3_flag_analysis = 2 #2は非稼働日を表す
            
        else:
            st.sidebar.success(f"開始日時: {start_datetime}")
            st.sidebar.success(f"終了日時: {end_datetime}")
            
            #st.sidebar.info(step1_flag_analysis)
            #st.sidebar.info(step2_flag_analysis)
            #st.sidebar.info(step3_flag_analysis)

            step3_flag_analysis = 0

            #st.sidebar.success(f"開始日時: {start_datetime}, インデックス: {start_index}")
            #st.sidebar.success(f"終了日時: {end_datetime}, インデックス: {end_index}")
            bar_df, df2, line_df = analysis_v3.step2(data, rf_model, X, start_index, end_index, step3_flag_analysis)
            min_datetime = start_datetime
            max_datetime = end_datetime
            step3_flag_analysis = 1

            # モデルとデータを保存
            save_flag_analysis(step1_flag_analysis, step2_flag_analysis, step3_flag_analysis)
            
    else:

        if step3_flag_analysis == 0:
            st.sidebar.warning("開始日、終了日を選択し、「登録する」ボタンを押してください。")
            min_datetime = min_datetime.to_pydatetime()
            max_datetime = max_datetime.to_pydatetime()
            
        elif step3_flag_analysis == 1:
            st.sidebar.success(f"開始日時: {start_datetime}")
            st.sidebar.success(f"終了日時: {end_datetime}")
            min_datetime = start_datetime
            max_datetime = end_datetime
            step2_flag_analysis = 1

            # モデルとデータを保存
            save_flag_analysis(step1_flag_analysis, step2_flag_analysis, step3_flag_analysis)
            
        
    #--------------------------------------------------------------------------------
    
    #! ステップ３
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

        step3_flag_analysis = 1

        bar_df, df2, line_df = analysis_v3.step2(data, rf_model, X, start_index, end_index, step3_flag_analysis, selected_datetime)
        #zaikosu = line_df.loc[line_df['日時'] == selected_datetime, '在庫数（箱）'].values[0]
        analysis_v3.step3(bar_df, df2, selected_datetime, line_df)

        st.sidebar.success(f"選択された日時: {selected_datetime}")#、在庫数（箱）：{int(zaikosu)}")

        step2_flag_analysis = 0
        
        # モデルとデータを保存
        save_flag_analysis(step1_flag_analysis, step3_flag_analysis, step3_flag_analysis)
    
    elif (step2_flag_analysis == 0) or (step3_flag_analysis == 0) or (step2_flag_analysis == 1):
        st.sidebar.warning("要因分析の結果を表示する日時を選択し、「登録する」ボタンを押してください")

#-----------------------------------------------------------------------------------------------------------------------------------

#! 全体ページ構成
def main():

    st.sidebar.title("メインメニュー")

    # メインメニューの選択
    main_menu = st.sidebar.radio("ページ選択", ["🏠 ホーム", "🔍 可視化", "📊 分析", "⏳ 予測（準備中）", "📖 マニュアル"])

    #ページ変数の初期化
    #page = None

    # メインメニューの選択に応じた処理
    if main_menu == "🏠 ホーム":
        page = "🏠 ホーム"
    elif main_menu == "🔍 可視化":
        page = "🔍 可視化"
    elif main_menu == "📊 分析":
        # 分析のサブメニュー
        main_menu_analysis = st.sidebar.radio("分析ページ選択", ["要因分析"], key='analysis')
        page = main_menu_analysis
    elif main_menu == "⏳ 予測（準備中）":
        # 予測のサブメニュー
        main_menu_prediction = st.sidebar.radio("予測ページ選択", ["在庫リミット計算", "在庫予測"], key='prediction')
        page = main_menu_prediction
    elif main_menu == "📖 マニュアル":
        page = "📖 マニュアル"
    #else:
        #st.title("ページを選択してください。")
        
    
    #! 折り返し線を追加
    st.sidebar.markdown("---")

    if page == "🏠 ホーム":

        #! 関数を呼び出してCSSを適用
        apply_custom_css()
    
        #! アプリ立ち上げ時に分析ページの実行フラグを初期化（キャッシュのキーとして利用）
        step1_flag_analysis = 0
        step3_flag_analysiss = 0
        step3_flag_analysis = 0

        #! アプリ立ち上げ時に予測ページの実行フラグを初期化（キャッシュのキーとして利用）
        step1_flag_predict = 0
        step2_flag_predict = 0
        step3_flag_predict = 0
                
        #! 分析用の各ステップの実行フラグを保存
        save_flag_analysis(step1_flag_analysis, step3_flag_analysiss, step3_flag_analysis)

        #! 予測用の各ステップの実行フラグを保存
        save_flag_predict(step1_flag_predict, step2_flag_predict, step3_flag_predict)
        
        #! トップページのタイトル表示
        st.title("在庫分析アプリ（トライ版）")
        
        #!　更新履歴用の日付とメッセージのデータを作成
        data = {
            "日付": ["2024年9月30日（月）", ""],
            "メッセージ　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　": ["トライ用アプリを公開しました", ""]
        }

        #! pandasデータフレームを作成
        df = pd.DataFrame(data)

        st.write("\n\n")
        st.subheader("**🚩 ナビゲーション**")
        st.info("**👈 左側のサイドバーで各種機能を使用できます。詳細は📖 マニュアルをご参照ください。**")
        st.write("・🏠 ホーム：アプリについての情報を確認できます。")
        st.write("・⏳ 予測（準備中）：在庫リミット計算を行うことができます。")
        st.write("・📊 分析：在庫変動の要因分析を行うことができます。")
        st.write("・📖 マニュアル：本アプリの使用方法を確認できます。")

        # Streamlitでデータフレームを表示
        st.write("\n\n")
        st.subheader("**🆕 更新履歴**")
        st.dataframe(df)
    
    elif page == "在庫リミット計算":
        forecast_page()

    elif page == "在庫予測":
        st.write("開発中")

    elif page == "要因分析":
        analysis_page()

    elif page == "🔍 可視化":

        #from plotly.subplots import make_subplots
        #import plotly.graph_objects as go

        #! 関数を呼び出してCSSを適用
        apply_custom_css()

        # データを読み込む（Shift_JISエンコード）
        #file_path = '中間成果物/所在管理MBデータ_統合済&特定日時抽出済.csv'
        file_path = '中間成果物/所在管理MBデータ_統合済&特定日時抽出済.csv'
        df = pd.read_csv(file_path, encoding='shift_jis')

        # タイムスタンプ関連の列を抽出
        df_filtered = df[['品番', '納入日', '発注〜印刷LT', '発注〜検収LT', '発注〜順立装置入庫LT', '発注〜順立装置出庫LT', '発注〜回収LT', 
                '発注日時', '印刷日時', '検収日時', '順立装置入庫日時', '順立装置出庫日時', '回収日時']].copy()

        
        # Streamlit アプリケーション
        st.title('かんばん数の可視化（アニメーション、複数品番対応）')

        # 指定時刻が範囲内にあるかんばん数を計算する関数
        def count_kanban_between(df, start_col, end_col, target_time):
            return df[(df[start_col] <= target_time) & (df[end_col] >= target_time)].shape[0]
        
        # 時刻を datetime 型に変換
        df_filtered['発注日時'] = pd.to_datetime(df_filtered['発注日時'], errors='coerce')
        df_filtered['印刷日時'] = pd.to_datetime(df_filtered['印刷日時'], errors='coerce')
        df_filtered['検収日時'] = pd.to_datetime(df_filtered['検収日時'], errors='coerce')
        df_filtered['順立装置入庫日時'] = pd.to_datetime(df_filtered['順立装置入庫日時'], errors='coerce')
        df_filtered['順立装置出庫日時'] = pd.to_datetime(df_filtered['順立装置出庫日時'], errors='coerce')
        df_filtered['回収日時'] = pd.to_datetime(df_filtered['回収日時'], errors='coerce')

        # 複数の品番を選択可能にする
        品番選択肢 = df_filtered['品番'].unique()
        選択された品番 = st.multiselect('品番を選択してください（複数選択可）', 品番選択肢)

        # データを選択された品番にフィルタリング
        df_filtered = df_filtered[df_filtered['品番'].isin(選択された品番)]

        # 開始日と時間を選択できるようにする
        開始日 = st.date_input('開始日を選択してください', pd.to_datetime('2023-10-31'), key="start_date_input_unique")
        開始時間 = st.time_input('開始時間を選択してください', pd.to_datetime('11:00').time(), key="start_time_input_unique")

        # 終了日と時間を選択できるようにする
        終了日 = st.date_input('終了日を選択してください', pd.to_datetime('2023-10-31'), key="end_date_input_unique")
        終了時間 = st.time_input('終了時間を選択してください', pd.to_datetime('14:00').time(), key="end_time_input_unique")

        # 開始日時と終了日時を作成
        開始日時 = pd.to_datetime(f'{開始日} {開始時間}')
        終了日時 = pd.to_datetime(f'{終了日} {終了時間}')

        # 1時間ごとに時間範囲を作成
        時間範囲 = pd.date_range(start=開始日時, end=終了日時, freq='H')

        # 各時間、各品番でのかんばん数を集計
        kanban_counts_per_hour = []

        for target_time in 時間範囲:
            for 品番 in 選択された品番:
                # 各関所でのかんばん数を集計
                発注_印刷_かんばん数 = count_kanban_between(df_filtered[df_filtered['品番'] == 品番], '発注日時', '印刷日時', target_time)
                印刷_検収_かんばん数 = count_kanban_between(df_filtered[df_filtered['品番'] == 品番], '印刷日時', '検収日時', target_time)
                検収_入庫_かんばん数 = count_kanban_between(df_filtered[df_filtered['品番'] == 品番], '検収日時', '順立装置入庫日時', target_time)
                入庫_出庫_かんばん数 = count_kanban_between(df_filtered[df_filtered['品番'] == 品番], '順立装置入庫日時', '順立装置出庫日時', target_time)
                出庫_回収_かんばん数 = count_kanban_between(df_filtered[df_filtered['品番'] == 品番], '順立装置出庫日時', '回収日時', target_time)

                # 1時間ごとのデータを追加
                kanban_counts_per_hour.append({
                    '品番': 品番,
                    '時間': target_time.strftime('%Y-%m-%d %H:%M'),
                    '発注ー印刷': 発注_印刷_かんばん数,
                    '印刷ー検収': 印刷_検収_かんばん数,
                    '検収ー入庫': 検収_入庫_かんばん数,
                    '入庫ー出庫': 入庫_出庫_かんばん数,
                    '出庫ー回収': 出庫_回収_かんばん数
                })

        # DataFrameに変換
        df_kanban_counts = pd.DataFrame(kanban_counts_per_hour)

        # データの中身を確認する
        st.write(df_kanban_counts.head())

        # Plotlyを使ってアニメーションを作成（品番ごとに色分け）
        fig = px.bar(df_kanban_counts.melt(id_vars=['時間', '品番'], var_name='関所', value_name='かんばん数'),
                    x='関所', y='かんばん数', color='品番', animation_frame='時間',
                    range_y=[0, df_kanban_counts[['発注ー印刷', '印刷ー検収', '検収ー入庫', '入庫ー出庫', '出庫ー回収']].values.max()],
                    title=f'選択された品番ごとのかんばん数の変化')

        # Streamlitで表示
        st.plotly_chart(fig)

        #--------------------------------------------------------------------------------------------------------------------------

    elif page == "📖 マニュアル":

        #! 関数を呼び出してCSSを適用
        apply_custom_css()

        st.title("マニュアル")

        # 表示するPDFファイルのパス
        pdf_file_path = "sample.pdf"  # ここに表示したいPDFのパスを指定
        doc = fitz.open(pdf_file_path)

        # PDFの各ページを画像に変換して表示
        for page_number in range(doc.page_count):
            page = doc.load_page(page_number)  # ページを読み込む
            pix = page.get_pixmap()  # ピクセルマップを取得
            img = pix.tobytes("png")  # 画像としてバイトデータに変換
            st.image(img, caption=f"ページ {page_number + 1}", use_column_width=True)
    #else:
        #st.title("ページを選択してください。")
        

#! 本スクリプトが直接実行されたときに実行
if __name__ == "__main__":
    print("プログラムが実行中です")
    main()
