#メイン用

#! ライブラリのimport
import streamlit as st
import time
# ページ設定: 名前とアイコンを変更
st.set_page_config(
    page_title="在庫管理補助システム",  # ここに新しい名前を設定
    page_icon="🌊",              # 波アイコン（または他のアイコン）
)
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
from functions_v3 import display_message

#! 自作ライブラリのimport
from read_v3 import read_data, process_Activedata, read_syozailt_by_using_archive_data, read_activedata_by_using_archive_data,\
      read_zaiko_by_using_archive_data, calculate_supplier_truck_arrival_types2

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
            max-width: 80%;
            margin-left: auto;
            margin-right: auto;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

#-----------------------------------------------------------------------------------------------------------------------------------

#! リミット計算ページ
def forecast_page():

    # ページタイトル
    st.title("在庫リミット計算")
    display_message("**この画面では、数時間先の在庫を計算することができます。実行する際は左側のサイドバーで各種設定を行ってください。**","user")

    # 折り返し線を追加
    st.markdown("---")

    # カスタムCSSを適用して画面サイズを設定する
    apply_custom_css()

    # session_stateに初期値が入っていない場合は作成
    if "product_limit" not in st.session_state:
        st.session_state.product_limit = None

    # session_stateに初期値が入っていない場合は作成
    now = datetime.now()
    if "start_date_limit_count" not in st.session_state:
        st.session_state.start_date_limit_count = 0 # datetime(now.year, now.month, now.day, now.hour, 0, 0, 0) #現在時間をデフォルト値

    # session_stateに初期値が入っていない場合は作成
    if "selected_zaiko_count" not in st.session_state:
        st.session_state.selected_zaiko_count = 0  # デフォルト値に設定
 
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
        unique_product = st.selectbox("品番を選択してください", hinban_seibishitsu_df['品番_整備室'])
        
        # 「適用」ボタンをフォーム内に追加
        submit_button_step1 = st.form_submit_button(label='登録する')

    #! 適用ボタンが押されたときの処理
    if submit_button_step1 == True:

        st.sidebar.success(f"新たに選択された品番: {unique_product}")
        
        st.session_state.product_limit = unique_product
        
        #!　品番情報を表示
        display_hinban_info(unique_product)

        # 折り返し線を追加
        st.markdown("---")

    elif ("product_limit" in st.session_state) and (st.session_state.product_limit != None):
        st.sidebar.success(f"過去に選択した品番: {unique_product}")

        #!　品番情報を表示
        display_hinban_info(unique_product)

        # 折り返し線を追加
        st.markdown("---")

    #! 適用ボタンが押されなかったときの処理
    else:
        st.sidebar.warning("品番を選択してください")

    
    #!-------------------------------------------------------------------------------
    #! 予測ページのステップ2のサイドバータイトル
    #!-------------------------------------------------------------------------------
    st.sidebar.title("ステップ２：日時選択")
    
    default_values = {
        'start_date': datetime.now().date(),
        'start_time': now.replace(minute=0, second=0, microsecond=0),
        #'end_time': datetime.strptime("23:00", "%H:%M").time(),  # 23:00として初期化
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

        st.sidebar.success(f"開始日時: {start_datetime}")

        st.session_state.start_date_limit_count = 1

    elif ("start_date_limit_count" in st.session_state) and (st.session_state.start_date_limit_count != 0):
        st.sidebar.success(f"過去に選択した開始日時: {start_datetime}")
        
    # ボタンを押されなかった時       
    else:
        st.sidebar.warning("開始日、開始時間を選択し、登録するボタンを押してください。")

    
    #!-------------------------------------------------------------------------------
    #! 予測ページのステップ3のサイドバータイトル
    #!-------------------------------------------------------------------------------
    st.sidebar.title("ステップ３：在庫数入力")

    LT = 5
    if st.session_state.start_date_limit_count == 0:

        zaiko_teian = 0

    else:
        
        if unique_product:

            product = unique_product.split('_')[0]
            seibishitsu = unique_product.split('_')[1]

            # todo 引数関係なく全データ読み込みしてる
            zaiko_df = read_zaiko_by_using_archive_data(start_datetime.strftime('%Y-%m-%d-%H'), start_datetime.strftime('%Y-%m-%d-%H'))
            # todo
            #! 品番列を昇順にソート
            zaiko_df = zaiko_df.sort_values(by='品番', ascending=True)
            #! 無効な値を NaN に変換
            zaiko_df['拠点所番地'] = pd.to_numeric(zaiko_df['拠点所番地'], errors='coerce')
            #! 品番ごとに欠損値（NaN）を埋める(前方埋め後方埋め)
            zaiko_df['拠点所番地'] = zaiko_df.groupby('品番')['拠点所番地'].transform(lambda x: x.fillna(method='ffill').fillna(method='bfill'))
            #! それでも置換できないものはNaN を 0 で埋める
            zaiko_df['拠点所番地'] = zaiko_df['拠点所番地'].fillna(0).astype(int).astype(str)
            #! str型に変換
            zaiko_df['拠点所番地'] = zaiko_df['拠点所番地'].astype(int).astype(str)
            #! 受入場所情報準備
            file_path = 'temp/マスター_品番&仕入先名&仕入先工場名.csv'
            syozaikyotenchi_data = pd.read_csv(file_path, encoding='shift_jis')
            #! 空白文字列や非数値データをNaNに変換
            syozaikyotenchi_data['拠点所番地'] = pd.to_numeric(syozaikyotenchi_data['拠点所番地'], errors='coerce')
            #! str型に変換
            syozaikyotenchi_data['拠点所番地'] = syozaikyotenchi_data['拠点所番地'].fillna(0).astype(int).astype(str)
            #! 受入場所追加
            zaiko_df = pd.merge(zaiko_df, syozaikyotenchi_data[['品番','拠点所番地','受入場所','仕入先工場名']], on=['品番', '拠点所番地'], how='left')
            #st.dataframe(zaiko_df.head(100))
            #! 日付列を作成
            zaiko_df['日付'] = zaiko_df['日時'].dt.date
            #! 品番_受入番号作成
            zaiko_df['品番_受入場所'] = zaiko_df['品番'].astype(str) + "_" + zaiko_df['受入場所'].astype(str)
            zaiko_df = zaiko_df[(zaiko_df['品番'] == product) & (zaiko_df['受入場所'] == seibishitsu)]
            zaiko_extracted = zaiko_df[['日時', '在庫数（箱）']]
            # '日時' 列でデータをソート
            zaiko_extracted = zaiko_extracted.sort_values(by=['日時'])
            # 在庫数（箱）が NULL の場合、前の時間の在庫数（箱）で補完
            #lagged_features['在庫数（箱）'] = lagged_features.groupby('品番')['在庫数（箱）'].transform(lambda x: x.fillna(method='ffill'))
            zaiko_extracted = zaiko_extracted[zaiko_extracted['日時'] == start_datetime]
            #st.write(zaiko_extracted['在庫数（箱）'].iloc[0])
            if len(zaiko_extracted['在庫数（箱）']) != 0:
                zaiko_teian = int(zaiko_extracted['在庫数（箱）'].iloc[0])
            else:
                zaiko_teian = 0
            #st.write(zaiko_teian)

            arrival_times_df = calculate_supplier_truck_arrival_types2()
            arrival_times_df = arrival_times_df[
                (arrival_times_df['仕入先名'].isin(zaiko_df['仕入先名'])) &
                (arrival_times_df['発送場所名'].isin(zaiko_df['仕入先工場名']))
            ]

            #st.write(arrival_times_df)

            LT = int(arrival_times_df["LT"].iloc[0])

            #st.write(LT)

        else:
            zaiko_teian = 0

    # フォーム作成
    with st.sidebar.form("date_selector_form"):
        # 日時選択用セレクトボックス
        selected_zaiko = st.selectbox("工場内の在庫数（箱）を入力してください", list(range(0,100)), index = zaiko_teian,
                                      help="現在在庫を参考にして、在庫数を選んでください。")
        submit_button_step3 = st.form_submit_button("登録する")

    # ボタンが押された時
    if submit_button_step3:
        
        st.sidebar.success(f"入力された在庫数: {selected_zaiko}")#、在庫数（箱）：{int(zaikosu)}")

        st.session_state.selected_zaiko_count = 1

    elif ("selected_zaiko_count" in st.session_state) and (st.session_state.selected_zaiko_count != 0):
        st.sidebar.success(f"過去に選択した開始日時: {selected_zaiko}")
            
    # ボタンが押されなかった時
    else:
        st.sidebar.warning("在庫数を入力してください")

    #!-------------------------------------------------------------------------------
    #! 予測ページのステップ3のサイドバータイトル
    #!-------------------------------------------------------------------------------
    st.sidebar.title("ステップ４：需要調整")

    # フォームの作成
    with st.sidebar.form("zyuyo_form"):
        st.write("日量をご選択ください")

        # 2つのカラムを横並びに作成
        col1, col2 = st.columns(2)

        # 各カラムにボタンを配置
        with col1:
            btn1 = st.form_submit_button("日量を採用する",help="通常の日量を使用する")

        with col2:
            btn2 = st.form_submit_button("日量MAXを採用する",help="最大値の日量を使用する（生産数が多い場合で計算したい）")

    # フォームの送信処理
    if btn1:
        st.sidebar.success("日量が採用されました")
        forecast_v3.show_forecast(unique_product,start_datetime,selected_zaiko, LT, 0)

    if btn2:
        st.sidebar.success("日量MAXが採用されました")
        forecast_v3.show_forecast(unique_product,start_datetime,selected_zaiko, LT, 1)

    # 両方のボタンが押されていなかった場合のメッセージ
    if not btn1 and not btn2:
        st.sidebar.warning("日量をご選択ください") 

    
#! 在庫シミュレーション
def zaiko_simulation_page():

    #! ページタイトル
    st.title("在庫予測シミュレーション")
    display_message("**この画面では、24時間先の在庫予測を行うことができます。実行する際は左側のサイドバーで各種設定を行ってください**","user")

    #! カスタムCSSを適用して画面サイズを設定する
    apply_custom_css()
 
    # #! サイドバートップメッセージ
    # st.sidebar.write("## 操作バー")
    
    #!-------------------------------------------------------------------------------
    #! サイドバー設定
    #!-------------------------------------------------------------------------------

    # # （テスト用）変数リセット
    if st.sidebar.button("初期値を現在時間に設定（キャッシュ削除）"):
        st.session_state.clear()

    # session_stateに初期値が入っていない場合は作成
    if "start_date" not in st.session_state:
        st.session_state.start_date = datetime.now()  # 現在の日付をデフォルト値に設定

    if "start_time" not in st.session_state:
        current_time = datetime.now().time()
        st.session_state.start_time = dt_time(current_time.hour, 0)  # 現在の時間（分は0にリセット）

    if "start_datetime" not in st.session_state:
        st.session_state.start_datetime = ""

    if "change_rate" not in st.session_state:
        st.session_state.change_rate = 0

    # 折り畳み可能なメッセージ

    st.sidebar.title("シミュレーション設定")
    with st.sidebar.form(key='form_start_datetime'):

        # 開始日
        st.session_state.start_date = st.date_input("開始日",
                                                     st.session_state.start_date,
                                                     help="初期設定は現在日です")
        
        # 開始時間の選択肢をセレクトボックスで提供
        hours = [f"{i:02d}:00" for i in range(24)]
        start_time_str = st.selectbox("開始時間", hours,
                                       index=st.session_state.start_time.hour,
                                       help="初期設定は現在時間です")
        
        # 選択された時間をdt_timeオブジェクトに変換
        start_time_hours = int(start_time_str.split(":")[0])

        # 時間を更新
        st.session_state.start_time = dt_time(start_time_hours, 0)

        # フォームの送信ボタン
        submit_button_step1 = st.form_submit_button(label='登録する')
    
        # 開始日時と終了日時を結合
        start_datetime = datetime.combine(st.session_state.start_date, st.session_state.start_time)
    
    # ボタンを押された時
    if submit_button_step1:
    
        # フォームを送信したらsession_stateに保存
        st.session_state.start_datetime = start_datetime
        st.sidebar.success(f"選択した日時：{st.session_state.start_datetime}")

        forecast_v3.show_zaiko_simulation( st.session_state.start_datetime, st.session_state.change_rate)

    # ボタンを押されなかったが、過去にボタンが押され変数に値が保存されているとき       
    elif ("start_datetime" in st.session_state) and (st.session_state.start_datetime != ""):
        st.sidebar.success(f"選択した日時：{st.session_state.start_datetime}")
    
    # それ以外
    else:
        st.sidebar.warning("開始時間を入力し、「登録する」ボタンを押してください")

    # st.sidebar.title("ステップ２：変動率選択")
    # with st.sidebar.form(key='form_change_rate'):

    #     # number_inputの引数で範囲や刻み幅を指定できます
    #     selected_value = st.number_input(
    #         "変動率を選択",
    #         min_value=0.0,
    #         max_value=2.0,
    #         value=1.0,  # デフォルト値
    #         step=0.1
    #     )
        
    #     submit_button_step2 = st.form_submit_button("登録する")

    # # ボタンが押されたとき
    # if submit_button_step2:

    #     st.session_state.change_rate = selected_value
    #     st.sidebar.success(f"新しく選した変動率: {st.session_state.change_rate}")

    #     forecast_v3.show_zaiko_simulation( st.session_state.start_datetime, st.session_state.change_rate)
        
    # # ボタンを押されなかったが、過去にボタンが押され変数に値が保存されているとき
    # elif ("change_rate" in st.session_state) and (st.session_state.change_rate != 0):
    #     st.sidebar.success(f"過去に選択した変動率{st.session_state.change_rate}")

    # # それ以外
    # else:
    #     st.sidebar.warning("フレ率を入力し、「登録する」ボタンを押してください")

#-----------------------------------------------------------------------------------------------------------------------------------
#! 要因分析ページ            
def analysis_page():

    st.title("在庫変動要因分析")

    #! 説明
    display_message("**この画面では、在庫変動の要因分析を行うことができます。実行する際は左側のサイドバーで各種設定を行ってください**","user")

    # カスタムCSSを適用して画面サイズを設定する
    apply_custom_css()

    #*---------------------------------------------------------------------------------------------

    step0_flag_analysis = 0

    # 分析用の各ステップの実行フラグを読み込む
    step1_flag_analysis, step2_flag_analysis, step3_flag_analysis = load_flag_analysis()
    
    # 確認用
    # フラグ状態どうなっている？
    #st.sidebar.success(f"{step1_flag_analysis}")
    #st.sidebar.success(f"{step3_flag_analysis}")
    #st.sidebar.success(f"{step3_flag_analysis}")

    st.sidebar.write("## 🔥各ステップを順番に実行してください🔥")

    #! ステップ0
    st.sidebar.title("ステップ０：異常の確認（任意）")

    # フォーム作成
    with st.sidebar.form(key='analysis_step0_form'):

        # 日付入力
        selected_date = st.date_input("日付を選択してください", value=datetime.now().date())

        # 1時間ごとの選択肢を作成
        hourly_times = [f"{hour:02d}:00" for hour in range(24)]  # 00:00～23:00の時間リスト

        # 現在の時刻を取得
        current_time = datetime.now()

        # 現在の時刻の次の1時間を計算
        current_hour = (current_time).replace(minute=0, second=0, microsecond=0)

        # 次の1時間のインデックスを計算
        default_time_index = current_hour.hour

        # 時間選択（1時間ごと）
        selected_time = st.selectbox("時間を選択してください", hourly_times, index=default_time_index)  # デフォルトを8:00に設定

        # フォームの送信ボタン
        submit_button_step0 = st.form_submit_button("登録する")

    # 適用ボタンが押されたときの処理
    if submit_button_step0 == True: 

        #! 在庫上下限フレの計算（デモ用）-----------------------------------------------------------------

        #折り返し線を追加
        st.markdown("---")

        step0_flag_analysis = 1
        
        with st.spinner("実行中です。しばらくお待ちください..."):
            analysis_v3.show_abnormal( selected_date, selected_time)

        #折り返し線を追加
        st.markdown("---")

        st.sidebar.info("🎉 処理が完了しました！")

    # 適用ボタンが押されなかったときの処理
    else:
        
        # まだ一度もSTEP1が実行されていない時
        if step0_flag_analysis == 0:
            st.sidebar.code("このステップは任意です。スキップできます。")

        #1度はボタン押されている
        elif step0_flag_analysis == 1:
            st.sidebar.success(f"過去に選択された品番: {product}")

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
        #! 学習
        with st.spinner("実行中です。しばらくお待ちください..."):
            data, rf_model, rf_model2, rf_model3, X = analysis_v3.show_analysis(product)
        st.sidebar.success("処理が完了しました！")
        #data, rf_model, rf_model2, rf_model3, X = analysis_v3.show_analysis(product)
        #data, rf_model2, X= analysis_v3.show_analysis(product, '2024-05-01-00', '2024-08-31-00')
        #data, rf_model3, X= analysis_v3.show_analysis(product, '2024-05-01-00', '2024-08-31-00')

        # #!　全品番動作テスト
        # for product_i in hinban_seibishitsu_df['品番_整備室']:
        #     part_number = product_i.split('_')[0]
        #     seibishitsu = product_i.split('_')[1]
        #     if part_number == "01912ECB040":
        #         break
        #     data, rf_model, rf_model2, rf_model3, X= analysis_v3.show_analysis(product_i)

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

        st.sidebar.info("🎉 処理が完了しました！")

    # 適用ボタンが押されなかったときの処理
    else:
        
        # まだ一度もSTEP1が実行されていない時
        if step1_flag_analysis == 0:
            st.sidebar.warning("品番を選択し、「登録する」ボタンを押してください")

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

            st.sidebar.info("🎉 処理が完了しました！")
            
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

        st.sidebar.info("🎉 処理が完了しました！")
    
    elif (step2_flag_analysis == 0) or (step3_flag_analysis == 0) or (step2_flag_analysis == 1):
        st.sidebar.warning("要因分析の結果を表示する日時を選択し、「登録する」ボタンを押してください")

#-----------------------------------------------------------------------------------------------------------------------------------

#! 全体ページ構成
def main():

    #! サイドバーの定義
    # タイトル
    st.sidebar.title("メインメニュー")
    # メインメニューの選択
    main_menu = st.sidebar.radio("ページ選択", ["🏠 ホーム", "🔍 可視化（準備中）", "📊 分析", "⏳ 予測（準備中）", "📖 マニュアル"])

    #ページ変数の初期化
    #page = None

    # メインメニューの選択に応じた処理
    if main_menu == "🏠 ホーム":
        page = "🏠 ホーム"
    elif main_menu == "🔍 可視化（準備中）":
        #page = "🔍 可視化"
        main_menu_visual = st.sidebar.radio("可視化ページ選択", ["上下限外れ確認","関所別かんばん数可視化（アニメーション）","フレ可視化"], key='analysis')
        page = main_menu_visual
    elif main_menu == "📊 分析":
        # 分析のサブメニュー
        main_menu_analysis = st.sidebar.radio("分析ページ選択", ["要因分析"], key='analysis')
        page = main_menu_analysis
    elif main_menu == "⏳ 予測（準備中）":
        # 予測のサブメニュー
        main_menu_prediction = st.sidebar.radio("予測ページ選択", ["在庫リミット計算", "在庫予測","在庫シミュレーション（仮名）"], key='prediction')
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
        with open("draw_wave.html", "r", encoding="utf-8") as f:
            html_content = f.read()
        # Streamlit にHTMLコードを埋め込み
        # 高さだけを指定するだけで、横はブラウザの大きさに追従する

        html_code = """
        <!DOCTYPE html>
        <html lang="ja">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Gemini Embedding Model Blog Thumbnail</title>
            <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
            <style>
                html, body {
                    width: 100%;
                    height: 100%;
                    margin: 0;
                    padding: 0;
                    background: transparent;
                    font-family: 'Helvetica Neue', Arial, sans-serif;
                    display: flex;
                    justify-content: center;
                    align-items: center;
                }

                .thumbnail-container {
                    width: 99vw;
                    height: 99vh;
                    background-color: #ffffff;
                    border-radius: 12px;
                    overflow: hidden;
                    box-shadow: 0 15px 25px rgba(0,0,0,0.1);
                    position: relative;
                    display: flex;
                }

                .content-area {
                    flex: 3;
                    padding: 40px;
                    display: flex;
                    flex-direction: column;
                    justify-content: center;
                    position: relative;
                    z-index: 2;
                }

                .visual-area {
                    flex: 3;
                    background-color: #005eff;
                    display: flex;
                    align-items: center;
                    justify-content: center;
                    position: relative;
                    overflow: hidden;
                }

                .title {
                    font-size: 3vw;
                    font-weight: 800;
                    color: #005eff;
                    margin-bottom: 2vh;
                    line-height: 1.2;
                }

                .subtitle {
                    font-size: 1.5vw;
                    color: #000000;
                    margin-bottom: 3vh;
                    font-weight: 500;
                }

                .feature-list {
                    display: flex;
                    flex-wrap: wrap;
                    gap: 1vw;
                    margin-top: 1vh;
                }

                .feature-item {
                    background-color: rgba(0, 207, 255, 0.1);
                    color: #000000;
                    padding: 0.5vw 1vw;
                    border-radius: 50px;
                    font-size: 1vw;
                    display: flex;
                    align-items: center;
                    gap: 0.5vw;
                    border: 1px solid rgba(0, 207, 255, 0.3);
                }

                .feature-item i {
                    color: #00cfff;
                }

                .blog-tag {
                    position: absolute;
                    top: 1vh;
                    left: 1vw;
                    background-color: #0099ff;
                    color: #fff;
                    padding: 0.5vh 1vw;
                    border-radius: 4px;
                    font-size: 0.8vw;
                    font-weight: 700;
                    text-transform: uppercase;
                    letter-spacing: 1px;
                }

                .accent-line {
                    position: absolute;
                    height: 100%;
                    width: 0.5vw;
                    background: linear-gradient(to bottom, #00cfff, #0099ff, #005eff);
                    left: 0;
                    top: 0;
                }

                /* ここからはエフェクトやパーティクルの設定（省略せずにそのまま） */
                .abstract-container {
                    position: absolute;
                    width: 100%;
                    height: 100%;
                    overflow: hidden;
                }

                .node, .connection, .particle, .wave {
                    position: absolute;
                }

                /* ノード、コネクション、パーティクル、波動のアニメーションは省略してない！ */
                .node {
                    background-color: rgba(255, 255, 255, 0.7);
                    border-radius: 50%;
                    transform-origin: center;
                }

                /* ノードの位置やアニメーション（省略せずにフル） */
                .node:nth-child(1) { width: 1vw; height: 1vw; top: 30%; left: 40%; animation: pulse 3s infinite alternate, float 8s infinite linear; }
                .node:nth-child(2) { width: 1.5vw; height: 1.5vw; top: 60%; left: 60%; animation: pulse 4s infinite alternate-reverse, float 12s infinite linear reverse; }
                .node:nth-child(3) { width: 1vw; height: 1vw; top: 20%; left: 70%; animation: pulse 5s infinite alternate, float 10s infinite linear; }
                .node:nth-child(4) { width: 1.2vw; height: 1.2vw; top: 70%; left: 20%; animation: pulse 3.5s infinite alternate-reverse, float 11s infinite linear reverse; }
                .node:nth-child(5) { width: 0.8vw; height: 0.8vw; top: 40%; left: 80%; animation: pulse 4.5s infinite alternate, float 9s infinite linear; }

                @keyframes pulse {
                    0% { transform: scale(1); box-shadow: 0 0 0.5vw rgba(255, 255, 255, 0.6); }
                    100% { transform: scale(1.5); box-shadow: 0 0 1vw rgba(255, 255, 255, 0.8); }
                }

                @keyframes float {
                    0% { transform: translate(0, 0); }
                    25% { transform: translate(2vw, 1.5vw); }
                    50% { transform: translate(0.5vw, -1.5vw); }
                    75% { transform: translate(-2vw, 1vw); }
                    100% { transform: translate(0, 0); }
                }
            </style>
        </head>
        <body>
            <div class="thumbnail-container">
                <div class="accent-line"></div>
                <div class="blog-tag">Made with love in Kariya by DS部</div>

                <div class="content-area">
                    <h1 class="title">在庫管理補助システム</h1>
                    <p class="subtitle">主な機能について</p>

                    <div class="feature-list">
                        <div class="feature-item"><i class="fas fa-robot"></i><span>？</span></div>
                        <div class="feature-item"><i class="fas fa-chart-line"></i><span>予測</span></div>
                        <div class="feature-item"><i class="fas fa-code"></i><span>？</span></div>
                        <div class="feature-item"><i class="fas fa-brain"></i><span>分析</span></div>
                    </div>
                </div>

                <div class="visual-area">
                    <div class="abstract-container">
                        <div class="node"></div>
                        <div class="node"></div>
                        <div class="node"></div>
                        <div class="node"></div>
                        <div class="node"></div>
                    </div>
                </div>
            </div>
        </body>
        </html>
        """

        st.components.v1.html(html_code, height=800, scrolling=True)
    
        # HTMLコード（ボタンとクリック時の吹き出しメッセージ）
        html_code = """
        <div style="text-align: center;">
            <!-- ボタン1 -->
            <button id="infoButton1" style="
                font-size: 20px;
                padding: 10px 20px;
                background-color: #ffffff; /* ボタンの背景色を白 */
                color: #007BFF; /* ボタンの文字色を青 */
                border: 2px solid #007BFF; /* ボタンの枠線 */
                border-radius: 5px;
                cursor: pointer;
                margin: 10px;
            ">
                🛈 機能の説明を表示
            </button>

            <!-- ボタン2 -->
            <button id="infoButton2" style="
                font-size: 20px;
                padding: 10px 20px;
                background-color: #ffffff; /* ボタンの背景色を白 */
                color: #007BFF; /* ボタンの文字色を青 */
                border: 2px solid #007BFF; /* ボタンの枠線 */
                border-radius: 5px;
                cursor: pointer;
                margin: 10px;
            ">
                🛈 更新履歴を表示
            </button>

            <!-- 吹き出しメッセージ1 -->
            <div id="tooltip1" style="
                display: none;
                margin-top: 10px;
                background-color: #ffffff; /* 背景色をグレー */
                color: #000; /* 文字色を黒 */
                padding: 2px 2px;
                border-radius: 10px;
                box-shadow: 0 2px 6px rgba(0, 0, 0, 0.2);
                font-size: 20px;
                text-align: left; /* 左詰め */
                white-space: pre-wrap;
            ">
            ・🏠 ホーム：アプリについての情報を確認できます。
            ・🔍 可視化（準備中）：在庫状況を確認することができます。
            ・📊 分析：在庫変動の要因分析を行うことができます。
            ・⏳ 予測（準備中）：在庫リミット計算を行うことができます。
            ・📖 マニュアル：本アプリの使用方法を確認できます。
            </div>

            <!-- 吹き出しメッセージ2 -->
            <div id="tooltip2" style="
                display: none;
                margin-top: 10px;
                background-color: #ffffff; /* 背景色を白 */
                color: #000; /* 文字色を黒 */
                padding: 10px 15px;
                border-radius: 10px;
                box-shadow: 0 2px 6px rgba(0, 0, 0, 0.2);
                font-size: 14px;
                text-align: left;
                white-space: pre-wrap;
            ">
                🔍 これはボタン2の追加情報です。
            </div>
        </div>

        <script>
            // ボタン1とボタン2、対応する吹き出し
            const infoButton1 = document.getElementById("infoButton1");
            const infoButton2 = document.getElementById("infoButton2");
            const tooltip1 = document.getElementById("tooltip1");
            const tooltip2 = document.getElementById("tooltip2");

            // ボタン1のクリックイベント
            infoButton1.addEventListener("click", () => {
                tooltip1.style.display = "block"; // ボタン1の吹き出しを表示
                tooltip2.style.display = "none"; // ボタン2の吹き出しを非表示
            });

            // ボタン2のクリックイベント
            infoButton2.addEventListener("click", () => {
                tooltip2.style.display = "block"; // ボタン2の吹き出しを表示
                tooltip1.style.display = "none"; // ボタン1の吹き出しを非表示
            });
        </script>
        """

        # StreamlitでHTMLを埋め込む
        #st.components.v1.html(html_code, height=400)
    
        # # Streamlitでデータフレームを表示
        # st.write("\n\n")
        # st.subheader("**🆕 更新履歴**")
        # st.dataframe(df)

        st.sidebar.header("管理メニュー")

        # 折り畳み可能なメッセージ
        with st.sidebar.expander("💡 ヘルプ "):
            st.write("ここに詳細情報を記載する。クリックすると折り畳み/展開が切り替わります。")
            #st.image("https://via.placeholder.com/150", caption="例画像")

        # 実行したい時刻をリストで設定（24時間表記）
        schedule_times = ["09:00", "12:00", "18:00"]

        def next_run_time(now, schedule_times):
            today = now.date()
            times_today = [datetime.strptime(f"{today} {t}", '%Y-%m-%d %H:%M') for t in schedule_times]
            
            # 次に来る時刻を探す
            for t in times_today:
                if now < t:
                    return t
            
            # 全部過ぎてたら翌日の一番最初の時間
            tomorrow = today + timedelta(days=1)
            return datetime.strptime(f"{tomorrow} {schedule_times[0]}", '%Y-%m-%d %H:%M')

        def save_random_to_csv():
            random_value = 10
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            
            df = pd.DataFrame([[timestamp, random_value]], columns=["timestamp", "random_value"])
            df.to_csv('yosoku_test/random_numbers.csv', mode='a', header=False, index=False)
            
            print(f"{timestamp} に {random_value} を保存しました。")

        # 初期化
        now = datetime.now()
        next_time = next_run_time(now, schedule_times)

        print(f"最初の実行予定時刻: {next_time}")

        #st.write(st.session_state.processing)
        
        # 処理状態を保存するセッション変数
        if 'processing' not in st.session_state:
            st.session_state.processing = False

        # st.write(st.session_state.processing)

        #!!------------------------------------------------------------------------------------------------------------

        #! オリジナル開始終了ボタン

        # cdnjs.cloudflare.comは読み込める、use.fontawesome.comだと読み込めない（古い & 非推奨 & 不安定らしい）
        # st.sidebar.markdown("""
        #     <!-- Font Awesome 読み込み -->
        #     <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
        #     <style>
        #         .description-text {
        #             font-size: 20px;       /* 👈 大きくした！ */
        #             margin-bottom: 15px;   /* ボタンとの余白をちょっと広げても◎ */
        #             color: #444;           /* 控えめなグレー */
        #             font-weight: bold;     /* 太字にして存在感を出しても◎ */
        #         }
        #         .button-container {
        #             display: flex;
        #             gap: 10px;          /* ボタン間の余白 */
        #             align-items: center;
        #             width: 100%;        /* 横幅いっぱいに広げる */
        #             flex-wrap: wrap;    /* 画面幅が狭い場合に折り返せるようにする */
        #         }
        #         /* 各フォームを等分し、横幅を可変に */
        #         .button-container form {
        #             flex: 1;            /* 横幅を均等配分 */
        #             min-width: 120px;   /* 小さすぎないように最小幅を設定 */
        #         }
        #         .custom-button {
        #             width: 100%;        /* form 内で100%にして可変幅 */
        #             padding: 8px 16px;
        #             font-size: 20px;
        #             font-weight: bold;
        #             text-align: center;
        #             text-decoration: none;
        #             color: #333;            /* 文字色(地味め) */
        #             background-color: #ddd; /* ボタン背景色(地味め) */
        #             border: none;
        #             border-radius: 5px;
        #             box-shadow: 0 2px #999;
        #             transition: all 0.3s ease;
        #         }
        #         .custom-button:hover {
        #             background-color: #ccc; /* ホバー時の背景色 */
        #         }
        #         .custom-button:active {
        #             box-shadow: 0 1px #666;
        #             transform: translateY(1px);
        #         }
        #     </style>
        #     <!-- 説明文 -->
        #     <div class="description-text">
        #         <i class="fa-regular fa-clock"></i>&nbsp;&nbsp;定期予測の設定
        #     </div>
        #     <div class="button-container">
        #         <!-- 実行ボタン -->
        #         <form action="" method="get">
        #             <button class="custom-button" type="submit" name="run_forecast" value="true">
        #                 <i class="fa-solid fa-circle" style="color: #3EB489;"></i>&nbsp;&nbsp;起動
        #             </button>
        #         </form>
        #         <!-- 停止ボタン -->
        #         <form action="" method="get">
        #             <button class="custom-button" type="submit" name="stop_forecast" value="true">
        #                 <i class="fa-solid fa-circle-xmark" style="color: #FF6347;"></i>&nbsp;&nbsp;停止
        #             </button>
        #         </form>
        #     </div>
        # """, unsafe_allow_html=True)

        # クエリパラメータを取得（ボタン押下を検出）
        # query_params_run_or_stop = st.query_params
        # # 予測を開始するボタンを押したら
        # if query_params_run_or_stop.get("run_forecast") == "true":
        #     #st.sidebar.success("在庫予測を実行中... 🚀")
        #     st.session_state.processing = True #
        #     query_params_run_or_stop.clear() #　これをすることでボタンを押したことを忘れる（ページ移動して戻ってきても自動で再開しないようにする）
        # # 予測を終了するボタンを押したら
        # elif query_params_run_or_stop.get("stop_forecast") == "true":
        #     #st.sidebar.info("予測は停止中 🛑")
        #     st.session_state.processing = False

        #!!------------------------------------------------------------------------------------------------------------

        # セッションステート初期化
        if 'processing' not in st.session_state:
            st.session_state.processing = False

        def start_processing():
            st.session_state.processing = True

        def stop_processing():
            st.session_state.processing = False

        with st.sidebar.form(key="control_form"):
            st.subheader("定期予測の設定")

            # ボタンを横に並べる
            col1, col2 = st.columns(2)

            with col1:
                start_btn = st.form_submit_button("🟢　開始　")
                if start_btn:
                    start_processing()

            with col2:
                stop_btn = st.form_submit_button("🔴　停止　")
                if stop_btn:
                    stop_processing()

            # 下にステータス表示
            if st.session_state.processing:
                st.info("定期予測を実行中です！停止する場合は「停止」を押してください。")
            else:
                st.warning("停止しています。開始するには「開始」を押してください。")

        # HTML/CSS/JavaScriptの定義
        html_template = """
        <!DOCTYPE html>
        <html lang="ja">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>在庫予測</title>
            <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
            <style>
                body {{
                    font-family: 'Helvetica Neue', Arial, sans-serif;
                    margin: 0;
                    padding: 0;
                    display: flex;
                    justify-content: center;
                    align-items: center;
                    overflow: hidden;
                    background: none;
                }}
                .container {{
                    display: flex;
                    width: 99%;
                }}
                .left-panel {{
                    text-align: center;
                    background: #ffffff;
                    padding: 40px;
                    border-radius: 12px;
                    box-shadow: 0 15px 25px rgba(0,0,0,0.1);
                    max-width: 300px;
                    margin-right: 20px;
                    display: flex;
                    flex-direction: column;
                    justify-content: center;
                    height: auto;
                    position: relative;
                    overflow: hidden;
                }}
                .right-panel {{
                    background: #ffffff;
                    color: black;
                    padding: 40px;
                    border-radius: 12px;
                    flex-grow: 1;
                    box-shadow: 0 15px 25px rgba(0,0,0,0.1);
                    position: relative;
                    overflow: hidden;
                }}
                h1 {{
                    font-size: 24px;
                    font-weight: 700;
                    color: #005eff;
                    margin-bottom: 10px;
                }}
                h2 {{
                    font-size: 20px;
                    margin-bottom: 20px;
                    color: #005eff;
                }}
                p {{
                    font-size: 16px;
                    color: #000000;
                    margin-bottom: 20px;
                }}
                .loader {{
                    position: relative;
                    width: 100px;
                    height: 100px;
                    margin: 0 auto 20px;
                    display: {display_loader};
                }}
                .loader::before,
                .loader::after {{
                    content: '';
                    position: absolute;
                    top: 0;
                    left: 0;
                    right: 0;
                    bottom: 0;
                    border-radius: 50%;
                    border: 4px solid transparent;
                    animation: spin 1.5s linear infinite;
                }}
                .loader::before {{
                    border-top-color: #00cfff;
                    border-right-color: #00cfff;
                }}
                .loader::after {{
                    border-bottom-color: #005eff;
                    border-left-color: #005eff;
                    animation-delay: -0.75s;
                }}
                @keyframes spin {{
                    0% {{
                        transform: rotate(0deg);
                    }}
                    100% {{
                        transform: rotate(360deg);
                    }}
                }}
                .timestamp {{
                    font-size: 14px;
                    color: #0099ff;
                    font-weight: 500;
                    margin-top: 15px;
                }}
                .timestamp-label {{
                    display: block;
                    color: #666;
                    font-size: 12px;
                    margin-bottom: 5px;
                }}
                .faq-item {{
                    margin-bottom: 15px;
                    padding-bottom: 12px;
                    border-bottom: 1px solid #eaeaea;
                }}
                .faq-item:last-child {{
                    border-bottom: none;
                    padding-bottom: 0;
                }}
                .faq-item strong {{
                    display: block;
                    margin-bottom: 5px;
                    color: #000000;
                    font-weight: 600;
                }}
                .accent-line {{
                    position: absolute;
                    height: 100%;
                    width: 0.5vw;
                    background: linear-gradient(to bottom, #00cfff, #0099ff, #005eff);
                    left: 0;
                    top: 0;
                }}
                .tag {{
                    position: absolute;
                    top: 1vh;
                    left: 1vw;
                    background-color: #0099ff;
                    color: #fff;
                    padding: 0.5vh 1vw;
                    border-radius: 4px;
                    font-size: 0.8vw;
                    font-weight: 700;
                    text-transform: uppercase;
                    letter-spacing: 1px;
                    z-index: 10;
                }}
                @media (max-width: 768px) {{
                    .container {{
                        flex-direction: column;
                    }}
                    .left-panel {{
                        max-width: none;
                        margin-right: 0;
                        margin-bottom: 20px;
                    }}
                    .tag {{
                        font-size: 12px;
                        padding: 3px 8px;
                    }}
                }}
                .abstract-container {{
                    position: absolute;
                    right: 0;
                    top: 0;
                    width: 100%;
                    height: 100%;
                    overflow: hidden;
                    opacity: 0.1;
                    z-index: 0;
                }}
                .node {{
                    position: absolute;
                    background-color: rgba(0, 94, 255, 0.7);
                    border-radius: 50%;
                    transform-origin: center;
                }}
                .node:nth-child(1) {{ width: 1vw; height: 1vw; top: 30%; left: 40%; animation: pulse 3s infinite alternate, float 8s infinite linear; }}
                .node:nth-child(2) {{ width: 1.5vw; height: 1.5vw; top: 60%; left: 60%; animation: pulse 4s infinite alternate-reverse, float 12s infinite linear reverse; }}
                .node:nth-child(3) {{ width: 1vw; height: 1vw; top: 20%; left: 70%; animation: pulse 5s infinite alternate, float 10s infinite linear; }}
                @keyframes pulse {{
                    0% {{ transform: scale(1); box-shadow: 0 0 0.5vw rgba(0, 94, 255, 0.6); }}
                    100% {{ transform: scale(1.5); box-shadow: 0 0 1vw rgba(0, 94, 255, 0.8); }}
                }}
                @keyframes float {{
                    0% {{ transform: translate(0, 0); }}
                    25% {{ transform: translate(2vw, 1.5vw); }}
                    50% {{ transform: translate(0.5vw, -1.5vw); }}
                    75% {{ transform: translate(-2vw, 1vw); }}
                    100% {{ transform: translate(0, 0); }}
                }}
            </style>
        </head>
        <body>
            <div class="container">
                <div class="left-panel">
                    <div class="accent-line"></div>
                    <div class="tag">{status_tag}</div>
                    <div class="loader"></div>
                    <h1>{main_title}</h1>
                    <p>{description}</p>
                    <div class="timestamp">
                        <span class="timestamp-label">開始日時:</span>
                        <span id="current-datetime">{current_datetime}</span>
                    </div>
                    <div class="abstract-container">
                        <div class="node"></div>
                        <div class="node"></div>
                        <div class="node"></div>
                    </div>
                </div>
                <div class="right-panel">
                    <h2>サポート情報・FAQ</h2>
                    <div class="faq-item">
                        <strong>Q: 在庫予測はどのくらい時間がかかりますか？</strong>
                        <span>A: 平均で約5分程度です。</span>
                    </div>
                    <div class="faq-item">
                        <strong>Q: 結果はどこで確認できますか？</strong>
                        <span>A: 処理が完了すると画面上に結果が表示されます。</span>
                    </div>
                    <div class="faq-item">
                        <strong>Q: 予測の精度はどのくらいですか？</strong>
                        <span>A: 過去データに基づき、90%以上の精度を達成しています。</span>
                    </div>
                    <div class="faq-item">
                        <strong>Q: エラーが発生した場合はどうすればよいですか？</strong>
                        <span>A: お手数ですが、サポートセンターまでご連絡ください。</span>
                    </div>
                </div>
            </div>
        </body>
        </html>
        """

        # 処理状態に応じてHTMLの内容を変更
        if st.session_state.processing:
            display_loader = "block"
            status_tag = "処理中"
            main_title = "在庫予測を定期実行中"
            description = "在庫予測は1時間毎に実行されます。"
            # 処理中の時だけ日時を取得
            now = datetime.now()
            formatted_datetime = now.strftime("%Y年%m月%d日 %H:%M")
        else:
            display_loader = "none"
            status_tag = "待機中"
            main_title = "在庫定期予測を停止中"
            description = "「開始」ボタンをクリックして処理を開始してください。"
            formatted_datetime = "まだ開始されていません"

        # テンプレートに変数を挿入
        html_content = html_template.format(
            display_loader=display_loader,
            status_tag=status_tag,
            main_title=main_title,
            description=description,
            current_datetime=formatted_datetime
        )

        # HTMLを表示
        st.components.v1.html(html_content, height=600)

        import os

        # 開きたいディレクトリのパス（例: CドライブのDocumentsフォルダ）
        folder_path = '定期予測結果'

        # 処理状態を保存するセッション変数
        # if 'processing' not in st.session_state:
        #     st.session_state.processing = False

        # cdnjs.cloudflare.comは読み込める、use.fontawesome.comだと読み込めない（古い & 非推奨 & 不安定らしい）
        # st.sidebar.markdown("""
        #     <!-- Font Awesome 読み込み -->
        #     <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
        #     <style>
        #         .description-text {
        #             margin-top: 15px;
        #             font-size: 20px;       /* 👈 大きくした！ */
        #             margin-bottom: 15px;   /* ボタンとの余白をちょっと広げても◎ */
        #             color: #444;           /* 控えめなグレー */
        #             font-weight: bold;     /* 太字にして存在感を出しても◎ */
        #         }
        #         .button-container {
        #             display: flex;
        #             gap: 10px;          /* ボタン間の余白 */
        #             align-items: center;
        #             width: 100%;        /* 横幅いっぱいに広げる */
        #             flex-wrap: wrap;    /* 画面幅が狭い場合に折り返せるようにする */
        #         }
        #         /* 各フォームを等分し、横幅を可変に */
        #         .button-container form {
        #             flex: 1;            /* 横幅を均等配分 */
        #             min-width: 120px;   /* 小さすぎないように最小幅を設定 */
        #         }
        #         .custom-button {
        #             width: 100%;        /* form 内で100%にして可変幅 */
        #             padding: 8px 16px;
        #             font-size: 20px;
        #             font-weight: bold;
        #             text-align: center;
        #             text-decoration: none;
        #             color: #333;            /* 文字色(地味め) */
        #             background-color: #ddd; /* ボタン背景色(地味め) */
        #             border: none;
        #             border-radius: 5px;
        #             box-shadow: 0 2px #999;
        #             transition: all 0.3s ease;
        #         }
        #         .custom-button:hover {
        #             background-color: #ccc; /* ホバー時の背景色 */
        #         }
        #         .custom-button:active {
        #             box-shadow: 0 1px #666;
        #             transform: translateY(1px);
        #         }
        #     </style>
        #     <!-- 説明文 -->
        #     <div class="description-text">
        #         <i class="fa-regular fa-clock"></i>&nbsp;&nbsp;結果の確認
        #     </div>
        #     <div class="button-container">
        #         <!-- 実行ボタン -->
        #         <form action="" method="get">
        #             <button class="custom-button" type="submit" name="check_forecast_result" value="true">
        #                 <i class="fa-solid fa-circle" style="color: #3EB489;"></i>&nbsp;&nbsp;フォルダーを開く
        #             </button>
        #     </div>
        # """, unsafe_allow_html=True)

        # # クエリパラメータを取得（ボタン押下を検出）
        # query_params = st.query_params

        # # フォルダーを開くボタンを押した（予測の結果を確認する）場合
        # if query_params.get("check_forecast_result") == "true":
        #     os.startfile(folder_path)
        #     st.query_params.clear()
        # else:
        #     print("1")

        # セッションステートの初期化（必要なら）
        if 'check_results' not in st.session_state:
            st.session_state.check_results = False

        # 関数（コールバック）
        def check_results():
            st.session_state.check_results = True

        # サイドバー内にフォームを作成
        with st.sidebar.form(key="results_form"):
            st.subheader("予測結果の確認")

            # フォーム内のボタン
            check_button = st.form_submit_button("📂　フォルダーを開く　")

            # ボタンが押されたら処理実行
            if check_button:
                check_results()

            # フォーム外で処理を実行（セッションステートによる判定）
            if st.session_state.check_results:
                os.startfile(folder_path)  # フォルダーを開く

            # 下にステータス表示
            if st.session_state.check_results:
                st.info("エクスプローラーでフォルダーを開いています。")
            else:
                st.warning("予測結果を確認する場合は、「フォルダーを開く」を押してください。")


        #無限ループ
        if st.session_state.processing:
            # === 実行したい時刻をリストで指定（24時間表記） ===
            # 例えば ["10:00", "11:00", "12:00"] とか
            target_times = ["00:57", "01:06", "22:50"]

            # 実行済みフラグ（同じ時刻に複数回動かないようにする）
            executed_times = set()

            def job(run_time):
                #now = datetime.now()

                current_time = datetime.now()
                now = current_time.replace(minute=0, second=0, microsecond=0)

                st.write(now)

                print(f"{now.strftime('%Y-%m-%d %H:%M:%S')} に {run_time} を実行！")

                # ここにやりたい処理を書く
                df = pd.DataFrame({
                    '実行時刻': [now.strftime('%Y-%m-%d %H:%M:%S')],
                    'データ': [42]
                })

                df = forecast_v3.show_zaiko_simulation( now,1)

                # ファイル名は実行時間ベースでOK
                filename = f"定期予測結果/data_{now.strftime('%Y%m%d_%H%M')}.csv"
                df.to_csv(filename, index=False)
                print(f"{filename} を保存しました。\n")

            print("=== 指定した時刻に処理を実行します ===")

            while True:
                now = datetime.now()
                current_time_str = now.strftime("%H:%M")

                # 実行予定時刻と一致したら
                if current_time_str in target_times and current_time_str not in executed_times:
                    job(current_time_str)
                    executed_times.add(current_time_str)  # 一度実行したら記録する

                # 翌日になったらリセット
                if current_time_str == "00:00":
                    executed_times.clear()

                time.sleep(1)  # 1秒ごとにチェック
    
    elif page == "在庫リミット計算":
        forecast_page()
        st.session_state.processing = False

    elif page == "在庫予測":
        st.write("開発中")
        st.session_state.processing = False
    
    elif page == "在庫シミュレーション（仮名）":
        zaiko_simulation_page()
        st.session_state.processing = False

    elif page == "要因分析":
        analysis_page()
        st.session_state.processing = False

    elif page == "上下限外れ確認":

        #! 関数を呼び出してCSSを適用
        apply_custom_css()

        #* ＜ローカルデータ利用する場合＞
        start_date = '2024-05-01-00'
        end_date = '2024-08-29-00'
        #*＜実行時間で日時を選択する場合＞
        #current_time = datetime.now()# 現在の実行時間を取得
        #end_date = (current_time - timedelta(days=1)).strftime('%Y-%m-%d-%H')# end_dateを実行時間の前日
        #start_date = (current_time - timedelta(days=1) - timedelta(days=180)).strftime('%Y-%m-%d-%H')# start_dateを実行時間の前日からさらに半年前

        #! 自動ラックの在庫データを読み込み
        # todo 引数関係なく全データ読み込みしてる
        zaiko_df = read_zaiko_by_using_archive_data(start_date, end_date)
        # 実行結果の確認
        # 開始時間と終了時間を取得
        #min_datetime = zaiko_df['日時'].min()
        #max_datetime = zaiko_df['日時'].max()
        #st.write(min_datetime, max_datetime)

        #! 品番列を昇順にソート
        zaiko_df = zaiko_df.sort_values(by='品番', ascending=True)
        #! 無効な値を NaN に変換
        zaiko_df['拠点所番地'] = pd.to_numeric(zaiko_df['拠点所番地'], errors='coerce')
        #! 品番ごとに欠損値（NaN）を埋める(前方埋め後方埋め)
        zaiko_df['拠点所番地'] = zaiko_df.groupby('品番')['拠点所番地'].transform(lambda x: x.fillna(method='ffill').fillna(method='bfill'))
        #! それでも置換できないものはNaN を 0 で埋める
        zaiko_df['拠点所番地'] = zaiko_df['拠点所番地'].fillna(0).astype(int).astype(str)
        #! str型に変換
        zaiko_df['拠点所番地'] = zaiko_df['拠点所番地'].astype(int).astype(str)

        #! 
        file_path = 'temp/マスター_品番&仕入先名&仕入先工場名.csv'
        syozaikyotenchi_data = pd.read_csv(file_path, encoding='shift_jis')
        #! 空白文字列や非数値データをNaNに変換
        syozaikyotenchi_data['拠点所番地'] = pd.to_numeric(syozaikyotenchi_data['拠点所番地'], errors='coerce')
        #! str型に変換
        syozaikyotenchi_data['拠点所番地'] = syozaikyotenchi_data['拠点所番地'].fillna(0).astype(int).astype(str)

        #! 受入場所追加
        zaiko_df = pd.merge(zaiko_df, syozaikyotenchi_data[['品番','拠点所番地','受入場所']], on=['品番', '拠点所番地'], how='left')
        #! 日付列を作成
        zaiko_df['日付'] = zaiko_df['日時'].dt.date

        #st.dataframe(zaiko_df.head(20000))

        #! Activedata
        file_path = 'temp/activedata.csv'#ステップ１,2で併用しているため、変数ではなく一時フォルダーに格納して使用
        Activedata = pd.read_csv(file_path, encoding='shift_jis')
        #st.dataframe(Activedata)

        #! 両方のデータフレームで '日付' 列を datetime 型に統一
        zaiko_df['日付'] = pd.to_datetime(zaiko_df['日付'])
        Activedata['日付'] = pd.to_datetime(Activedata['日付'])

        #! 日時でデータフレームを結合
        zaiko_df = pd.merge(zaiko_df, Activedata, on=['品番','受入場所','日付'])

        #! 特定の時間帯のデータを抽出
        zaiko_df = zaiko_df[(zaiko_df['日時'] >= start_date) & (zaiko_df['日時'] <= end_date)]

        st.dataframe(zaiko_df.head(20000))

        data = zaiko_df

        # Step 1: 「品番」列と「受入場所」列の内容を統合し、新しい列「品番_受入場所」を作成
        data['品番_受入場所'] = data['品番'].astype(str) + "_" + data['受入場所'].astype(str)

        # Step 2: ユニークな「品番_受入場所」を計算
        unique_items = data['品番_受入場所'].unique()

        # Step 3: 「品番_受入場所」に対してループを回して下限割れ回数と上限越え回数を計算
        results = []

        for item in unique_items:
            subset = data[(data['品番_受入場所'] == item)]
            below_min_count = (subset['在庫数（箱）'] < subset['設計値MIN']).sum()
            above_max_count = (subset['在庫数（箱）'] > subset['設計値MAX']).sum()
            meanzaiko = subset['在庫数（箱）'].mean()
            results.append({'品番_受入場所': item, '下限割れ発生回数': below_min_count,
                             '上限越え発生回数': above_max_count, '平均在庫数': meanzaiko})

        # Step 4: データフレームにまとめる
        results_df = pd.DataFrame(results)

        st.dataframe(results_df)

    elif page == "関所別かんばん数可視化（アニメーション）":

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
    elif page == "フレ可視化":

        #! 関数を呼び出してCSSを適用
        apply_custom_css()

        start_date = '2024-05-01-00'
        end_date = '2024-08-31-00'

        # Streamlit アプリケーション
        st.title('フレ可視化')

        hinban_seibishitsu_df = create_hinban_info()
        # サイドバーに品番選択ボックスを作成
        product = st.selectbox("品番を選択してください", hinban_seibishitsu_df['品番_整備室'])

        #st.write()

        #! 品番、整備室コードを抽出
        part_number = product.split('_')[0]
        seibishitsu = product.split('_')[1]
    
        activedata = read_activedata_by_using_archive_data(start_date, end_date, 0)
        # 特定の品番の商品データを抽出
        activedata = activedata[(activedata['品番'] == part_number) & (activedata['受入場所'] == seibishitsu)]
        #st.dataframe(activedata)
        activedata['日量数（箱数）']=activedata['日量数']/activedata['収容数']

        #file_path = '中間成果物/所在管理MBデータ_統合済&特定日時抽出済.csv'
        Timestamp_df = read_syozailt_by_using_archive_data(start_date, end_date)
        # '更新日時'列に無効な日時データがある行を削除する
        data_cleaned = Timestamp_df.dropna(subset=['検収日時'])
        #st.dataframe(data_cleaned.head(50000))
        # 特定の品番の商品データを抽出
        data_cleaned = data_cleaned[(data_cleaned['品番'] == part_number) & (data_cleaned['整備室コード'] == seibishitsu)]
        #data_cleaned = data_cleaned[ (data_cleaned['整備室コード'] == seibishitsu)]
        # 日付部分を抽出
        #st.dataframe(data_cleaned)
        data_cleaned['納入日'] = pd.to_datetime(data_cleaned['納入日']).dt.date
        # 納入日ごとにかんばん数をカウント
        df_daily_sum = data_cleaned.groupby(data_cleaned['納入日']).size().reset_index(name='納入予定かんばん数')

        #st.dataframe(df_daily_sum)

        # 実績データの納入日も日付型に変換
        activedata['納入日'] = activedata['日付']
        activedata['納入日'] = pd.to_datetime(activedata['納入日'])
        df_daily_sum['納入日'] = pd.to_datetime(df_daily_sum['納入日'])

        # 再度、両データを納入日で結合
        df_merged = pd.merge(df_daily_sum, activedata[['納入日', '日量数（箱数）']], on='納入日', how='left')

        # 差分を計算
        df_merged['フレ'] = df_merged['日量数（箱数）'] - df_merged['納入予定かんばん数']

        #st.dataframe(df_merged)

        # Streamlitで開始日と終了日を選択
        #st.title("納入予定かんばん数と日量数の差分")
        default_start_date = datetime.strptime('2024-05-01', '%Y-%m-%d').date()
        start_date = st.date_input("開始日", value=default_start_date)
        end_date = st.date_input("終了日", value=df_merged['納入日'].max())

        # 開始日と終了日に基づいてデータをフィルタリング
        filtered_data = df_merged[(df_merged['納入日'] >= pd.to_datetime(start_date)) &
                                (df_merged['納入日'] <= pd.to_datetime(end_date))]

        # フィルタリングされたデータの差分の折れ線グラフ作成
        fig = px.line(filtered_data, x='納入日', y='フレ', title='納入フレ（日量箱数と納入予定かんばん数の差分）の推移')

        # y=0に赤線を追加
        fig.add_shape(
            type='line',
            x0=filtered_data['納入日'].min(), x1=filtered_data['納入日'].max(),
            y0=0, y1=0,
            line=dict(color='red', width=2),
            name='フレ0'
        )

        # 赤線に名前を追加
        fig.add_annotation(
            x=filtered_data['納入日'].max(), y=0,
            text="フレ0",
            showarrow=False,
            yshift=10,
            font=dict(color="red", size=12)
        )

        # 土日を強調するために、納入日の曜日をチェック
        filtered_data['weekday'] = pd.to_datetime(filtered_data['納入日']).dt.weekday

        # 土日だけを抽出（5:土曜日, 6:日曜日）
        weekends = filtered_data[filtered_data['weekday'] >= 5]

        # グラフ描画後に土日を強調する縦線を追加
        for date in weekends['納入日']:
            fig.add_shape(
                type='line',
                x0=date, x1=date,
                y0=filtered_data['フレ'].min(), y1=filtered_data['フレ'].max(),
                line=dict(color='black', width=2),
                name='土日'
            )

        # 1日単位で横軸のメモリを設定
        fig.update_xaxes(dtick="D1")

        # Streamlitでグラフを表示
        st.plotly_chart(fig)

        st.info("赤線より上は、実績＜内示。赤線より下は、実績＞内示")
    
        #------------------------------------------------------------------------------------------

        # start_date = '2024-05-01-00'
        # end_date = '2024-08-31-00'

        # # データの読み込み（例: hinban_seibishitsu_df, activedataのデータは事前に準備）
        # #hinban_seibishitsu_df = read_syozailt_by_using_archive_data(start_date, end_date)  # 品番・整備室のデータファイル
        # #hinban_seibishitsu_df['納入日'] = pd.to_datetime(hinban_seibishitsu_df['納入日']).dt.date
        # # 納入日ごとにかんばん数をカウント
        # #hinban_seibishitsu_df = hinban_seibishitsu_df.groupby(hinban_seibishitsu_df['納入日']).size().reset_index(name='納入予定かんばん数')
        # #activedata = read_activedata_by_using_archive_data(start_date, end_date, 0)

        # # サイドバーに品番選択ボックスを作成
        # selected_products = st.multiselect("品番を選択してください", hinban_seibishitsu_df['品番_整備室'].unique())

        # # 複数品番を選択した場合に対応
        # if selected_products:
        #     # 選択された品番のデータを処理
        #     filtered_data_list = []
            
        #     for product in selected_products:
        #         part_number = product.split('_')[0]
        #         seibishitsu = product.split('_')[1]
                
        #         # 特定の品番の商品データを抽出
        #         activedata_filtered = activedata[(activedata['品番'] == part_number) & (activedata['受入場所'] == seibishitsu)]
        #         activedata_filtered['日量数（箱数）'] = activedata_filtered['日量数'] / activedata_filtered['収容数']

        #         # Timestampデータの処理
        #         Timestamp_df = read_syozailt_by_using_archive_data(start_date, end_date)
        #         data_cleaned = Timestamp_df.dropna(subset=['検収日時'])
        #         data_cleaned = data_cleaned[(data_cleaned['品番'] == part_number) & (data_cleaned['整備室コード'] == seibishitsu)]
                
        #         # 日付の処理
        #         data_cleaned['納入日'] = pd.to_datetime(data_cleaned['納入日']).dt.date
        #         df_daily_sum = data_cleaned.groupby('納入日').size().reset_index(name='納入予定かんばん数')

        #         # 実績データの納入日も日付型に変換
        #         activedata_filtered['納入日'] = pd.to_datetime(activedata_filtered['日付'])
        #         df_daily_sum['納入日'] = pd.to_datetime(df_daily_sum['納入日'])

        #         # 両データを納入日で結合
        #         df_merged = pd.merge(df_daily_sum, activedata_filtered[['納入日', '日量数（箱数）']], on='納入日', how='left')

        #         # 差分を計算
        #         df_merged['差分'] = df_merged['日量数（箱数）'] - df_merged['納入予定かんばん数']
        #         df_merged['品番'] = part_number  # 品番情報を追加して区別
                
        #         filtered_data_list.append(df_merged)

        #     # 複数の品番を結合
        #     final_filtered_data = pd.concat(filtered_data_list, ignore_index=True)

        #     st.dataframe(final_filtered_data)

        #     # Streamlitで開始日と終了日を選択
        #     st.title("納入予定かんばん数と日量数の差分")
        #     start_date = st.date_input("開始日", value=final_filtered_data['納入日'].min(), key="start_date")
        #     end_date = st.date_input("終了日", value=final_filtered_data['納入日'].max(), key="end_date")

        #     # 開始日と終了日に基づいてデータをフィルタリング
        #     filtered_data = final_filtered_data[(final_filtered_data['納入日'] >= pd.to_datetime(start_date)) &
        #                                         (final_filtered_data['納入日'] <= pd.to_datetime(end_date))]

        #     # フィルタリングされたデータの差分の折れ線グラフ作成（品番ごとに区別）
        #     fig = px.line(filtered_data, x='納入日', y='差分', color='品番', title='納入予定かんばん数と日量数の差分（複数品番対応）')

        #     # Streamlitでグラフを表示
        #     st.plotly_chart(fig)


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

        st.session_state.processing = False
        

#! 本スクリプトが直接実行されたときに実行
if __name__ == "__main__":
    print("プログラムが実行中です")
    main()
