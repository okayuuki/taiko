import streamlit as st
# ページ設定: 名前とアイコンを変更
st.set_page_config(
    page_title="部品在庫分析システム",  # 名前を設定
    page_icon="📦",              # アイコン
)
import datetime as dt
import pandas as pd
import os
import time
import fitz  # PyMuPDFのインポート
import json

# 自作ライブラリのimport（備忘：関数として認識されないときは、vscodeを再起動する）
from get_data import get_hinban_master, compute_hourly_buhin_zaiko_data_by_hinban, compute_hourly_tehai_data_by_hinban, get_hinban_info_detail
from visualize_data import show_abnormal_results
from preprocess_data import compute_features_and_target
from apply_models import pipeline, show_zaiko_with_baseline, compute_feature_importance, show_feature_importance
from forecast_data import show_results_of_zaiko_limit, show_results_of_future_zaiko

# 工場設定の取得
# 現在のファイルの絶対パスを取得
current_dir = os.path.dirname(os.path.abspath(__file__))
# 設定ファイルの絶対パスを作成
CONFIG_PATH = os.path.join(current_dir, "..", "..", "configs", "settings.json")
# 日本語があるため、UTF-8でファイルを読み込む
with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
    config = json.load(f)
#　対象工場の読み込み
selected_data = config['selected_data']
kojo = selected_data["kojo"]
#kojo = 'anjo1'# 決め打ちの場合
flag_useDataBase = 1

# 命名規則
# - 拡張性を高めるために変数名は機能毎STEP毎に設定する

# 定期予測の結果を保存するフォルダー
RESULT_FOLDER_PATH = 'kari'

# MARK: カスタムCSSを適用して画面サイズを整える
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

#MARK: 初期化ボタン設定
def clear_session_button(label="キャッシュ削除（初期化）", key="clear_session_button"):
    """
    セッションステートを削除するサイドバーボタン。
    複数回呼ばれても安全なように key を指定。
    """

    # Streamlitのキャッシュをクリア
    if st.sidebar.button(label, key=key):
        st.cache_resource.clear()
        st.session_state.clear()
        st.sidebar.info("🎉 セッションステートを初期化しました！")

#MARK: 特徴量重要度を活用して要因分析を行うページ設定
def page_of_analyze_feature_importance():

    # カスタムCSSを適用して画面サイズを設定する
    apply_custom_css()

    # タイトル
    st.title("在庫変動要因分析（特徴量重要度活用ver）※1Yのみ")
    # 区切り線を追加
    st.divider()

    # キャッシュ削除
    clear_session_button()

    # ステップ0 
    st.sidebar.title("ステップ０：異常の確認（任意）")

    # 異常確認日
    if "selected_date_for_analyze_feature_importance0" not in st.session_state:
        st.session_state.selected_date_for_analyze_feature_importance0 = ""

    # 異常確認時間
    if "selected_time_for_analyze_feature_importance0" not in st.session_state:
        current_time = dt.datetime.now().time()
        st.session_state.selected_time_for_analyze_feature_importance0 = ""

    #st.write("実行確認")
    #st.write(st.session_state.selected_date_for_analyze_feature_importance0)
    #st.write(st.session_state.selected_time_for_analyze_feature_importance0)

    # フォーム作成
    with st.sidebar.form(key='form_analyze_feature_importance0'):

        # 現在の時刻を取得
        current_time = dt.datetime.now()

        # 現在の日付を取得
        current_day = current_time.date()

        # 日付入力
        # 初期値は現在日
        selected_date = st.date_input("日付を選択してください", value = current_day)

        # その時のちょうどの時（分・秒・ミリ秒を0にする）を計算する
        # 2025-03-27 14:38:45.123456 → 2025-03-27 14:00:00
        current_hour = (current_time).replace(minute=0, second=0, microsecond=0)

        # 時間（hour）だけを整数で取得
        default_time_index = current_hour.hour

        # 1時間ごとの選択肢を作成
        hourly_times = [f"{hour:02d}:00" for hour in range(24)]  # 00:00～23:00の時間リスト

        # 時間選択（1時間ごと）
        selected_time = st.selectbox("時間を選択してください", hourly_times, index = default_time_index)

        # フォームの送信ボタン
        submit_button_step0 = st.form_submit_button("登録する")

    # 登録するボタンが押されたときの処理
    if submit_button_step0: 

        st.session_state.selected_date_for_analyze_feature_importance0 = selected_date

        st.session_state.selected_time_for_analyze_feature_importance0 = selected_time
        
        with st.spinner("実行中です。しばらくお待ちください..."):
            show_abnormal_results( st.session_state.selected_date_for_analyze_feature_importance0,
             st.session_state.selected_time_for_analyze_feature_importance0, flag_useDataBase, kojo)
            
            #実行確認
            #st.write(st.session_state.selected_date_for_analyze_feature_importance0,
            #         st.session_state.selected_time_for_analyze_feature_importance0)

        st.sidebar.info("🎉 処理が完了しました！")

    # 登録するボタンが押されなかったときの処理
    else:
        st.sidebar.code("このステップは任意です。スキップできます。")

    # ステップ１
    st.sidebar.title("ステップ１：品番選択")

    # ○○_for_analyze_feature_importance1と変数名をつける

    # 選択品番
    if "selected_hinban_for_analyze_feature_importance1" not in st.session_state:
        st.session_state.selected_hinban_for_analyze_feature_importance1 = ""

    # モデル
    if "model_for_analyze_feature_importance1" not in st.session_state:
        st.session_state.model_for_analyze_feature_importance1 = ""
    
    # 説明変数X
    if "features_for_analyze_feature_importance1" not in st.session_state:
        st.session_state.features_for_analyze_feature_importance1 = ""
    
    # 目的変数と説明変数含む完全データ
    if "data_for_analyze_feature_importance1" not in st.session_state:
        st.session_state.data_for_analyze_feature_importance1 = ""

    # フォーム作成
    with st.sidebar.form(key='form_analyze_feature_importance1'):
    
        #　ユニークな '品番_整備室' 列を作成
        hinban_seibishitsu_df = get_hinban_master()
        columns_names = ['品番_整備室コード']  # 必要な列名を指定
        hinban_seibishitsu_df = pd.DataFrame(hinban_seibishitsu_df, columns=columns_names)

        # サイドバーに品番選択ボックスを作成
        product = st.selectbox("品番を選択してください", hinban_seibishitsu_df['品番_整備室コード'])
        
        # 「登録する」ボタンをフォーム内に追加
        submit_button_step1 = st.form_submit_button(label='登録する')

    # 登録するボタンが押されたときの処理
    if submit_button_step1:

        st.session_state.selected_hinban_for_analyze_feature_importance1 = product

        # 品番情報表示
        flag_display = 1
        hinban_info = ["", ""]
        hinban_info[0], hinban_info[1] = product.split("_", 1)
        now = dt.datetime.now()
        end_datetime = now.replace(minute=0, second=0, microsecond=0)
        get_hinban_info_detail(hinban_info, end_datetime, flag_display, flag_useDataBase, kojo)

        # メッセージ
        st.sidebar.success(f"新たに選択された品番: {st.session_state.selected_hinban_for_analyze_feature_importance1}")
        
        # 要因分析パイプライン実行
        with st.spinner("実行中です。しばらくお待ちください..."):
            hinban_info = ["", ""]  # 2つの空文字列で初期化
            hinban_info[0], hinban_info[1] = (st.session_state.selected_hinban_for_analyze_feature_importance1).split('_')
            # 現在時刻を取得し、分以降を00に設定
            now = dt.datetime.now()
            end_datetime = now.replace(minute=0, second=0, microsecond=0)
            # end から6ヶ月前の日時を計算
            start_datetime = end_datetime - dt.timedelta(days=180)  # 約6ヶ月（180日）
            # datetime オブジェクトを文字列に変換（必要な場合）
            start_datetime_str = start_datetime.strftime('%Y-%m-%d %H:00:00')
            end_datetime_str = end_datetime.strftime('%Y-%m-%d %H:00:00')
            time_granularity = 'h'
            model, X, df = pipeline(hinban_info, start_datetime_str, end_datetime_str, time_granularity, flag_useDataBase, kojo)
            st.session_state.model_for_analyze_feature_importance1 = model
            st.session_state.features_for_analyze_feature_importance1 = X
            st.session_state.data_for_analyze_feature_importance1 = df

        # 品番情報提示
        #display_hinban_info(product)

        st.sidebar.info("🎉 処理が完了しました！")

    # 登録するボタンが押されなかったときの処理
    else:
        
        # まだ一度もSTEP1が実行されていない時
        if st.session_state.selected_hinban_for_analyze_feature_importance1 == "":
            st.sidebar.warning("品番を選択し、「登録する」ボタンを押してください")

        #1度はボタン押されている
        else:

            # 品番情報表示
            flag_display = 1
            hinban_info = ["", ""]
            hinban_info[0], hinban_info[1] = product.split("_", 1)
            now = dt.datetime.now()
            end_datetime = now.replace(minute=0, second=0, microsecond=0)
            get_hinban_info_detail(hinban_info, end_datetime, flag_display, flag_useDataBase, kojo)

            st.sidebar.success(f"過去に選択された品番: {st.session_state.selected_hinban_for_analyze_feature_importance1}")
            
    # ステップ２
    st.sidebar.title("ステップ２：在庫確認")

    # 初期値設定
    # 開始日時
    if 'start_date_for_analyze_feature_importance2' not in st.session_state:
        st.session_state.start_date_for_analyze_feature_importance2 = dt.date.today()
    # 終了日時
    if 'end_date_for_analyze_feature_importance2' not in st.session_state:
        st.session_state.end_date_for_analyze_feature_importance2 = dt.date.today()
    # # 選択日時
    if 'highlight_datetime_for_analyze_feature_importance2' not in st.session_state:
        st.session_state.highlight_datetime_for_analyze_feature_importance2 = None
    # 実行有無確認用
    if 'flag_for_analyze_feature_importance2' not in st.session_state:
        st.session_state.flag_for_analyze_feature_importance2 = 0

    # サイドバーにフォームの作成
    with st.sidebar.form(key='form_analyze_feature_importance2'):

        # 日付を選んだら、その値を st.session_state.start_date に上書き保存する
        st.session_state.start_date_for_analyze_feature_importance2 = st.date_input("開始日", st.session_state.start_date_for_analyze_feature_importance2)
        st.session_state.end_date_for_analyze_feature_importance2 = st.date_input("終了日", st.session_state.end_date_for_analyze_feature_importance2)

        # 時間型を更新
        # 開始日 → 00時00分00秒で固定
        st.session_state.start_date_for_analyze_feature_importance2 = dt.datetime.combine(st.session_state.start_date_for_analyze_feature_importance2, dt.time(0, 0))
        # 終了日 → 選択した日付によって異なる処理を行う
        # 必要変数計算
        today = dt.date.today()# 現在の日付（今日）を取得
        selected_end_date = st.session_state.end_date_for_analyze_feature_importance2# 選択された終了日（date型）
        # 終了日が今日なら → 現在の hour:00
        if selected_end_date == today:
            current_hour = dt.datetime.now().hour
            end_time = dt.time(current_hour, 0)
        # それ以外なら → 23:00
        else:
            end_time = dt.time(23, 0)
        # 終了日を datetime に変換
        st.session_state.end_date_for_analyze_feature_importance2 = dt.datetime.combine(selected_end_date, end_time)
        
        # フォームの送信ボタン
        submit_button_step2 = st.form_submit_button(label='登録する')

    # 登録するボタンが押されたときの処理
    if submit_button_step2:

        # 実行結果確認
        # st.write(st.session_state.start_date_for_analyze_feature_importance2,
        #          st.session_state.end_date_for_analyze_feature_importance2)
        
        # STEP3を実行してからSTEP2を実行すると、キャッシュが残るので
        st.session_state.highlight_datetime_for_analyze_feature_importance2 = None

        # 在庫推移描画
        show_zaiko_with_baseline( st.session_state.data_for_analyze_feature_importance1,
                                  st.session_state.start_date_for_analyze_feature_importance2,
                                  st.session_state.end_date_for_analyze_feature_importance2,
                                  st.session_state.highlight_datetime_for_analyze_feature_importance2)
        
        # 実行フラグ更新
        st.session_state.flag_for_analyze_feature_importance2 = 1
        
        st.sidebar.info("🎉 処理が完了しました！")

    # 登録するボタンが押されなかったときの処理
    else:
        
        # まだ一度もSTEP2が実行されていない時
        if st.session_state.flag_for_analyze_feature_importance2 == 0:
            st.sidebar.warning("開始日終了日を選択し、「登録する」ボタンを押してください")

        #1度はボタン押されている
        else:
            st.sidebar.success(f"過去に選択された開始日時: {st.session_state.start_date_for_analyze_feature_importance2}")
            st.sidebar.success(f"過去に選択された終了日時: {st.session_state.end_date_for_analyze_feature_importance2}")

    # ステップ３
    st.sidebar.title("ステップ３：要因分析")

    # 初期値設定
    # 開始日時
    if 'selected_datetime_for_analyze_feature_importance3' not in st.session_state:
        st.session_state.selected_datetime_for_analyze_feature_importance3 = None

    # フォーム作成
    with st.sidebar.form("form_analyze_feature_importance3"):

        # 日時の選択肢を生成
        datetime_range = pd.date_range(st.session_state.start_date_for_analyze_feature_importance2,
                                        st.session_state.end_date_for_analyze_feature_importance2, freq='h')
        datetime_options = [dt.replace(minute=0, second=0, microsecond=0) for dt in datetime_range]

        # 日時選択用セレクトボックス
        selected_datetime = st.selectbox(
            "要因分析の結果を表示する日時を選択してください",
            datetime_options
        )

        submit_button_step3 = st.form_submit_button("登録する")

        
    if submit_button_step3:

        # 在庫プロット
        st.session_state.highlight_datetime_for_analyze_feature_importance2 = selected_datetime
        # 在庫推移描画（ハイライトあり）)
        show_zaiko_with_baseline( st.session_state.data_for_analyze_feature_importance1,
                                st.session_state.start_date_for_analyze_feature_importance2,
                                st.session_state.end_date_for_analyze_feature_importance2,
                                st.session_state.highlight_datetime_for_analyze_feature_importance2)

        # 要因分析の結果プロット
        st.session_state.selected_datetime_for_analyze_feature_importance3 = selected_datetime
        st.sidebar.success(f"選択された日時: {st.session_state.selected_datetime_for_analyze_feature_importance3}")
        # 特徴量重要度の計算
        shap_values = compute_feature_importance( st.session_state.model_for_analyze_feature_importance1,
                                    st.session_state.features_for_analyze_feature_importance1)
        # 結果を描画
        show_feature_importance( st.session_state.data_for_analyze_feature_importance1,
                                 st.session_state.features_for_analyze_feature_importance1,
                                   selected_datetime,
                                     shap_values)

        st.sidebar.info("🎉 処理が完了しました！")
    
    else:
        st.sidebar.warning("要因分析の結果を表示する日時を選択し、「登録する」ボンを押してください")

#MARK: 在庫リミット計算を行うページ設定（1時間単位）#!不使用
def page_of_zaiko_limit_calc():

    # カスタムCSSを適用して画面サイズを設定する
    apply_custom_css()

    # タイトル
    st.title("在庫リミット計算")

    # キャッシュ削除
    clear_session_button()

    # 選択品番
    if "selected_hinban_for_zaiko_limit_calc1" not in st.session_state:
        st.session_state.selected_hinban_for_zaiko_limit_calc1 = "品番_整備室コード"
 
    # サイドバートップメッセージ
    st.sidebar.write("## 🔥各ステップを順番に実行してください🔥")

    st.sidebar.title("ステップ１：品番選択")

    # フォーム作成
    with st.sidebar.form(key='form_zaiko_limit_calc1'):
    
        hinban_seibishitsu_df = get_hinban_master()
        columns_names = ['品番_整備室コード']  # 必要な列名を指定
        hinban_seibishitsu_df = pd.DataFrame(hinban_seibishitsu_df, columns=columns_names)
        #実行結果の確認
        #st.dataframe(hinban_seibishitsu_df)

        # サイドバーに品番選択ボックスを作成
        unique_product = st.selectbox("品番を選択してください", hinban_seibishitsu_df['品番_整備室コード'])
        
        # 「適用」ボタンをフォーム内に追加
        submit_button_step1 = st.form_submit_button(label='登録する')

    # 登録するボタンが押されたときの処理
    if submit_button_step1 == True:

        st.sidebar.success(f"新たに選択された品番: {unique_product}")
        
        st.session_state.selected_hinban_for_zaiko_limit_calc1 = unique_product
        
        #todo　品番情報を表示
        #display_hinban_info(unique_product)

        st.sidebar.info("🎉 処理が完了しました！")

        # 折り返し線を追加
        #st.markdown("---")

    else:

        if st.session_state.selected_hinban_for_zaiko_limit_calc1 == "品番_整備室コード":
            st.sidebar.warning("品番を選択し、「登録する」ボタンを押してください")

        else:
            st.sidebar.success(f"過去に選択した品番: {st.session_state.selected_hinban_for_zaiko_limit_calc1}")

            #todo　品番情報を表示
            #display_hinban_info(unique_product)

            # 折り返し線を追加
            #st.markdown("---")

    st.sidebar.title("ステップ２：日時選択")

    # 初期値設定
    # 開始日
    if 'start_date_for_zaiko_limit_calc2' not in st.session_state:
        st.session_state.start_date_for_zaiko_limit_calc2 = dt.date.today()

    # 開始時間
    if 'start_time_for_zaiko_limit_calc2' not in st.session_state:
        current_time = dt.datetime.now().time()
        st.session_state.start_time_for_zaiko_limit_calc2 = dt.time(current_time.hour, 0)

    # 開始日時
    if 'start_datetime_for_zaiko_limit_calc2' not in st.session_state:
        st.session_state.start_datetime_for_zaiko_limit_calc2 = ""
    
    with st.sidebar.form(key='form_zaiko_limit_calc2'):

        # 開始日
        st.session_state.start_date_for_zaiko_limit_calc2 = st.date_input("開始日", st.session_state.start_date_for_zaiko_limit_calc2)
        
        # 開始時間の選択肢をセレクトボックスで提供

        hours = [f"{i:02d}:00" for i in range(24)]
        start_time_str = st.selectbox("開始時間", hours,
                                      index = st.session_state.start_time_for_zaiko_limit_calc2.hour)
        # 選択された時間をdt_timeオブジェクトに変換
        start_time_hours = int(start_time_str.split(":")[0])
        # 時間を更新
        st.session_state.start_time_for_zaiko_limit_calc2 = dt.time(start_time_hours, 0)

        # フォームの送信ボタン
        submit_button_step2 = st.form_submit_button(label='登録する')
    
        # 開始日時と終了日時を結合
        start_datetime = dt.datetime.combine(st.session_state.start_date_for_zaiko_limit_calc2,
                                     st.session_state.start_time_for_zaiko_limit_calc2)
        
    # 登録するボタンを押された時
    if submit_button_step2:

        st.session_state.start_datetime_for_zaiko_limit_calc2 = start_datetime

        st.sidebar.success(f"開始日時: {st.session_state.start_datetime_for_zaiko_limit_calc2}")
        st.sidebar.info("🎉 処理が完了しました！")

    # 登録するボタンを押されなかった時
    else:
        
        if(st.session_state.start_datetime_for_zaiko_limit_calc2 == ""):
            st.sidebar.warning("開始日、開始時間を選択し、登録するボタンを押してください。")
             
        else:
            st.sidebar.success(f"過去に選択した開始日時: {st.session_state.start_datetime_for_zaiko_limit_calc2}")

    st.sidebar.title("ステップ３：在庫数入力")

    # 初期値設定
    # 在庫数箱
    if 'selected_zaiko_hako_for_zaiko_limit_calc3' not in st.session_state:
        st.session_state.selected_zaiko_hako_for_zaiko_limit_calc3 = ""
    if 'selected_zaiko_buhin_for_zaiko_limit_calc3' not in st.session_state:
        st.session_state.selected_zaiko_buhin_for_zaiko_limit_calc3 = ""

    # STEP1とSTEP2が実行されているとき
    if (st.session_state.selected_hinban_for_zaiko_limit_calc1 != "") and (st.session_state.start_datetime_for_zaiko_limit_calc2 != ""):

        try:
            hinban_info = ["", ""]  # 2つの空文字列で初期化
            hinban_info[0], hinban_info[1] = (st.session_state.selected_hinban_for_zaiko_limit_calc1).split('_')
            # 文字型に戻す
            start_datetime_for_zaiko = st.session_state.start_datetime_for_zaiko_limit_calc2.strftime('%Y-%m-%d %H:%M:%S')
            end_datetime_for_zaiko = st.session_state.start_datetime_for_zaiko_limit_calc2.strftime('%Y-%m-%d %H:%M:%S')
            # 指摘期間で読み込む
            zaiko_df = compute_hourly_buhin_zaiko_data_by_hinban(hinban_info,start_datetime_for_zaiko, end_datetime_for_zaiko,flag_useDataBase, kojo)
            #st.dataframe(zaiko_df)
            # 在庫数を保存
            zaiko_teian_hako = int(zaiko_df['在庫数（箱）'].values[0])
            zaiko_teian_buhin = int(zaiko_df['現在在庫（台）'].values[0])
        except Exception as e:
            zaiko_teian_hako = 0  # 最終的なエラー時は0を設定
            zaiko_teian_buhin = 0

    # STEP1とSTEP2、どちらかが実行されていないとき
    else:
        # 在庫の初期値を0とする
        zaiko_teian_hako = 0
        zaiko_teian_buhin = 0

    
    # フォーム作成
    with st.sidebar.form(key='form_zaiko_limit_calc3'):

        # 箱数選択用セレクトボックス
        # 100以上の在庫数ありえるので最大100だとエラー出る
        selected_zaiko_hako = st.selectbox("工場内の在庫数（箱）を入力してください", list(range(0,500)),
                                        index = zaiko_teian_hako,
                                            help="初期値は順立装置の現在庫数（箱）に設定されています")

        #
        selected_zaiko_buhin = st.selectbox("工場内の在庫数（部品数）を入力してください", list(range(0,50000)),
                                        index = zaiko_teian_buhin,
                                            help="初期値は順立装置の現在庫数（部品数）に設定されています")

        submit_button_step3 = st.form_submit_button("登録する")

    # 登録するボタンが押された時
    if submit_button_step3:

        st.session_state.selected_zaiko_hako_for_zaiko_limit_calc3 = selected_zaiko_hako
        st.session_state.selected_zaiko_buhin_for_zaiko_limit_calc3 = selected_zaiko_buhin
        
        st.sidebar.success(f"入力された在庫数（箱）: {st.session_state.selected_zaiko_hako_for_zaiko_limit_calc3}")
        st.sidebar.success(f"入力された在庫数（部品の数）: {st.session_state.selected_zaiko_buhin_for_zaiko_limit_calc3}")
        st.sidebar.info("🎉 処理が完了しました！")
            
    # 登録するボタンが押されなかった時
    else:

        if st.session_state.selected_zaiko_hako_for_zaiko_limit_calc3 == "":
            st.sidebar.warning("在庫数（箱or部品の数）を入力してください")

        else:
            st.sidebar.success(f"過去に入力された在庫数（箱）: {st.session_state.selected_zaiko_hako_for_zaiko_limit_calc3}")
            st.sidebar.success(f"過去に入力された在庫数（部品の数）: {st.session_state.selected_zaiko_buhin_for_zaiko_limit_calc3}")

    st.sidebar.title("ステップ４：需要調整")

    # フォームの作成
    with st.sidebar.form(key='form_zaiko_limit_calc4'):
        st.write("日量をご選択ください")

        # 2つのカラムを横並びに作成
        col1, col2 = st.columns(2)

        # 各カラムにボタンを配置
        with col1:
            submit_button_step4_mode1 = st.form_submit_button("日量を採用する",help="通常の日量を使用する")
        with col2:
            submit_button_step4_mode2 = st.form_submit_button("日量MAXを採用する",help="最大値の日量を使用する（生産数が多い場合で計算したい）")

    #　在庫リミット設定
    hinban_info = ["", ""]  # 2つの空文字列で初期化
    hinban_info[0], hinban_info[1] = (st.session_state.selected_hinban_for_zaiko_limit_calc1).split('_')
    target_column = "納入予定日時"
    if st.session_state.start_datetime_for_zaiko_limit_calc2 != "":
        start_datetime = (st.session_state.start_datetime_for_zaiko_limit_calc2).strftime('%Y-%m-%d %H:%M:%S')
    selected_zaiko_hako = st.session_state.selected_zaiko_hako_for_zaiko_limit_calc3
    selected_zaiko_buhin = st.session_state.selected_zaiko_buhin_for_zaiko_limit_calc3

    # 日量情報を表示
    if st.session_state.start_datetime_for_zaiko_limit_calc2 != "":
        tehai_data_df = compute_hourly_tehai_data_by_hinban( hinban_info, start_datetime, start_datetime, flag_useDataBase, kojo)
        #st.dataframe(tehai_data_df)

        nitiryo = tehai_data_df['日量数'].iloc[0]
        nitiryo_max = tehai_data_df['月末までの最大日量数'].iloc[0]

        # 2列に分けて表示する
        col1, col2 = st.sidebar.columns(2)

        # 左側の列に1つ目のmetricを表示
        with col1:
            st.metric(label="日量数", value = nitiryo)

        # 右側の列に2つ目のmetricを表示
        with col2:
            st.metric(label="現在日から月末までの最大日量数", value = nitiryo_max)

    # フォームの送信処理
    if submit_button_step4_mode1:
        st.sidebar.success("日量が採用されました")
        out_parameter = "日量を採用する"
        # 在庫リミット計算実行
        show_results_of_zaiko_limit(hinban_info, target_column, start_datetime, selected_zaiko_hako, selected_zaiko_buhin, out_parameter, kojo, flag_useDataBase)
        st.sidebar.info("🎉 処理が完了しました！")

    if submit_button_step4_mode2:
        st.sidebar.success("日量MAXが採用されました")
        out_parameter = "日量MAXを採用する"
        # 在庫リミット計算実行
        show_results_of_zaiko_limit(hinban_info, target_column, start_datetime, selected_zaiko_hako, selected_zaiko_buhin, out_parameter, kojo, flag_useDataBase)
        st.sidebar.info("🎉 処理が完了しました！")

    # 両方のボタンが押されていなかった場合のメッセージ
    if not submit_button_step4_mode1 and not submit_button_step4_mode2:
        st.sidebar.warning("日量をご選択ください") 

#MARK: 在庫リミット計算を行うページ設定（15分単位）
def page_of_zaiko_limit_calc2():

    # カスタムCSSを適用して画面サイズを設定する
    apply_custom_css()

    # タイトル
    st.title("在庫リミット計算")
    #st.write(f"{kojo}で計算します")

    # キャッシュ削除
    clear_session_button()

    # 選択品番
    if "selected_hinban_for_zaiko_limit_calc1" not in st.session_state:
        st.session_state.selected_hinban_for_zaiko_limit_calc1 = "品番_整備室コード"
 
    # サイドバートップメッセージ
    st.sidebar.write("## 🔥各ステップを順番に実行してください🔥")

    st.sidebar.title("ステップ１：品番選択")

    # フォーム作成
    with st.sidebar.form(key='form_zaiko_limit_calc1'):

        hinban_seibishitsu_df = get_hinban_master()
        columns_names = ['品番_整備室コード']  # 必要な列名を指定
        hinban_seibishitsu_df = pd.DataFrame(hinban_seibishitsu_df, columns=columns_names)
        #実行結果の確認
        #st.dataframe(hinban_seibishitsu_df)
        #st.dataffa(hinban_seibishitsu_df)

        # サイドバーに品番選択ボックスを作成
        unique_product = st.selectbox("品番を選択してください", hinban_seibishitsu_df['品番_整備室コード'])
        
        # 「適用」ボタンをフォーム内に追加
        submit_button_step1 = st.form_submit_button(label='登録する')

    # 登録するボタンが押されたときの処理
    if submit_button_step1 == True:

        st.sidebar.success(f"新たに選択された品番: {unique_product}")
        
        st.session_state.selected_hinban_for_zaiko_limit_calc1 = unique_product
        
        #todo　品番情報を表示
        #display_hinban_info(unique_product)

        st.sidebar.info("🎉 処理が完了しました！")

        # 折り返し線を追加
        #st.markdown("---")

    else:

        if st.session_state.selected_hinban_for_zaiko_limit_calc1 == "品番_整備室コード":
            st.sidebar.warning("品番を選択し、「登録する」ボタンを押してください")

        else:
            st.sidebar.success(f"過去に選択した品番: {st.session_state.selected_hinban_for_zaiko_limit_calc1}")

            #todo　品番情報を表示
            #display_hinban_info(unique_product)

            # 折り返し線を追加
            #st.markdown("---")

    st.sidebar.title("ステップ２：日時選択")

    # 初期値設定
    # 開始日
    if 'start_date_for_zaiko_limit_calc2' not in st.session_state:
        st.session_state.start_date_for_zaiko_limit_calc2 = dt.date.today()

    # 開始時間
    if 'start_time_for_zaiko_limit_calc2' not in st.session_state:
        current_time = dt.datetime.now().time()
        st.session_state.start_time_for_zaiko_limit_calc2 = dt.time(current_time.hour, 0)

    # 開始日時
    if 'start_datetime_for_zaiko_limit_calc2' not in st.session_state:
        st.session_state.start_datetime_for_zaiko_limit_calc2 = ""
    
    with st.sidebar.form(key='form_zaiko_limit_calc2'):

        # 開始日
        st.session_state.start_date_for_zaiko_limit_calc2 = st.date_input("開始日", st.session_state.start_date_for_zaiko_limit_calc2)
        
        # 開始時間の選択肢をセレクトボックスで提供

        hours = []
        for hour in range(24):
            for minute in [0, 15, 30, 45]:
                hours.append(f"{hour:02d}:{minute:02d}")

        start_time_str = st.selectbox("開始時間", hours,
                                    index=hours.index(f"{st.session_state.start_time_for_zaiko_limit_calc2.hour:02d}:{st.session_state.start_time_for_zaiko_limit_calc2.minute:02d}"))
        
        # 選択された時間と分を取得
        start_time_hours = int(start_time_str.split(":")[0])
        start_time_minutes = int(start_time_str.split(":")[1])

        # 時間と分を更新
        st.session_state.start_time_for_zaiko_limit_calc2 = dt.time(start_time_hours, start_time_minutes)

        # フォームの送信ボタン
        submit_button_step2 = st.form_submit_button(label='登録する')
    
        # 開始日時と終了日時を結合
        start_datetime = dt.datetime.combine(st.session_state.start_date_for_zaiko_limit_calc2,
                                     st.session_state.start_time_for_zaiko_limit_calc2)
        
    # 登録するボタンを押された時
    if submit_button_step2:

        st.session_state.start_datetime_for_zaiko_limit_calc2 = start_datetime

        st.sidebar.success(f"開始日時: {st.session_state.start_datetime_for_zaiko_limit_calc2}")
        st.sidebar.info("🎉 処理が完了しました！")

    # 登録するボタンを押されなかった時
    else:
        
        if(st.session_state.start_datetime_for_zaiko_limit_calc2 == ""):
            st.sidebar.warning("開始日、開始時間を選択し、登録するボタンを押してください。")
             
        else:
            st.sidebar.success(f"過去に選択した開始日時: {st.session_state.start_datetime_for_zaiko_limit_calc2}")

    st.sidebar.title("ステップ３：在庫数入力")

    # 初期値設定
    # 在庫数箱
    if 'selected_zaiko_hako_for_zaiko_limit_calc3' not in st.session_state:
        st.session_state.selected_zaiko_hako_for_zaiko_limit_calc3 = ""
    if 'selected_zaiko_buhin_for_zaiko_limit_calc3' not in st.session_state:
        st.session_state.selected_zaiko_buhin_for_zaiko_limit_calc3 = ""

    # STEP1とSTEP2が実行されているとき
    if (st.session_state.selected_hinban_for_zaiko_limit_calc1 != "") and (st.session_state.start_datetime_for_zaiko_limit_calc2 != ""):

        try:
            hinban_info = ["", ""]  # 2つの空文字列で初期化
            hinban_info[0], hinban_info[1] = (st.session_state.selected_hinban_for_zaiko_limit_calc1).split('_')
            # 文字型に戻す
            #! 00分単位に変換（在庫データ読み込み用）
            start_datetime_for_zaiko = st.session_state.start_datetime_for_zaiko_limit_calc2.replace(minute=0, second=0).strftime('%Y-%m-%d %H:%M:%S')
            end_datetime_for_zaiko = st.session_state.start_datetime_for_zaiko_limit_calc2.replace(minute=0, second=0).strftime('%Y-%m-%d %H:%M:%S')
            # 指摘期間で読み込む
            zaiko_df = compute_hourly_buhin_zaiko_data_by_hinban(hinban_info,start_datetime_for_zaiko, end_datetime_for_zaiko,flag_useDataBase, kojo)
            #st.dataframe(zaiko_df)
            # 在庫数を保存
            zaiko_teian_hako = int(zaiko_df['在庫数（箱）'].values[0])
            zaiko_teian_buhin = int(zaiko_df['現在在庫（台）'].values[0])
        except Exception as e:
            zaiko_teian_hako = 0  # 最終的なエラー時は0を設定
            zaiko_teian_buhin = 0

    # STEP1とSTEP2、どちらかが実行されていないとき
    else:
        # 在庫の初期値を0とする
        zaiko_teian_hako = 0
        zaiko_teian_buhin = 0

    
    # フォーム作成
    with st.sidebar.form(key='form_zaiko_limit_calc3'):

        # 箱数選択用セレクトボックス
        # 100以上の在庫数ありえるので最大100だとエラー出る
        selected_zaiko_hako = st.selectbox("工場内の在庫数（箱）を入力してください", list(range(0,500)),
                                        index = zaiko_teian_hako,
                                            help="初期値は順立装置の現在庫数（箱）に設定されています")

        #
        selected_zaiko_buhin = st.selectbox("工場内の在庫数（部品数）を入力してください", list(range(0,50000)),
                                        index = zaiko_teian_buhin,
                                            help="初期値は順立装置の現在庫数（部品数）に設定されています")

        submit_button_step3 = st.form_submit_button("登録する")

    # 登録するボタンが押された時
    if submit_button_step3:

        st.session_state.selected_zaiko_hako_for_zaiko_limit_calc3 = selected_zaiko_hako
        st.session_state.selected_zaiko_buhin_for_zaiko_limit_calc3 = selected_zaiko_buhin
        
        st.sidebar.success(f"入力された在庫数（箱）: {st.session_state.selected_zaiko_hako_for_zaiko_limit_calc3}")
        st.sidebar.success(f"入力された在庫数（部品の数）: {st.session_state.selected_zaiko_buhin_for_zaiko_limit_calc3}")
        st.sidebar.info("🎉 処理が完了しました！")
            
    # 登録するボタンが押されなかった時
    else:

        if st.session_state.selected_zaiko_hako_for_zaiko_limit_calc3 == "":
            st.sidebar.warning("在庫数（箱or部品の数）を入力してください")

        else:
            st.sidebar.success(f"過去に入力された在庫数（箱）: {st.session_state.selected_zaiko_hako_for_zaiko_limit_calc3}")
            st.sidebar.success(f"過去に入力された在庫数（部品の数）: {st.session_state.selected_zaiko_buhin_for_zaiko_limit_calc3}")

    st.sidebar.title("ステップ４：需要調整")

    # フォームの作成
    with st.sidebar.form(key='form_zaiko_limit_calc4'):
        st.write("日量をご選択ください")

        # 2つのカラムを横並びに作成
        col1, col2 = st.columns(2)

        # 各カラムにボタンを配置
        with col1:
            submit_button_step4_mode1 = st.form_submit_button("日量を採用する",help="通常の日量を使用する")
        with col2:
            submit_button_step4_mode2 = st.form_submit_button("日量MAXを採用する",help="最大値の日量を使用する（生産数が多い場合で計算したい）")

    #　在庫リミット設定
    hinban_info = ["", ""]  # 2つの空文字列で初期化
    hinban_info[0], hinban_info[1] = (st.session_state.selected_hinban_for_zaiko_limit_calc1).split('_')
    target_column = "納入予定日時"
    if st.session_state.start_datetime_for_zaiko_limit_calc2 != "":
        #! 15分単位なので00分形式に変換（日量データ読み込み用）
        start_datetime_00 = (st.session_state.start_datetime_for_zaiko_limit_calc2).replace(minute=0, second=0).strftime('%Y-%m-%d %H:%M:%S')
        #! 15分単位の変数
        start_datetime = (st.session_state.start_datetime_for_zaiko_limit_calc2).strftime('%Y-%m-%d %H:%M:%S')
    selected_zaiko_hako = st.session_state.selected_zaiko_hako_for_zaiko_limit_calc3
    selected_zaiko_buhin = st.session_state.selected_zaiko_buhin_for_zaiko_limit_calc3

    # 日量情報を表示
    if st.session_state.start_datetime_for_zaiko_limit_calc2 != "":
        time_granularity = '15min'
        tehai_data_df = compute_hourly_tehai_data_by_hinban( hinban_info, start_datetime_00, start_datetime_00, time_granularity, flag_useDataBase, kojo)
        #st.dataframe(tehai_data_df)

        nitiryo = tehai_data_df['日量数'].iloc[0]
        nitiryo_max = tehai_data_df['月末までの最大日量数'].iloc[0]

        # 2列に分けて表示する
        col1, col2 = st.sidebar.columns(2)

        # 左側の列に1つ目のmetricを表示
        with col1:
            st.metric(label="日量数", value = nitiryo)

        # 右側の列に2つ目のmetricを表示
        with col2:
            st.metric(label="現在日から月末までの最大日量数", value = nitiryo_max)

    # フォームの送信処理
    if submit_button_step4_mode1:
        st.sidebar.success("日量が採用されました")
        out_parameter = "日量を採用する"
        # 在庫リミット計算実行
        show_results_of_zaiko_limit(hinban_info, target_column, start_datetime, selected_zaiko_hako, selected_zaiko_buhin, out_parameter, kojo, flag_useDataBase)
        st.sidebar.info("🎉 処理が完了しました！")

    if submit_button_step4_mode2:
        st.sidebar.success("日量MAXが採用されました")
        out_parameter = "日量MAXを採用する"
        # 在庫リミット計算実行
        show_results_of_zaiko_limit(hinban_info, target_column, start_datetime, selected_zaiko_hako, selected_zaiko_buhin, out_parameter, kojo, flag_useDataBase)
        st.sidebar.info("🎉 処理が完了しました！")

    # 両方のボタンが押されていなかった場合のメッセージ
    if not submit_button_step4_mode1 and not submit_button_step4_mode2:
        st.sidebar.warning("日量をご選択ください") 

#MARK: 在庫予測を行うページ設定
def page_of_future_zaiko_calc():

    # カスタムCSSを適用して画面サイズを設定する
    apply_custom_css()

    # ページタイトル
    st.title("在庫予測")

    # キャッシュ削除
    clear_session_button()

    st.sidebar.title("シミュレーション設定")

    # 開始日
    if 'start_date_for_future_zaiko_calc1' not in st.session_state:
        st.session_state.start_date_for_future_zaiko_calc1 = dt.date.today()

    # 開始時間
    if 'start_time_for_future_zaiko_calc1' not in st.session_state:
        current_time = dt.datetime.now().time()
        st.session_state.start_time_for_future_zaiko_calc1 = dt.time(current_time.hour, 0)

    # 開始日時
    if 'start_datetime_for_future_zaiko_calc1' not in st.session_state:
        st.session_state.start_datetime_for_future_zaiko_calc1 = ""

    if "change_rate" not in st.session_state:
        st.session_state.change_rate = 0

    # 折り畳み可能なメッセージ

    with st.sidebar.form(key='form_future_zaiko_calc1'):

        # 開始日
        st.session_state.start_date_for_future_zaiko_calc1 = st.date_input("開始日",
                                                     st.session_state.start_date_for_future_zaiko_calc1,
                                                     help="初期設定は現在日です")
        
        # 開始時間の選択肢をセレクトボックスで提供
        hours = [f"{i:02d}:00" for i in range(24)]
        start_time_str = st.selectbox("開始時間", hours,
                                       index=st.session_state.start_time_for_future_zaiko_calc1.hour,
                                       help="初期設定は現在時間です")
        
        # 選択された時間をdt_timeオブジェクトに変換
        start_time_hours = int(start_time_str.split(":")[0])

        # 時間を更新
        st.session_state.start_time_for_future_zaiko_calc1 = dt.time(start_time_hours, 0)

        # フォームの送信ボタン
        submit_button_step1 = st.form_submit_button(label='登録する')
    
        # 開始日時と終了日時を結合
        start_datetime = dt.datetime.combine(st.session_state.start_date_for_future_zaiko_calc1,
                                              st.session_state.start_time_for_future_zaiko_calc1)
    
    # ボタンを押された時
    if submit_button_step1:
    
        # フォームを送信したらsession_stateに保存
        st.session_state.start_datetime_for_future_zaiko_calc1 = start_datetime
        st.sidebar.success(f"選択した日時：{st.session_state.start_datetime_for_future_zaiko_calc1}")

        hinban_info = ["", ""]  # 2つの空文字列で初期化
        #hinban_info[0], hinban_info[1] = (st.session_state.selected_hinban_for_zaiko_limit_calc1).split('_')
        run_mode = "手動実行"
        target_column = "納入予定日時"
        out_parameter = "日量を採用する"
        if st.session_state.start_datetime_for_future_zaiko_calc1 != "":
            start_datetime = (st.session_state.start_datetime_for_future_zaiko_calc1).strftime('%Y-%m-%d %H:%M:%S')
        show_results_of_future_zaiko(target_column,  start_datetime, run_mode, out_parameter, kojo, flag_useDataBase)

        st.sidebar.info("🎉 処理が完了しました！")
    
    # それ以外
    else:
        st.sidebar.warning("開始時間を入力し、「登録する」ボタンを押してください")

#MARk: 自動実行モード
def page_of_auto_run():

    # 画面サイズ設定
    apply_custom_css()

    st.sidebar.title("ステップ１：スケジュール設定")

    # スケジュール設定
    # サイドバーにフォームを作成
    with st.sidebar.form(key="time_selection_form"):

        # 時間リストを作成
        hours = [f"{h:02d}:00" for h in range(24)]  # "00:00", "01:00", ..., "23:00"

        selected_hours = st.multiselect(
            "実行したい時刻を選んでください（1時間単位）",
            options=hours,
            default=["10:00", "11:00", "12:00"]  # デフォルト
        )
        submitted = st.form_submit_button("実行時刻を確定する")

        # 下にステータス表示
        if submitted:
            st.info("スケジュール設定を完了しました")
        else:
            st.warning("スケジュール設定を行ってください")

    st.sidebar.title("ステップ２：開始 or 停止")

    # 定期予測の処理状態を保存するセッション変数
    if 'processing' not in st.session_state:
        st.session_state.processing = False

    #　定期予測の開始停止設定
    with st.sidebar.form(key="control_form"):

        # 開始状態にする
        def start_processing():
            st.session_state.processing = True

        # 停止状態にする
        def stop_processing():
            st.session_state.processing = False

        st.write("以下のボタンで定期予測を開始または停止できます。")

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

    # 処理状態に応じてHTMLの内容を変更
    if st.session_state.processing:
        display_loader = "block"
        status_tag = "処理中"
        main_title = "在庫予測を定期実行中"
        description = "在庫予測は1時間毎に実行されます。"
        # 処理中の時だけ日時を取得
        now = dt.datetime.now()
        formatted_datetime = now.strftime("%Y年%m月%d日 %H:%M")

    else:
        display_loader = "none"
        status_tag = "待機中"
        main_title = "在庫定期予測を停止中"
        description = "「開始」ボタンをクリックして処理を開始してください。"
        formatted_datetime = "まだ開始されていません"

    # HTMLテンプレート読み込み
    # ★相対パスで読み込み
    # with open("../frontend/periodic_forecasting_page.html", "r", encoding="utf-8") as f:
    #     html_template = f.read()
    # 現在のファイルの絶対パスを取得
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # 目的のファイルの絶対パスを作成
    html_path = os.path.join(current_dir, "..", "frontend", "periodic_forecasting_page.html")
    # HTMLテンプレート読み込み
    with open(html_path, "r", encoding="utf-8") as f:
        html_template = f.read()

    # 変数埋め込み
    html_rendered = html_template.format(
        main_title=main_title,
        description=description,
        current_datetime=formatted_datetime,
        status_tag=status_tag,
        display_loader=display_loader
    )

    # 定期予測HTMLをStreamlitで表示
    st.components.v1.html(html_rendered, height=600, scrolling=True)

    # 予測結果を確認する

    st.sidebar.title("ステップ３：結果の確認")

    # セッションステートの初期化（必要なら）
    if 'check_results' not in st.session_state:
        st.session_state.check_results = False

    # 関数（コールバック）
    def check_results():
        st.session_state.check_results = True

    # サイドバー内にフォームを作成
    with st.sidebar.form(key="results_form"):
        
        st.write("以下ボタンで結果を確認できます")

        # フォーム内のボタン
        check_button = st.form_submit_button("📂　フォルダーを開く　")

        # ボタンが押されたら処理実行
        if check_button:
            check_results()

        # フォーム外で処理を実行（セッションステートによる判定）
        if st.session_state.check_results:
            os.startfile(RESULT_FOLDER_PATH)  # フォルダーを開く

        # 下にステータス表示
        if st.session_state.check_results:
            st.info("エクスプローラーでフォルダーを開いています。")
        else:
            st.warning("予測結果を確認する場合は、「フォルダーを開く」を押してください。")

    # -------------　定期予測の処理開始

    # 無限ループ
    if st.session_state.processing:

        # === 実行したい時刻をリストで指定（24時間表記） ===
        # 例えば ["10:00", "11:00", "12:00"] とか
        #target_times = ["00:57", "01:06", "22:50"]
        target_times = selected_hours  # ← ユーザーが選んだものを使う
        print(target_times)

        # 実行済みフラグ（同じ時刻に複数回動かないようにする）
        executed_times = set()

        #　実行する処理
        def job(run_time):

            current_time = dt.datetime.now()
            now = current_time.replace(minute=0, second=0, microsecond=0)

            st.write(now)

            print(f"{now.strftime('%Y-%m-%d %H:%M:%S')} に {run_time} を実行！")

            # ここにやりたい処理を書く
            df = pd.DataFrame({
                '実行時刻': [now.strftime('%Y-%m-%d %H:%M:%S')],
                'データ': [42]
            })

            # df = forecast_v3.show_zaiko_simulation( now,1)

            # ファイル名は実行時間ベースでOK
            filename = f"{RESULT_FOLDER_PATH}/data_{now.strftime('%Y%m%d_%H%M')}.csv"
            df.to_csv(filename, index=False)
            print(f"{filename} を保存しました。\n")

        print("=== 指定した時刻に処理を実行します ===")

        # 無限ループ
        while True:
            now = dt.datetime.now()
            current_time_str = now.strftime("%H:%M")

            # 実行予定時刻と一致したら
            if current_time_str in target_times and current_time_str not in executed_times:
                job(current_time_str)
                executed_times.add(current_time_str)  # 一度実行したら記録する

            # 翌日になったらリセット
            if current_time_str == "00:00":
                executed_times.clear()

            time.sleep(1)  # 1秒ごとにチェック

# MARK: 全体ページ構成決定
def main():

    # ヘッダー
    # ★相対パスで読み込み
    # with open("../frontend/letter_glitch.html", "r", encoding="utf-8") as f:
    #         html_top_page = f.read()
    # st.components.v1.html(html_top_page, height=120, scrolling=True)
    # 現在のファイルの絶対パスを取得
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # 目的のファイルの絶対パスを作成
    html_path = os.path.join(current_dir, "..", "frontend", "letter_glitch.html")
    # ファイルを開く
    with open(html_path, "r", encoding="utf-8") as f:
        html_top_page = f.read()
    st.components.v1.html(html_top_page, height=120, scrolling=True)

    # サイドバーの定義
    # タイトル
    st.sidebar.title("メインメニュー")
    # メインメニューの選択
    # ↓　このリストをいじれば各ページを非表示にできる
    main_menu = st.sidebar.radio("ページ選択", ["🏠 ホーム","🖥️ 自動実行モード", "🔍 可視化", "📊 分析", "⏳ 予測", "📖 マニュアル"])
    # サイドバーに折り返し線を追加
    st.sidebar.markdown("---")

    # メインメニューの選択に応じた処理
    if main_menu == "🏠 ホーム":
        page = "🏠 ホーム"

    elif main_menu == "🖥️ 自動実行モード":
        page = "🖥️ 自動実行モード"

    elif main_menu == "🔍 可視化":
        #page = "🔍 可視化"
        main_menu_visual = st.sidebar.radio("可視化ページ選択", ["仕入先ダイヤマスター準備支援"], key='analysis')
        page = main_menu_visual

    elif main_menu == "📊 分析":
        # 分析のサブメニュー
        main_menu_analysis = st.sidebar.radio("分析ページ選択", ["要因分析（特徴量重要度活用ver）"], key='analysis')
        page = main_menu_analysis

    elif main_menu == "⏳ 予測":
        # 予測のサブメニュー
        main_menu_prediction = st.sidebar.radio("予測ページ選択", ["在庫リミット計算", "在庫予測"], key='prediction')
        page = main_menu_prediction

    elif main_menu == "📖 マニュアル":
        page = "📖 マニュアル"

    # 詳細ページの設定
    if page == "🏠 ホーム":

        # 画面サイズ設定
        apply_custom_css()

        # 在庫予測定期実行HTML挿入

        #! いつか公開
        #st.sidebar.header("管理メニュー")
        # # 折り畳み可能なメッセージ
        # with st.sidebar.expander("💡 ヘルプ "):
        #     st.write("ここに詳細情報を記載する。クリックすると折り畳み/展開が切り替わります。")
        #     #st.image("https://via.placeholder.com/150", caption="例画像")

        # HTML描画（タイトルなど）
        # ★相対パスで読み込み
        # with open("../frontend/home_page.html", "r", encoding="utf-8") as f:
        #     html_top_page = f.read()
        # st.components.v1.html(html_top_page, height=1200, scrolling=True)
        # 現在のファイルの絶対パスを取得
        current_dir = os.path.dirname(os.path.abspath(__file__))
        # 目的のファイルの絶対パスを作成
        html_path = os.path.join(current_dir, "..", "frontend", "home_page.html")
        # ファイルを開く
        with open(html_path, "r", encoding="utf-8") as f:
            html_top_page = f.read()
        st.components.v1.html(html_top_page, height=1200, scrolling=True)

    elif page == "仕入先ダイヤマスター準備支援":

        # 画面サイズ設定
        apply_custom_css()

    elif page == "🖥️ 自動実行モード":
        page_of_auto_run()

    elif page == "要因分析（特徴量重要度活用ver）":
        page_of_analyze_feature_importance()
    
    elif page == "在庫リミット計算":
        #page_of_zaiko_limit_calc()#1時間単位
        page_of_zaiko_limit_calc2()#15分単位

    elif page == "在庫予測":
        page_of_future_zaiko_calc()

    elif page == "📖 マニュアル":
        apply_custom_css()

        st.title("マニュアル")

        # 表示するPDFファイルのパス
        pdf_file_path = "操作マニュアル.pdf"  # ここに表示したいPDFのパスを指定
        doc = fitz.open(pdf_file_path)

        # PDFの各ページを画像に変換して表示
        for page_number in range(doc.page_count):
            page = doc.load_page(page_number)  # ページを読み込む
            pix = page.get_pixmap()  # ピクセルマップを取得
            img = pix.tobytes("png")  # 画像としてバイトデータに変換
            st.image(img, caption=f"ページ {page_number + 1}", use_column_width=True)
        

#MARK: 本スクリプトが直接実行されたときに実行
if __name__ == "__main__":

    print("アプリを実行します")
    
    # アプリの立ち上げ
    main()