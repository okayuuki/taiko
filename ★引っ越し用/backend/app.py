import streamlit as st
import datetime as dt
import pandas as pd
import os
import time

# 自作ライブラリのimport（備忘：関数として認識されないときは、vscodeを再起動する）
from preprocess_data import compute_features_and_target
from apply_models import pipeline, show_zaiko_with_baseline, compute_feature_importance, show_feature_importance

# 命名規則
# - 拡張性を高めるために変数名は機能毎STEP毎に設定する

# 定期予測の結果を保存するフォルダー
RESULT_FOLDER_PATH = 'kari'

#! 仮！！！！！！！！！！！！！！！！！！！！！！！！！（品番情報表示）
def create_hinban_info():

    file_path = 'temp_activedata.csv'
    df = pd.read_csv(file_path, encoding='shift_jis')

    # ユニークな品番リストを作成
    df['品番'] = df['品番'].str.strip()
    unique_hinban_list = df['品番'].unique()

    # '品番' ごとに '整備室' のユニークな値を集める
    hinban_seibishitsu_df = df.groupby('品番')['整備室'].unique().reset_index()

    # '整備室' のユニークな値を行ごとに展開
    hinban_seibishitsu_df = hinban_seibishitsu_df.explode('整備室')

    #　ユニークな '品番_整備室' 列を作成
    hinban_seibishitsu_df['品番_整備室'] = hinban_seibishitsu_df.apply(lambda row: f"{row['品番']}_{row['整備室']}", axis=1)

    return hinban_seibishitsu_df

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
    if st.sidebar.button(label, key=key):
        st.session_state.clear()
        st.sidebar.info("🎉 セッションステートを初期化しました！")

#MARK: 特徴量重要度を活用して要因分析を行うページ設定
def page_of_analyze_feature_importance():

    # カスタムCSSを適用して画面サイズを設定する
    apply_custom_css()

    # タイトル
    st.title("在庫変動要因分析（特徴量重要度活用）")

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
            #analysis_v3.show_abnormal(  st.session_state.selected_date, st.session_state.selected_time)#ここに関数名
            #todo 異常の確認結果
            #実行確認
            st.write(st.session_state.selected_date_for_analyze_feature_importance0,
                      st.session_state.selected_time_for_analyze_feature_importance0)

        st.sidebar.info("🎉 処理が完了しました！")

    # 登録するボタンが押されなかったときの処理
    else:
        
        # まだ一度もSTEP0が実行されていない時
        if (
            (st.session_state.selected_date_for_analyze_feature_importance0 == "")
            and (st.session_state.selected_time_for_analyze_feature_importance0 == "")
        ):
            st.sidebar.code("このステップは任意です。スキップできます。")

        # 1度はボタン押されている
        else:
            st.sidebar.success(f"過去に選択された日: {st.session_state.selected_date_for_analyze_feature_importance0}")
            st.sidebar.success(f"過去に選択された時間: {st.session_state.selected_time_for_analyze_feature_importance0}")

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
        hinban_seibishitsu_df = create_hinban_info() #!　品番リスト 直す！！！！！！！！！！！！！！！！！！！

        # サイドバーに品番選択ボックスを作成
        product = st.selectbox("品番を選択してください", hinban_seibishitsu_df['品番_整備室'])
        
        # 「登録する」ボタンをフォーム内に追加
        submit_button_step1 = st.form_submit_button(label='登録する')

    # 登録するボタンが押されたときの処理
    if submit_button_step1:

        st.session_state.selected_hinban_for_analyze_feature_importance1 = product

        st.sidebar.success(f"新たに選択された品番: {st.session_state.selected_hinban_for_analyze_feature_importance1}")
        
        # パイプライン実行
        with st.spinner("実行中です。しばらくお待ちください..."):
            model, X, df = pipeline() #todo 品番と期間が指定される
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
        st.write(st.session_state.start_date_for_analyze_feature_importance2,
                 st.session_state.end_date_for_analyze_feature_importance2)
        
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
                                        st.session_state.end_date_for_analyze_feature_importance2, freq='H')
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

#MARK: 在庫リミット計算を行うページ設定
def page_of_zaiko_limit_calc():

    # カスタムCSSを適用して画面サイズを設定する
    apply_custom_css()

    # タイトル
    st.title("在庫リミット計算")

    # キャッシュ削除
    clear_session_button()

    # 選択品番
    if "selected_hinban_for_zaiko_limit_calc1" not in st.session_state:
        st.session_state.selected_hinban_for_zaiko_limit_calc1 = ""
 
    # サイドバートップメッセージ
    st.sidebar.write("## 🔥各ステップを順番に実行してください🔥")

    st.sidebar.title("ステップ１：品番選択")

    # フォーム作成
    with st.sidebar.form(key='form_zaiko_limit_calc1'):
    
        hinban_seibishitsu_df = create_hinban_info()

        # サイドバーに品番選択ボックスを作成
        unique_product = st.selectbox("品番を選択してください", hinban_seibishitsu_df['品番_整備室'])
        
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

        if st.session_state.selected_hinban_for_zaiko_limit_calc1 == "":
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
    # 在庫数
    if 'selected_zaiko_for_zaiko_limit_calc3' not in st.session_state:
        st.session_state.selected_zaiko_for_zaiko_limit_calc3 = ""

    # STEP1とSTEP2が実行されているとき
    if (st.session_state.selected_hinban_for_zaiko_limit_calc1 != "") and (st.session_state.start_datetime_for_zaiko_limit_calc2 != ""):

        #todo 対象品番の対象時間の在庫を読み込む
        st.write("対象品番の対象時間の在庫を読み込む")
        zaiko_teian = 0

    # STEP1とSTEP2、どちらかが実行されていないとき
    else:
        # 在庫の初期値を0とする
        zaiko_teian = 0

    # フォーム作成
    with st.sidebar.form(key='form_zaiko_limit_calc3'):
        # 日時選択用セレクトボックス
        selected_zaiko = st.selectbox("工場内の在庫数（箱）を入力してください", list(range(0,100)),
                                        index = zaiko_teian,
                                            help="初期値は順立装置の現在庫数（箱）に設定されています")
        submit_button_step3 = st.form_submit_button("登録する")

    # 登録するボタンが押された時
    if submit_button_step3:

        st.session_state.selected_zaiko_for_zaiko_limit_calc3 = selected_zaiko
        
        st.sidebar.success(f"入力された在庫数（箱）: {st.session_state.selected_zaiko_for_zaiko_limit_calc3}")
        st.sidebar.info("🎉 処理が完了しました！")
            
    # 登録するボタンが押されなかった時
    else:

        if st.session_state.selected_zaiko_for_zaiko_limit_calc3 == "":
            st.sidebar.warning("在庫数（箱）を入力してください")

        else:
            st.sidebar.success(f"過去に入力された在庫数（箱）: {st.session_state.selected_zaiko_for_zaiko_limit_calc3}")

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

    # フォームの送信処理
    if submit_button_step4_mode1:
        st.sidebar.success("日量が採用されました")
        #forecast_v3.show_forecast(unique_product,start_datetime,selected_zaiko, LT, 0)
        st.sidebar.info("🎉 処理が完了しました！")

    if submit_button_step4_mode2:
        st.sidebar.success("日量MAXが採用されました")
        #forecast_v3.show_forecast(unique_product,start_datetime,selected_zaiko, LT, 1)
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

        #forecast_v3.show_zaiko_simulation( st.session_state.start_datetime, st.session_state.change_rate)

        st.sidebar.info("🎉 処理が完了しました！")
    
    # それ以外
    else:
        st.sidebar.warning("開始時間を入力し、「登録する」ボタンを押してください")

# MARK: 全体ページ構成決定
def main():

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
        main_menu_visual = st.sidebar.radio("可視化ページ選択", ["上下限外れ確認", "関所別かんばん数可視化（アニメーション）", "フレ可視化"], key='analysis')
        page = main_menu_visual

    elif main_menu == "📊 分析":
        # 分析のサブメニュー
        main_menu_analysis = st.sidebar.radio("分析ページ選択", ["要因分析（特徴量重要度活用）"], key='analysis')
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

        st.sidebar.header("管理メニュー")

        # 折り畳み可能なメッセージ
        with st.sidebar.expander("💡 ヘルプ "):
            st.write("ここに詳細情報を記載する。クリックすると折り畳み/展開が切り替わります。")
            #st.image("https://via.placeholder.com/150", caption="例画像")

        # HTML描画（タイトルなど）
        with open("top_page.html", "r", encoding="utf-8") as f:
            html_top_page = f.read()
        st.components.v1.html(html_top_page, height=800, scrolling=True)


    elif page == "🖥️ 自動実行モード":

        # 画面サイズ設定
        apply_custom_css()

        # 定期予測の処理状態を保存するセッション変数
        if 'processing' not in st.session_state:
            st.session_state.processing = False

        #　ボタン設定
        with st.sidebar.form(key="control_form"):

            # 開始状態にする
            def start_processing():
                st.session_state.processing = True

            # 停止状態にする
            def stop_processing():
                st.session_state.processing = False

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
        with open("periodic_forecasting_page.html", "r", encoding="utf-8") as f:
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
        st.components.v1.html(html_rendered, height=800, scrolling=True)

        # 予測結果を確認する

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
                os.startfile(RESULT_FOLDER_PATH)  # フォルダーを開く

            # 下にステータス表示
            if st.session_state.check_results:
                st.info("エクスプローラーでフォルダーを開いています。")
            else:
                st.warning("予測結果を確認する場合は、「フォルダーを開く」を押してください。")

        # -------------　定期予測の処理開始

        #無限ループ
        if st.session_state.processing:

            # === 実行したい時刻をリストで指定（24時間表記） ===
            # 例えば ["10:00", "11:00", "12:00"] とか
            target_times = ["00:57", "01:06", "22:50"]

            # 実行済みフラグ（同じ時刻に複数回動かないようにする）
            executed_times = set()

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
    
    elif page == "要因分析（特徴量重要度活用）":
        page_of_analyze_feature_importance()
    
    elif page == "在庫リミット計算":
        page_of_zaiko_limit_calc()

    elif page == "在庫予測":
        page_of_future_zaiko_calc()

    elif page == "📖 マニュアル":
        apply_custom_css()
        compute_features_and_target()

#MARK: 本スクリプトが直接実行されたときに実行
if __name__ == "__main__":

    print("アプリを実行します")
    
    # アプリの立ち上げ
    main()