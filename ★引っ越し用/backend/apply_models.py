import streamlit as st
import pandas as pd
import json
import re
import datetime as dt
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import shap
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error 
import lightgbm as lgb
from lightgbm import LGBMRegressor
import matplotlib.pyplot as plt

# 自作ライブラリのimport
from preprocess_data import compute_features_and_target

#MARK: 機械学習のパイプライン
def pipeline(hinban_info, start_datetime, end_datetime, time_granularity, flag_useDataBase, kojo):

    # 目的変数と説明変数を作成する
    df = compute_features_and_target(hinban_info, start_datetime, end_datetime, time_granularity, flag_useDataBase, kojo)
    st.header("統合テーブル生成（特徴量計算まで完了）")
    st.dataframe(df)

    # 学習
    # 必要変数を返す
    # modelはSHAP値計算用、X、dfは結果描画用
    # ランダムフォレスト
    #model, X, df = train_random_forest(df)
    # lightgbm
    model, X, df = train_lightgbm(df)

    return model, X, df

#MARK: ランダムフォレストモデルを適用する
def train_random_forest(df):

    # 説明変数の設定
    feature_columns = [col for col in df.columns if 'feature' in col]
    print(feature_columns)
    # 抽出された列を使って新しいデータフレームを作る
    X = df[feature_columns]

    # 目的変数の設定
    target_columns = [col for col in df.columns if 'target' in col]
    # 抽出された列を使って新しいデータフレームを作る
    y = df[target_columns]

    # 学習
    X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.05, random_state=42)

    # 機械学習モデルの適用
    # 今はランダムフォレスト
    model = RandomForestRegressor(n_estimators=10, max_depth=30, random_state=42)
    model.fit(X_train, y_train.squeeze().values)

    # 誤差
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)

    print(mae)

    return model, X, df

#MARK: lightgbmを適用する
def train_lightgbm(df):

    """
    LightGBMを用いてモデルを学習する関数
    
    Parameters:
    -----------
    df : pandas.DataFrame
        学習用データフレーム
    
    Returns:
    --------
    tuple
        (学習済みモデル, 説明変数, 元のデータフレーム)
    """

    # 説明変数の設定
    feature_columns = [col for col in df.columns if 'feature' in col]
    #print("使用する特徴量:", feature_columns)
    X = df[feature_columns]

    # 目的変数の設定
    target_columns = [col for col in df.columns if 'target' in col]
    y = df[target_columns]

    # 学習データとテストデータの分割
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.05, random_state=42
    )

    # 特徴量ごとの制約を定義
    # constraint_dict = {
    #     "feature_No1_入庫予定かんばん数_スナップ済": 1,   # 単調増加
    #     'feature_No2_最近の着工数の状況': -1,  # 単調減少
    #     'feature_No3_過去のかんばん入出差分': 1,    # 単調増加
    #     "feature_No4_西尾東~部品置き場の間の滞留かんばん数": -1,  # 単調減少
    # }
    # feature_columnsの順序に合わせてリストを作成
    #monotone_constraints = [constraint_dict[col] for col in feature_columns]

    # 列名のパターンに基づいて制約を定義する関数
    def get_constraint(column_name):
        if column_name.startswith('feature_No1'):  # feature_No1で始まる列
            return 1    # 単調増加
        elif column_name.startswith('feature_No2'):  # feature_No2で始まる列
            return -1   # 単調減少
        elif column_name.startswith('feature_No3'):  # feature_No3で始まる列
            return 1    # 単調増加
        elif column_name.startswith('feature_No4'):  # feature_No4で始まる列
            return -1   # 単調減少
        elif column_name.startswith('feature_No5'):  # feature_No3で始まる列
            return 1    # 単調増加
        else:
            return 0    # その他の列

    # 制約リストの作成
    monotone_constraints = [get_constraint(col) for col in feature_columns]

    # LightGBMモデルの学習
    model = LGBMRegressor(
        n_estimators=100,
        max_depth=30,
        random_state=42,
        monotone_constraints=monotone_constraints
    )
    model.fit(X_train, y_train.squeeze().values)

    # 予測と評価
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    #print(f"Mean Absolute Error: {mae}")

    return model, X, df

#MARK: 特徴量重要度（SHAP値）の計算
def compute_feature_importance( model, X):

    explainer = shap.TreeExplainer( model, model_output='raw')
    shap_values = explainer.shap_values(X)

    # # SHAPのサマリ結果の表示
    # fig, ax = plt.subplots()
    # shap.summary_plot(shap_values, X, feature_names=X.columns, show=False)
    # # プロットをStreamlitで表示
    # st.pyplot(fig)

    # # SHAP値を標準化
    # shap_values_std = np.zeros_like(shap_values)
    # for i in range(shap_values.shape[1]):  # 各特徴量について
    #     values = shap_values[:, i]
    #     shap_values_std[:, i] = (values - np.mean(values)) / np.std(values)

    # # SHAPのサマリ結果の表示
    # fig, ax = plt.subplots()
    # shap.summary_plot(shap_values_std, X, feature_names=X.columns, show=False)
    # # プロットをStreamlitで表示
    # st.pyplot(fig)

    return shap_values

#　在庫推移を基準線と共に描画する
# todo  フロントは別ファイル？ 適切なファイルに置く？
def show_zaiko_with_baseline( merged_data_df, start_datetime, end_datetime, highlight_datetime):

    # ヘッダー表示
    st.header('在庫推移')

    #  日時カラムをdatetime型に変換
    merged_data_df['日時'] = pd.to_datetime(merged_data_df['日時'], errors='coerce')
    #st.dataframe(merged_data_df)

    #  指定した日時範囲でデータをフィルタ
    filtered_merged_data_df = merged_data_df[(merged_data_df['日時'] >= start_datetime) & (merged_data_df['日時'] <= end_datetime)]
    # 実行結果の確認
    # st.dataframe(filtered_merged_data_df)

    #  カラムを指定
    zaiko_col = '在庫数（箱）'  
    regular_zaiko_col = 'いつもの在庫数（箱）'
    min_col = '設計値MIN'
    max_col = '設計値MAX'

    # 在庫折れ線グラフの初期化
    fig = go.Figure()

    # ======== ハイライト用に色と透明度を動的に設定 ========
    if highlight_datetime is not None:
        # 完全一致で比較
        is_highlight = (filtered_merged_data_df['日時'] == highlight_datetime)
        # バーごとの不透明度: 一致するバー → 1.0, それ以外 → 0.3
        bar_opacities = np.where(is_highlight, 1.0, 0.1)

        zaiko_suii = "在庫数（箱）※選択時刻をハイライト"
        
    else:
        # ハイライトなし（全バー同じ不透明度）
        bar_opacities = [0.3]*len(filtered_merged_data_df)
        zaiko_suii = "在庫数（箱）"

    # 在庫数を棒グラフで表示
    fig.add_trace(go.Bar(
        x=filtered_merged_data_df['日時'].dt.strftime('%Y-%m-%d-%H'),
        y=filtered_merged_data_df[zaiko_col], 
        marker=dict(color='blue', opacity=bar_opacities),
        name=zaiko_suii))
    
    # グラフに表示    
    fig.add_trace(go.Scatter(
        x=filtered_merged_data_df['日時'].dt.strftime('%Y-%m-%d-%H'),
        y=filtered_merged_data_df[regular_zaiko_col],
        mode='lines+markers',
        name='いつもの在庫数（箱）',
        line=dict(color='gray')# 線の色をグレーに指定
    ))

    # 0のラインを赤線で追加
    fig.add_trace(go.Scatter(
        x=filtered_merged_data_df['日時'].dt.strftime('%Y-%m-%d-%H'),
        y=np.zeros(len(filtered_merged_data_df)),  # line_dfの長さに合わせて0の配列を作成
        mode='lines',
        name='在庫0',
        line=dict(color="#D70000", width=3)
    ))

    # 設計値MINのラインを追加
    fig.add_trace(go.Scatter(
        x=filtered_merged_data_df['日時'].dt.strftime('%Y-%m-%d-%H'),
        y=filtered_merged_data_df[min_col],  
        mode='lines',
        name='設計値MIN',
        line=dict(color="#FFA500", width=3)
    ))

    # 設計値MAXのラインを追加
    fig.add_trace(go.Scatter(
        x=filtered_merged_data_df['日時'].dt.strftime('%Y-%m-%d-%H'),
        y=filtered_merged_data_df[max_col],  
        mode='lines',
        name='設計値MAX',
        line=dict(color="#32CD32", width=3)
    ))

    fig.update_layout(
        hovermode='x unified'  # ← ホバーをX軸で統一（おすすめ）
    )

    # グラフを表示
    st.plotly_chart(fig, use_container_width=True)

# 特徴量重要度の結果を描画する
# todo  フロントは別ファイル？適切なファイルに置く？
def show_feature_importance( merged_data_df, X, selected_datetime, shap_values):

    # 開始・終了時間を計算して「対象期間」列を作る関数
    def extract_time_range(var_name, selected_datetime):

        # 正規表現で t-数字~t-数字 を抽出
        match = re.search(r't-(\d+)~t-(\d+)', var_name)
        
        if match:
            start_time = int(match.group(1))
            end_time = int(match.group(2))
            
            # timedelta で引き算（時間単位）
            start_dt = selected_datetime - dt.timedelta(hours=start_time)
            end_dt = selected_datetime - dt.timedelta(hours=end_time)
            
            # フォーマット指定
            start_str = start_dt.strftime('%Y-%m-%d %H:%M')
            end_str = end_dt.strftime('%Y-%m-%d %H:%M')
            
            # 開始と終了を統合（対象期間）
            return f'{start_str} ~ {end_str}'
        
        else:
            # パターンが見つからない場合は空
            return 'N/A'
    
    # 在庫増（ランキング用）
    # マッピングを '変数名' 列に基づいて '要因名' 列に適用する関数を定義
    def map_increase_factor_outline(variable):
        if "No1_" in variable:
            return "「納入数が多い」"
        elif "No2_" in variable:
            return "「生産が少ない」"
        elif "No3_" in variable:
            return "「かんばんが多い」"
        elif "No4_" in variable:
            return "「None_格納遅れがない」"
        elif "No5_" in variable:
            return "「設計時間外の入庫が多い」"
        else:
            return None

    # 在庫減（ランキング用）
    # マッピングを '変数名' 列に基づいて '要因名' 列に適用する関数を定義
    def map_decrease_factor_outline(variable):
        if "No1_" in variable:
            return "「納入数が少ない」"
        elif "No2_" in variable:
            return "「生産が多い」"
        elif "No3_" in variable:
            return "「かんばんが少ない」"
        elif "No4_" in variable:
            return "「格納遅れ」"
        elif "No5_" in variable:
            return "「設計時間外の入庫が少ない」"
        else:
            return None

    # 在庫増（詳細用）
    # マッピングを '変数名' 列に基づいて '要因名' 列に適用する関数を定義
    def map_increase_factor_detail(variable):
        if "No1_" in variable:
            return """
                    <strong>【発注納入要因】入庫予定かんばん数が多い</strong><br>
                    ＜説明＞<br>
                    この値は、ある時点tの在庫数に影響を与える納入要因として、t-τの時点に予定されていた納入かんばん数を示しています。<br>
                    ここでのτは納入から入庫までの基準リードタイムに相当し、t時点で在庫数に影響を与える納入かんばん数が過去にどれだけ存在していたかを表しています。<br>
                    ＜考えられる事象＞<br>
                    ・便Aveより納入かんばん数が多い<br>
                    ・仕入先挽回納入
                    """
        elif "No2_" in variable:
            return """
                    <strong>【生産要因】生産が少ない</strong><br>
                    ＜説明＞<br>
                    この値は、ある時点tの在庫数に影響を与える生産要因として、特定の時間帯における1分辺りの平均生産台数を表したものです。<br>
                    これは、ある時点tの在庫数に影響を与える短期減衰要因として働くと考えられます。<br>
                    ＜考えられる異常＞<br>
                    ・ライン停止<br>
                    ・生産変動/計画変更/得意先の需要減
                    """
        elif "No3_" in variable:
             return """
                    <strong>【かんばん要因】過去（かんばん回転日数前）のかんばん数が多い</strong><br>
                    この値は、ある時点tにおいて、t-かんばん回転日数以前に発注されたかんばんのうち、<br>
                    順立装置の入庫予定時間<=t<出庫予定時間を満たすかんばんの数をカウントしたものです。<br>
                    これはある時点tにおける順立装置の在庫水準を表現したもので、在庫の土台を決定する要因（ベース要因）として働くと考えられます。<br>
                    1日単位や1週間単位といった比較的長めのスパンで在庫の増減を決定づける要因です。<br>
                    ＜考えられる事象＞<br>
                    ・過去の生産が多いことで発注増<br>
                    ・臨時かんばんの発行
                    """
        elif "No4_" in variable:
            return "「格納遅れ（ここは表示されない）」"
        elif "No5_" in variable:
            return """
                    <strong>【入庫要因】今、入庫予定時間外の入庫が多い</strong><br>
                    この値は、ある時点tの在庫数に影響を与える要因として、通常の入庫予定（=過去の納入予定かんばん数に納入入庫LTを加味して予測される値）では説明できない、異常な入庫かんばん数を示したものです。<br>
                    すなわち、t時点に発生した入庫のうち、事前に予測されていなかった異常な増加分を抽出したもので、部品置き場からの入庫や前倒し入庫といった突発的な在庫増加の要因を捉えるものです。<br>
                    ＜考えられる事象＞<br>
                    ・滞留からの入庫がいつもより多い
                    """
        elif "No6_" in variable:
            return "「定期便が早着している」"
        elif "No7_" in variable:
            return "「間口の充足率が低い」"
        elif "No8_" in variable:
            return "「西尾東が部品置き場で滞留していない」"
        elif "No9_" in variable:
            return "「定期便にいつもよりモノが多い」"
        elif "No10_" in variable:
            return "「発注がある」"
        else:
            return None

    # 在庫減（詳細用）
    # マッピングを '変数名' 列に基づいて '要因名' 列に適用する関数を定義
    def map_decrease_factor_detail(variable):
        if "No1_" in variable:
            return """
                    <strong>【発注納入要因】入庫予定かんばん数が少ない</strong><br>
                    ＜説明＞<br>
                    この値は、ある時点tの在庫数に影響を与える納入要因として、t-τの時点に予定されていた納入かんばん数を示しています。<br>
                    ここでのτは納入から入庫までの基準リードタイムに相当し、t時点で在庫数に影響を与える納入かんばん数が過去にどれだけ存在していたかを表しています。<br>
                    ＜考えられる事象＞<br>
                    ・便Aveより納入かんばん数が少ない<br>
                    ・仕入先未納<br>
                    ・発注無し
                    """
        elif "No2_" in variable:
             return """
                    <strong>【生産要因】生産が多い</strong><br>
                    ＜説明＞<br>
                    この値は、ある時点tの在庫数に影響を与える生産要因として、特定の時間帯における1分辺りの平均生産台数を表したものです。<br>
                    これは、ある時点tの在庫数に影響を与える短期減衰要因として働くと考えられます。<br>
                    ＜考えられる異常＞<br>
                    ・生産変動/挽回生産<br>
                    ・計画変更/得意先の需要増<br>
                    """
        elif "No3_" in variable:
            return """
                    <strong>【かんばん要因】過去（かんばん回転日数前）のかんばん数が少ない</strong><br>
                    ＜説明＞<br>
                    要因の値は、ある時点tにおいて、t-かんばん回転日数以前に発注されたかんばんのうち、<br>
                    順立装置の入庫予定時間<=t<出庫予定時間を満たすかんばんの数をカウントしたものです。<br>
                    これはある時点tにおける順立装置の在庫水準を表現したもので、在庫の土台を決定する要因（ベース要因）として働くと考えられます。<br>
                    1日単位や1週間単位といった比較的長めのスパンで在庫の増減を決定づける要因です。<br>
                    ＜考えられる事象＞<br>
                    ・過去の生産が少ないことで発注減<br>
                    ・かんばんの出し忘れ<で発注減<br>
                    ・組立の取り忘れで発注減
                    """
        elif "No4_" in variable:
            return """
                    <strong>【滞留要因】今、西尾東BCから部品置き場の間で部品が滞留している</strong><br>
                    ＜説明＞<br>
                    入庫予定時間を経過しても入庫されていないかんばん数<br>
                    ＜考えられる事象＞<br>
                    ・順立装置の設備停止<br>
                    ・順立前の部品置き場で部品が残っている<br>
                    ・西尾東BCで誤転送<br>
                    ・工場ビットの部品OF<br>
                    ・西尾東で部品が残っている/定期便の乗り遅れ<br>
                    ・台風積雪などによるトラックの遅延<br>
                    など
                    """
        elif "No5_" in variable:
            return """
                    <strong>【入庫要因】今、入庫予定時間外の入庫が少ない</strong><br>
                    この値は、ある時点tの在庫数に影響を与える要因として、通常の入庫予定（=過去の納入予定かんばん数に納入入庫LTを加味して予測される値）では説明できない、異常な入庫予定かんばん数を示したものです。<br>
                    すなわち、t時点に発生した入庫のうち、事前に予測されていなかった異常な増加分を抽出したもので、部品置き場からの入庫や前倒し入庫といった突発的な在庫増加の要因を捉えるものです。
                    ＜考えられる事象＞<br>
                    ・滞留からの入庫がいつもより少ない
                    """
        elif "No6_" in variable:
            return "「定期便の遅着」"
        elif "No7_" in variable:
            return "「間口の充足率が高い」"
        elif "No8_" in variable:
            return "「西尾東BC or 部品置き場で滞留しているかんばん数が多い」"
        elif "No9_" in variable:
            return "「定期便にいつもよりモノが少ない」"
        elif "No10_" in variable:
            return "「発注がない」"
        else:
            return None

    #! shap_valuesはseries

    #  日時カラムをdatetime型に変換
    merged_data_df['日時'] = pd.to_datetime(merged_data_df['日時'], errors='coerce')

    # インデックスをリセット
    merged_data_df = merged_data_df.reset_index(drop=True)
    shap_values_df = pd.DataFrame(shap_values, columns=X.columns)#データフレームにする
    shap_values_df = shap_values_df.reset_index(drop=True)

    #  指定した日時範囲でデータをフィルタリング
    #  複数列あるので、データフレームのまま
    filtered_merged_data_df = merged_data_df[merged_data_df['日時'] == selected_datetime]
    #st.dataframe(filtered_merged_data_df)
    
    # -----指定した日時と一致する行を抽出-----

    # # 要因の値（説明変数X）
    # filtered_X = X.loc[filtered_merged_data_df.index[0]]# Pandas の .loc[]で1行のみを指定すると、seriesになる
    # filtered_X.name = '要因の値'
    # filtered_X.index.name = '変数名'
    # #st.dataframe(filtered_X)

    # 要因の値（説明変数X）#! 表示用の値に変更
    # "youin"を含む列だけを抽出
    youin_columns = [col for col in filtered_merged_data_df.columns if "youin" in col]
    # その列だけを取り出して新しいデータフレームに
    X = filtered_merged_data_df[youin_columns]
    filtered_X = X.loc[filtered_merged_data_df.index[0]]
    filtered_X.name = '要因の値'
    filtered_X.index.name = '変数名'
    #st.dataframe(filtered_X)

    # いつもの値（いつもの説明変数Xの値）
    average_X = X.mean()
    average_X.name = 'いつもの値（ベースライン）'
    average_X.index.name = '変数名'
    #st.dataframe(average_X)

    # 寄与度の値
    # 同様に特徴量重要度の方もフィルタリング
    filtered_shap_values = shap_values_df.loc[filtered_merged_data_df.index[0]]# Pandas の .loc[]で1行のみを指定すると、seriesになる
    filtered_shap_values.name = '寄与度（SHAP値）'
    filtered_shap_values.index.name = '変数名'
    # 寄与度結果の確認
    #st.dataframe(filtered_shap_values)

    # -----統合-----

    # データフレーム作成
    filtered_X = pd.DataFrame(filtered_X)
    average_X = pd.DataFrame(average_X)
    filtered_shap_values = pd.DataFrame(filtered_shap_values)
    # 変数名で結合
    result_df = filtered_X.merge(average_X, on = '変数名').merge(filtered_shap_values, on = '変数名')
    #st.dataframe(result_df)

    # -----情報追加-----

    # データフレームに「対象期間」列を追加
    result_df.reset_index(inplace=True) #変数名がインデックスになっているので、リセットしないと。下の処理で変数名列を指定できない
    result_df['対象期間'] = result_df['変数名'].apply(lambda x: extract_time_range(x, selected_datetime))
    #st.dataframe(result_df)

    # 「要因名」列を追加

    # 正の値で大きい上位と負の値で小さい（絶対値が大きい）上位を抽出
    top_increase_ranking_df = result_df[result_df['寄与度（SHAP値）'] > 0].nlargest(12, '寄与度（SHAP値）')
    top_decrease_ranking_df = result_df[result_df['寄与度（SHAP値）'] < 0].nsmallest(12, '寄与度（SHAP値）')

    # 増と減それぞれで'要因名' 列を作成
    top_increase_ranking_df['要因名_詳細'] = top_increase_ranking_df['変数名'].apply(map_increase_factor_detail)
    top_decrease_ranking_df['要因名_詳細'] = top_decrease_ranking_df['変数名'].apply(map_decrease_factor_detail)
    top_increase_ranking_df['要因名'] = top_increase_ranking_df['変数名'].apply(map_increase_factor_outline)
    top_decrease_ranking_df['要因名'] = top_decrease_ranking_df['変数名'].apply(map_decrease_factor_outline)
    # 実行結果の確認
    #st.dataframe(top_decrease_ranking_df)

    # 不要なやつ削除
    keyword = 'None'  # 部分一致のキーワード
    top_increase_ranking_df = top_increase_ranking_df[~top_increase_ranking_df['要因名'].str.contains(keyword, na=False)]
    top_decrease_ranking_df = top_decrease_ranking_df[~top_decrease_ranking_df['要因名'].str.contains(keyword, na=False)]

    # 順位列を追加する
    # 増
    top_increase_ranking_df.reset_index(drop=True, inplace=True)
    top_increase_ranking_df.index += 1
    top_increase_ranking_df['順位'] = top_increase_ranking_df.index
    # 減
    top_decrease_ranking_df.reset_index(drop=True, inplace=True)
    top_decrease_ranking_df.index += 1
    top_decrease_ranking_df['順位'] = top_decrease_ranking_df.index
    #st.dataframe(top_increase_ranking_df)

    # 順位、変数名、値だけを表示し、インデックスは消す
    top_increase_ranking_df = top_increase_ranking_df[['順位', '要因名', '要因名_詳細', '対象期間', '要因の値', 'いつもの値（ベースライン）', '寄与度（SHAP値）']]
    top_decrease_ranking_df = top_decrease_ranking_df[['順位', '要因名', '要因名_詳細', '対象期間', '要因の値', 'いつもの値（ベースライン）', '寄与度（SHAP値）']]
    #st.dataframe(top_increase_ranking_df)

    # 結果を描画する

    # ヘッダー表示
    st.header('要因分析')

    # 
    col1, col2, col3 = st.columns(3)
    base_zaiko = filtered_merged_data_df['いつもの在庫数（箱）'].values[0]
    zaiko = filtered_merged_data_df['在庫数（箱）'].values[0]
    col1.metric(label="選択された日時", value = selected_datetime.strftime('%Y-%m-%d %H:%M'))#, delta="1 mph")
    col2.metric(label="いつもの在庫数（箱）", value = int(base_zaiko))
    col3.metric(label="在庫数（箱）", value = int(zaiko), delta=f"{int(zaiko)-int(base_zaiko)} 箱（いつもの在庫数との差分）")

    # CSSスタイルの追加
    st.markdown("""
    <style>
        .sub-header {
            font-size: 1.5rem;
            color: #334155;
            margin-bottom: 1rem;
        }

    </style>
    """, unsafe_allow_html=True)

    # データを昇順で並べ替え（寄与度の高い順）
    top_increase_ranking_df = top_increase_ranking_df.sort_values(by='順位')
    top_decrease_ranking_df = top_decrease_ranking_df.sort_values(by='順位')
    #　実行結果の確認
    #st.dataframe(top_decrease_ranking_df)

    increse_color = "increase-factor-card"
    decrese_color = "decrease-factor-card"

    # 上位3カード描画
    def render_increse_top_factors(df, increse_color):

        # CSSスタイルの追加
        # 上位3位カードのデザイン決定
        st.markdown("""
        <style>
        .insights-container {
            display: block;
            width: 100%;
            max-width: 800px;
            margin: 0 auto;
        }
        .factor-card {
            background-color: white;
            border-radius: 12px;
            box-shadow: 0 10px 25px rgba(0,0,0,0.1);
            padding: 25px;
            transition: transform 0.3s ease;
            position: relative;
            overflow: hidden;
            margin-bottom: 20px;
        }
        .factor-card:hover {
            transform: scale(1.02);
        }
        .factor-rank {
            position: absolute;
            top: -10px;
            left: -10px;
            background-color: #4a4a4a;
            color: white;
            padding: 10px 20px;
            border-radius: 0 0 50px 0;
            font-size: 40px;
            font-weight: bold;
            width: 70px;
            height: 70px;
            display: flex;
            align-items: center;
            justify-content: center;
            z-index: 10;
        }
        .factor-name {
            font-size: 22px;
            font-weight: bold;
            color: #2c3e50;
            margin-bottom: 15px;
            padding-left: 90px;
        }
        .factor-details {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding-left: 90px;
        }
        .factor-contribution {
            font-size: 18px;
            color: #7f8c8d;
        }
        .progress-bar {
            width: 200px;
            height: 10px;
            background-color: #ecf0f1;
            border-radius: 5px;
            overflow: hidden;
        }
        .progress-fill {
            height: 100%;
            background-color: #3498db;
            border-radius: 5px;
        }
        .period {
            font-size: 14px;
            color: #95a5a6;
            margin-top: 10px;
            padding-left: 90px;
        }
        .increase-factor-card-1 { border-top: 6px solid #e74c3c; }
        .increase-factor-card-2 { border-top: 6px solid #e74c3c; }
        .increase-factor-card-3 { border-top: 6px solid #e74c3c; }
        .decrease-factor-card-1 { border-top: 6px solid #3498db; }
        .decrease-factor-card-2 { border-top: 6px solid #3498db; }
        .decrease-factor-card-3 { border-top: 6px solid #3498db; }
        </style>
        """, unsafe_allow_html=True)
        
        # 上位3つの要因を抽出（順位順にソート）
        top_factors = df.sort_values('順位').head(3)
        
        # インサイトコンテナの開始
        st.markdown('<div class="insights-container">', unsafe_allow_html=True)
        
        # 各要因のカードを生成
        for index, row in top_factors.iterrows():
            # カードのクラスを動的に設定
            card_class = f"factor-card {increse_color}-{row['順位']}"
            
            # 寄与度から進捗バーの幅を計算（例：最大75%と仮定）
            if row['寄与度（SHAP値）'] >= 0:
                # 正の場合（現行ロジック）
                progress_width = min(75 * (row['寄与度（SHAP値）'] / top_factors['寄与度（SHAP値）'].max()), 75)
            else:
                # 負の場合（最小値でスケーリング）
                progress_width = min(75 * (row['寄与度（SHAP値）'] / top_factors['寄与度（SHAP値）'].min()), 75)
            
            factor_card = f"""
            <div class="{card_class}">
                <div class="factor-rank">{row['順位']}</div>
                <div class="factor-name">{row['要因名']}</div>
                <div class="factor-details">
                    <div class="factor-contribution">寄与度: {row['寄与度（SHAP値）']}</div>
                    <div class="progress-bar">
                        <div class="progress-fill" style="width: {progress_width}%"></div>
                    </div>
                </div>
                <div class="period">対象期間: {row['対象期間']}</div>
            </div>
            """
            
            st.markdown(factor_card, unsafe_allow_html=True)
        
        # インサイトコンテナの終了
        st.markdown('</div>', unsafe_allow_html=True)

        # 詳細データ表示
        st.markdown('<div class="sub-header">詳細データ</div>', unsafe_allow_html=True)
        
        # データフレームから動的にHTMLテーブルを生成
        html_table = """
        <div style="overflow-x: auto;">
        <table style="width:100%; border-collapse: collapse;">
            <thead>
                <tr>
                    <th style="text-align:center; padding:10px; border:1px solid #ddd;">順位</th>
                    <th style="text-align:left; padding:10px; border:1px solid #ddd;">要因名_詳細</th>
                    <th style="text-align:center; padding:10px; border:1px solid #ddd;">対象期間</th>
                    <th style="text-align:center; padding:10px; border:1px solid #ddd;">要因の値</th>
                    <th style="text-align:center; padding:10px; border:1px solid #ddd;">いつもの値</th>
                    <th style="text-align:center; padding:10px; border:1px solid #ddd;">寄与度</th>
                </tr>
            </thead>
            <tbody>
        """
    
        # データフレームの各行をループしてHTMLに追加
        for _, row in df.iterrows():
            html_table += f"""
                <tr>
                    <td style="text-align:center; padding:10px; border:1px solid #ddd;">{row['順位']}</td>
                    <td style="text-align:left; padding:10px; border:1px solid #ddd;">{row['要因名_詳細']}</td>
                    <td style="text-align:center; padding:10px; border:1px solid #ddd;">{row['対象期間']}</td>
                    <td style="text-align:center; padding:10px; border:1px solid #ddd;">{row['要因の値']}</td>
                    <td style="text-align:center; padding:10px; border:1px solid #ddd;">{row['いつもの値（ベースライン）']:.4f}</td>
                    <td style="text-align:center; padding:10px; border:1px solid #ddd; font-weight:bold; color:#1E40AF;">{row['寄与度（SHAP値）']:.4f}</td>
                </tr>
            """
        
        st.components.v1.html(html_table, height=600, scrolling=True)

    #! タブの作成
    tab1, tab2 = st.tabs(["在庫増加の要因", "在庫減少の要因"])

    #! ランキング表示
    with tab1:
    
        st.markdown('<div class="sub-header">在庫増加の主要因</div>', unsafe_allow_html=True)

        render_increse_top_factors(top_increase_ranking_df, increse_color)

    with tab2:
        
        st.markdown('<div class="sub-header">在庫減少の主要因</div>', unsafe_allow_html=True)

        render_increse_top_factors(top_decrease_ranking_df, decrese_color)

    
#MARK: 単独テスト用
if __name__ == "__main__":
    
    st.header("test")
    st.sidebar.header("test")

    # streamlitのバージョンが1.36.0だとサイズ変更適用される
    # 1.43.2だとサイズ変わらない
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

    start_datetime = pd.to_datetime('2024-08-01 10')
    end_datetime = pd.to_datetime('2024-08-10 10')
    selected_datetime = pd.to_datetime('2024-08-05 10')
    highlight_datetime = pd.to_datetime("2024-08-09 04:00")

    # CSVファイルのパスを指定
    file_path = '統合テーブル.csv'  

    # CSVファイルを読み込む
    merged_data_df = pd.read_csv(file_path)

    model, X, merged_data_df = train_model(merged_data_df)

    shap_values = compute_feature_importance( model, X)

    # 在庫推移と基準線を描画する
    show_zaiko_with_baseline( merged_data_df, start_datetime, end_datetime, highlight_datetime)

    # 特徴量重要度の結果を可視化する
    show_feature_importance( merged_data_df, X, selected_datetime, shap_values)

    #print(df)