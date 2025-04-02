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

# 自作ライブラリのimport
from preprocess_data import compute_features_and_target

#MARK: 機械学習のパイプライン
def pipeline():

    # 目的変数と説明変数を作成する
    df = compute_features_and_target()

    # 学習
    # 必要変数を返す
    # modelはSHAP値計算用、X、dfは結果描画用
    model, X, df = train_model(df)

    return model, X, df

#MARK: 機械学習モデルを適用する
def train_model(df):

    # todo 仮
    # # 複数列を同時に変更（辞書形式）
    # df = df.rename(columns={
    #     '在庫数（箱）': 'target_在庫数（箱）',
    #     'No12_発注かんばん数（t-0~t-0）': 'feature_No12_発注かんばん数（t-0~t-0）',
    #     'No13_西尾東物流センターor部品置き場での滞留かんばん数（t-0~t-0）': 'feature_No13_西尾東物流センターor部品置き場での滞留かんばん数（t-0~t-0）',
    #     'No18_在庫水準（t-29~t-197）':'feature_No18_在庫水準（t-29~t-197）',
    #     'No19_直近の生産数を表したもの（t-0~t-8）':'feature_No19_直近の生産数を表したもの（t-0~t-8）',
    # })

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

#MARK: 特徴量重要度（SHAP値）の計算
def compute_feature_importance( model, X):

    explainer = shap.TreeExplainer( model, model_output='raw')
    shap_values = explainer.shap_values(X)

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
        
    # 在庫増
    # マッピングを '変数名' 列に基づいて '要因名' 列に適用する関数を定義
    def map_increase_factor(variable):
        if "No1_" in variable:
            return "「必要な生産に対して発注かんばん数が多い」"
        elif "No2_" in variable:
            return "「計画組立生産台数が少ない」"
        elif "No3_" in variable:
            return "「組立ラインの稼働率が低い」"
        elif "No4_" in variable:
            return "「納入数が多い（挽回納入）」"
        elif "No5_" in variable:
            return "「仕入先便が早着している」"
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
        elif "No11_" in variable:
            return "「予定外の入庫がある」"
        elif "No12_" in variable:
            return """
                    <strong>【No1.発注不備】現在、入庫予定かんばん数が多い</strong><br>
                    ＜説明＞<br>
                    入庫予定かんばん数は「LINKSのデータ」と「仕入先ダイヤのデータ」をもとに以下で計算されています。<br>
                    ①西尾東を経由する部品：現在から5時間程度前の納入予定かんばん数<br>
                    ②直納の部品：現在から1時間程度前の納入予定かんばん数<br>
                    ※納入と入庫のリードタイムや稼働時間（稼働有無は自動ラックの入出庫で判断）を考慮して計算しています<br>
                    ＜考えられる事象＞<br>
                    No1-1：便Aveより納入かんばん数が多い<br>
                    No1-2：仕入先挽回納入
                    """
        elif "No13_" in variable:
            return "「None西尾東BC or 部品置き場で滞留しているかんばん数が少ない」"
        elif "No14_" in variable:
            return """
                    <strong>【No7.設計外の入庫】現在、設計外の入庫数が多い</strong><br>
                    ＜説明＞<br>
                    設計外の入庫とは、設計通りの入庫ではないものを表します。<br>
                    ＜考えられる事象＞<br>
                    No7-1.部品置き場などで滞留していた部品を入庫している
                    """
        elif "No15_" in variable:
            return "「None_間口OK」"
        elif "No16_" in variable:
            return "「仕入先便早く到着し普段より早い定期便で工場にモノが届いている」"
        elif "No17_" in variable:
            return "「過去の生産計画が多いため、外れかんばんが多く、発注かんばんが多い」"
        elif "No18_" in variable:
            return """
                    <strong>【No6.過去のかんばん要因】過去（1週間前程度）の発注かんばん数が多かった</strong><br>
                    ＜説明＞<br>
                    在庫推移は時系列で変動しているため、過去の在庫水準が現在の在庫数に寄与していると考えられます。<br>
                    過去の在庫水準を「LINKSのデータ」をもとに以下で計算しています。<br>
                    ・かんばん回転日数前から＋1週間の間の発注かんばん数-回収かんばん数<br>
                    ＜考えられる事象＞<br>
                    No6-1：生産に対して納入かんばんが多かった
                    """
        elif "No19_" in variable:
            return """
                    <strong>【No4. 組立要因】直近（現在～1日前まで）の生産数が少ない</strong><br>
                    ＜説明＞<br>
                    生産物流システムの着工数<br>
                    ＜考えられる異常＞<br>
                    No4-1：ライン停止<br>
                    No4-2：生産変動/計画変更/得意先の需要変化
                    """
        elif "No20_" in variable:
            return "None他品番の入庫が優先されている"
        elif "No21_" in variable:
            return """
                    <strong>【No2.回収不備】直近（かんばん回転日数前）の回収かんばん数が多い</strong><br>
                    ＜説明＞<br>
                    回収かんばん数が少ないor多いと、発注かんばん数が少ないor多くなる可能性があります。<br>
                    回収かんばん数は「LINKSデータ」をもとに計算しています。<br>
                    ＜考えられる事象＞<br>
                    No2-1：過去の生産が多かった<br>
                    No2-2：かんばん出し忘れを挽回回収した
                    """
        else:
            return None

    # 在庫減
    # マッピングを '変数名' 列に基づいて '要因名' 列に適用する関数を定義
    def map_decrease_factor(variable):
        if "No1_" in variable:
            return "「必要な生産に対して発注かんばんが少ない」"
        elif "No2_" in variable:
            return "「計画組立生産台数が多い」"
        elif "No3_" in variable:
            return "「組立ラインの稼働率が高い」"
        elif "No4_" in variable:
            return "「納入数が少ない（未納）」"
        elif "No5_" in variable:
            return "「仕入先便の遅着」"
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
        elif "No11_" in variable:
            return "「予定外の入庫がない」"
        elif "No12_" in variable:
            return """
                    <strong>【No1.発注不備】現在、入庫予定かんばん数が少ない</strong><br>
                    ＜説明＞<br>
                    入庫予定かんばん数は「LINKSのデータ」と「仕入先ダイヤのデータ」をもとに以下で計算されています。<br>
                    ①西尾東を経由する部品：現在から5時間程度前の納入予定かんばん数<br>
                    ②直納の部品：現在から1時間程度前の納入予定かんばん数<br>
                    ※納入と入庫のリードタイムや稼働時間（稼働有無は自動ラックの入出庫で判断）を考慮して計算しています<br>
                    ＜考えられる事象＞<br>
                    No.1-1：便Aveより納入かんばん数が少ない<br>
                    No.1-2：仕入先未納
                    """
        elif "No13_" in variable:
            return """
                    <strong>【No3.入庫遅れ】現在、西尾東BCから部品置き場の間で部品が滞留している</strong><br>
                    ＜説明＞<br>
                    設計時間を経過しても入庫されていないかんばん数<br>
                    ＜考えられる事象＞<br>
                    No3-1：順立装置の設備停止<br>
                    No3-2：順立前の部品置き場で部品が残っている<br>
                    No3-3：西尾東BCで誤転送<br>
                    No3-4：工場ビットの部品OF<br>
                    No3-5：西尾東で部品が残っている/定期便の乗り遅れ<br>
                    No3-6：台風積雪などによるトラックの遅延<br>
                    など
                    """
        elif "No14_" in variable:
            return """
                    <strong>【No7.設計外の入庫】現在、設計外の入庫数が少ない</strong><br>
                    ＜説明＞<br>
                    設計外の入庫とは、設計通りの入庫ではないものを表します。<br>
                    ＜考えられる事象＞<br>
                    No7-1：部品置き場などで滞留していた部品を入庫していない
                    """
        elif "No15_" in variable:
            return """
                    <strong>【No5. 他品番の在庫異常】現在、投入間口が一杯で入庫できない</strong><br>
                    ＜説明＞<br>
                    いずれかの間口が一杯で投入できない状態を表す<br>
                    ＜考えられる異常＞<br>
                    No5-1：偏った箱種の入庫<br>
                    No5-2：入庫数が多く間口のキャパ越え
                    """
        elif "No16_" in variable:
            return "「仕入先便が遅く到着し、普段より遅い定期便で工場にモノが届いている」"
        elif "No17_" in variable:
            return "「過去の生産計画が少ないため、外れかんばんが少なく、発注かんばんが少ない」"
        elif "No18_" in variable:
            return """
                    <strong>【No6.過去のかんばん要因】過去（1週間程度前）の発注かんばん数が少なかった</strong><br>
                    ＜説明＞<br>
                    在庫推移は時系列で変動しているため、過去の在庫水準が現在の在庫数に寄与していると考えられます。<br>
                    過去の在庫水準を「LINKSのデータ」をもとに以下で計算しています。<br>
                    ・かんばん回転日数前から＋1週間の間の発注かんばん数-回収かんばん数<br>
                    ＜考えられる事象＞<br>
                    No6-1：生産に対して納入かんばんが少なかった
                    """
        elif "No19_" in variable:
            return """
                    <strong>【No4. 組立要因】直近（現在～1日前まで）の生産数が多い</strong><br>
                    ＜説明＞<br>
                    生産物流システムの着工数<br>
                    ＜考えられる異常＞<br>
                    No4-1：生産変動/挽回生産<br>
                    No4-2：計画変更/得意先の需要変化
                    """
        elif "No20_" in variable:
            return "None"
        elif "No21_" in variable:
            return """
                    <strong>【No2.回収不備】過去（かんばん回転日数前）の回収かんばん数が少ない</strong><br>
                    ＜説明＞<br>
                    回収かんばん数が少ないor多いと、発注かんばん数が少ないor多くなる可能性があります。<br>
                    回収かんばん数は「LINKSデータ」をもとに計算しています。<br>
                    ＜考えられる事象＞<br>
                    No2-1：過去の生産が少ない<br>
                    No2-2：かんばんの出し忘れ<br>
                    No2-3：組立の取り忘れ
                    """
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

    # 要因の値（説明変数X）
    filtered_X = X.loc[filtered_merged_data_df.index[0]]# Pandas の .loc[]で1行のみを指定すると、seriesになる
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

    # 増と減それぞれで'要因名' 列を作成
    top_increase_ranking_df['要因名'] = top_increase_ranking_df['変数名'].apply(map_increase_factor)
    top_decrease_ranking_df['要因名'] = top_decrease_ranking_df['変数名'].apply(map_decrease_factor)

    # 順位、変数名、値だけを表示し、インデックスは消す
    top_increase_ranking_df = top_increase_ranking_df[['順位', '要因名', '対象期間', '要因の値', 'いつもの値（ベースライン）', '寄与度（SHAP値）']]
    top_decrease_ranking_df = top_decrease_ranking_df[['順位', '要因名', '対象期間', '要因の値', 'いつもの値（ベースライン）', '寄与度（SHAP値）']]
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
                    <th style="text-align:left; padding:10px; border:1px solid #ddd;">要因名</th>
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
                    <td style="text-align:left; padding:10px; border:1px solid #ddd;">{row['要因名']}</td>
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

    # streamlitのバージョンが1.36.0だと適用される
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