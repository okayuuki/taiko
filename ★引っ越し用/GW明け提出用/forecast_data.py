# ライブラリのimport
import streamlit as st
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import base64
from joblib import Parallel, delayed
import time

# 自作関数の読み込み
from get_data import compute_hourly_buhin_zaiko_data_by_hinban, \
    compute_hourly_specific_checkpoint_kanbansu_data_by_hinban, \
        compute_hourly_tehai_data_by_hinban, \
            get_kado_schedule_from_172_20_113_185, \
                get_hinban_master,\
                    get_hinban_info_detail,\
                        compute_hourly_buhin_zaiko_data_by_all_hinban, \
                            compute_hourly_specific_checkpoint_kanbansu_data_by_all_hinban

# 日本語フォントの設定
plt.rcParams['font.family'] = 'MS Gothic'  # Windows

# 各種設定用ファイル
CONFIG_PATH = '../../configs/settings.json'

# MARK:シミュレーションを行う共通関数
def run_simulation(
    zaiko_extracted,
    hourly_kanban_count_full,
    filtered_tehai_data,
    kado_df,
    start_datetime_for_calc,
    end_datetime_for_calc,
    start_datetime_for_show,
    column_name,
    mode,
    out_parameter,
    selected_zaiko_hako = None,
    selected_zaiko_buhin = None
):

    # シミュレーションを実行する関数
    def calculate_inventory_adjustments(hourly_kanban_count_full, 
                                  filtered_tehai_data, kado_df, start_datetime_for_calc, 
                                  start_datetime_for_show, end_datetime_for_calc, 
                                  current_inventory_hako_or_buhin, out_parameter, column_name,incoming_column,
                                  daily_consumption_column, max_daily_consumption_column,
                                  min_baseline_column,max_baseline_column,unit_type):

        """
        各時間後の消費予定数および入庫予定数を考慮した在庫数を計算する関数
        
        Parameters:
        -----------
        hourly_kanban_count_full : pandas.DataFrame
            IN情報のデータフレーム
        filtered_tehai_data : pandas.DataFrame
            OUT情報データ
        kado_df : pandas.DataFrame
            稼働フラグのデータフレーム
        start_datetime_for_calc : datetime
            計算開始日時
        start_datetime_for_show : datetime
            表示開始日時
        end_datetime_for_calc : datetime
            計算終了日時
        current_inventory_hako_or_buhin : float
            現在の在庫数（箱 or 部品）
        out_parameter : str
            出力パラメータ（"日量を採用する" or "日量MAXを採用する"）
        column_name : str
            列名
        incoming_column : str
            '工場到着予定かんばん数' or '工場到着予定かんばん数（部品数）'
        daily_consumption_column : str
            "日量数/稼働時間" or 日量数（箱数）/稼働時間
        max_daily_consumption_column : str
            "月末までの最大日量数/稼働時間" or "月末までの最大日量数（箱数）/稼働時間"
        min_baseline_column : str
            下限基準線の列名
        max_baseline_column : str
            上限基準線の列名
        unit_type : str
            "箱換算" or "部品換算"
            
        Returns:
        --------
        pandas.DataFrame
            結合された最終的なデータフレーム
        """

        # 計算単位に応じて在庫数の名前を変更
        if unit_type == "箱換算":
            zaiko_name = '在庫数（箱）'
        elif unit_type == "部品換算":
            zaiko_name = '在庫数（部品数）'

        # 各時間後の消費予定数および入庫予定数を考慮した予測在庫を保存するリストを初期化
        inventory_after_adjustments = []
        
        count = 0
        # 時間ごとの在庫数を更新しながらリストに追加
        for i, row in filtered_tehai_data.iterrows():
            kanban_row = hourly_kanban_count_full[hourly_kanban_count_full['日時'] == row['日時']]
            filtered_kado_df = kado_df[kado_df['日時'] == row['日時']]
            kado_row = filtered_kado_df['稼働フラグ'].values[0]
            incoming_kanban = kanban_row[incoming_column].values[0] if not kanban_row.empty else 0

            # 選択時間から在庫増減の計算を行う
            if row['日時'] >= start_datetime_for_calc:
                inventory_after_adjustments.append({
                    '日時': row['日時'],
                    zaiko_name: current_inventory_hako_or_buhin,
                    '設計値MIN': row[min_baseline_column],
                    '設計値MAX': row[max_baseline_column]
                })

                # 最初のタイムスタンプでは消費を引かないが、以降は消費量と入庫量を調整
                if count != 0:
                    # 稼働フラグが0でない場合のみ減算
                    if kado_row != 0:
                        if out_parameter == "日量を採用する":
                            current_inventory_hako_or_buhin = current_inventory_hako_or_buhin - row[daily_consumption_column]
                        elif out_parameter == "日量MAXを採用する":
                            current_inventory_hako_or_buhin = current_inventory_hako_or_buhin - row[max_daily_consumption_column]
                    current_inventory_hako_or_buhin = current_inventory_hako_or_buhin + incoming_kanban

                count = count + 1

        # 計算結果をDataFrameに変換
        inventory_df_adjusted = pd.DataFrame(inventory_after_adjustments)

        # 納入時間の範囲を調整
        start_datetime_for_nonyu = start_datetime_for_show.strftime('%Y-%m-%d %H:%M:%S')
        end_datetime_for_nonyu = end_datetime_for_calc.strftime('%Y-%m-%d %H:%M:%S')
        hourly_kanban_count_filtered = hourly_kanban_count_full[
            (hourly_kanban_count_full['日時'] >= start_datetime_for_nonyu) & 
            (hourly_kanban_count_full['日時'] <= end_datetime_for_nonyu)
        ]

        # データフレームの結合
        merged_df = pd.merge(hourly_kanban_count_filtered, inventory_df_adjusted, on='日時', how='outer')
        merged_df = pd.merge(merged_df, filtered_tehai_data, on='日時', how='outer')
        merged_df = pd.merge(merged_df, kado_df, on='日時', how='outer')

        # 列の順番を変更
        new_column_order = ['日時','稼働フラグ', column_name, incoming_column, 
                        daily_consumption_column, zaiko_name]
        merged_df = merged_df[new_column_order]

        # 実行結果の確認
        #st.dataframe(inventory_df_adjusted)
        #st.dataframe(merged_df)

        return inventory_df_adjusted, merged_df

    
    # 現在の在庫数を初期値として設定
    if mode == "在庫予測":
        current_inventory_hako = zaiko_extracted.iloc[0]['在庫数（箱）']
        current_inventory_buhin = zaiko_extracted.iloc[0]['現在在庫（台）']
    elif mode == "リミット計算":
        current_inventory_hako = selected_zaiko_hako
        current_inventory_buhin = selected_zaiko_buhin

    # 箱換算で計算する
    daily_consumption_column = '日量数（箱数）/稼働時間'
    max_daily_consumption_column = '月末までの最大日量数（箱数）/稼働時間'
    min_baseline_column = '設計値MIN'
    max_baseline_column = '設計値MAX'
    unit_type = '箱換算'
    incoming_column = '工場到着予定かんばん数'
    current_inventory_hako_or_buhin = current_inventory_hako
    inventory_df_adjusted_hako, merged_df_hako = calculate_inventory_adjustments(hourly_kanban_count_full, 
                                            filtered_tehai_data, kado_df, start_datetime_for_calc, 
                                            start_datetime_for_show, end_datetime_for_calc, 
                                            current_inventory_hako_or_buhin, out_parameter, column_name,incoming_column,
                                            daily_consumption_column, max_daily_consumption_column,
                                            min_baseline_column,max_baseline_column,unit_type)

    # st.write("箱結果の確認")
    # st.dataframe(inventory_df_adjusted_hako)
    # st.dataframe(merged_df_hako)
    
    # 部品換算で計算する
    filtered_tehai_data['設計値MIN（部品数）'] = filtered_tehai_data['設計値MIN']*filtered_tehai_data['収容数']
    filtered_tehai_data['設計値MAX（部品数）'] = filtered_tehai_data['設計値MAX']*filtered_tehai_data['収容数']
    daily_consumption_column = '日量数/稼働時間'
    max_daily_consumption_column = '月末までの最大日量数/稼働時間'
    min_baseline_column = '設計値MIN（部品数）'
    max_baseline_column = '設計値MAX（部品数）'
    unit_type = '部品換算'
    hourly_kanban_count_full = pd.merge(hourly_kanban_count_full, filtered_tehai_data[['日時','収容数']], on='日時', how='left')
    hourly_kanban_count_full['工場到着予定かんばん数（部品数）'] = hourly_kanban_count_full['工場到着予定かんばん数']*hourly_kanban_count_full['収容数']
    incoming_column = '工場到着予定かんばん数（部品数）'
    current_inventory_hako_or_buhin = current_inventory_buhin
    inventory_df_adjusted_buhin, merged_df_buhin = calculate_inventory_adjustments(hourly_kanban_count_full, 
                                            filtered_tehai_data, kado_df, start_datetime_for_calc, 
                                            start_datetime_for_show, end_datetime_for_calc, 
                                            current_inventory_hako_or_buhin, out_parameter, column_name,incoming_column,
                                            daily_consumption_column, max_daily_consumption_column,
                                            min_baseline_column,max_baseline_column,unit_type)

    
    # 箱＋部品換算で計算する
    syuyosu_value = filtered_tehai_data[filtered_tehai_data['日時'] == start_datetime_for_calc]['収容数'].iloc[0]
    current_inventory_hako_or_buhin = current_inventory_hako*syuyosu_value + current_inventory_buhin
    inventory_df_adjusted_hako_and_buhin, merged_df_hako_and_buhin = calculate_inventory_adjustments(hourly_kanban_count_full, 
                                            filtered_tehai_data, kado_df, start_datetime_for_calc, 
                                            start_datetime_for_show, end_datetime_for_calc, 
                                            current_inventory_hako_or_buhin, out_parameter, column_name,incoming_column,
                                            daily_consumption_column, max_daily_consumption_column,
                                            min_baseline_column,max_baseline_column,unit_type)

    
    
    # st.write("部品結果の確認")
    # st.dataframe(inventory_df_adjusted_buhin)
    # st.dataframe(merged_df_buhin)

    return inventory_df_adjusted_hako, merged_df_hako, inventory_df_adjusted_buhin, merged_df_buhin, inventory_df_adjusted_hako_and_buhin, merged_df_hako_and_buhin

# MARK: 【方法１】シミュレーションに必要な変数を準備し、シミュレーションを行う共通関数
#! 今は在庫リミット専用
def setup_and_run_simulation(
    hinban_info,
    kojo,
    flag_useDataBase,
    start_datetime_for_calc, # 開始日時（datetime オブジェクト）
    end_datetime_for_calc, # 終了日時（datetime オブジェクト）
    start_datetime_for_show,# 結果を見せる開始日時（datetime オブジェクト）
    target_column,
    mode,
    out_parameter,
    selected_zaiko_hako=None,
    selected_zaiko_buhin=None
):
    # # 在庫
    # # 文字型に戻す
    # #! 15分単位なので変換
    # start_datetime_for_zaiko = start_datetime_for_calc.replace(minute=0, second=0).strftime('%Y-%m-%d %H:%M:%S')
    # end_datetime_for_zaiko = start_datetime_for_calc.replace(minute=0, second=0).strftime('%Y-%m-%d %H:%M:%S')
    # # 指摘期間で読み込む
    # zaiko_df = compute_hourly_buhin_zaiko_data_by_hinban(hinban_info,
    #  start_datetime_for_zaiko, end_datetime_for_zaiko,flag_useDataBase, kojo)
    # # 必要な列のみ抽出
    # zaiko_extracted = zaiko_df[['日時', '在庫数（箱）','現在在庫（台）']]
    # #! 15分単位の時刻に修正
    # zaiko_extracted.loc[zaiko_extracted.index[0], '日時'] = start_datetime_for_calc
    # # 実行結果の確認
    # #st.dataframe(zaiko_extracted)

    # # IN準備
    # #todo 時間遅れあるから前の時間を開始とする
    # #todo 更新日時で取っているから幅を見る必要ある
    # #! 近い時刻で更新されたものは後ろの時刻で更新されるから前後両方で見る必要がある
    # start_datetime_for_input = start_datetime_for_calc - timedelta(hours=24*10)
    # end_datetime_for_input = end_datetime_for_calc + timedelta(hours=24*10)
    # # 文字型に戻す
    # start_datetime_for_input = start_datetime_for_input.strftime('%Y-%m-%d %H:%M:%S')
    # end_datetime_for_input = end_datetime_for_input.strftime('%Y-%m-%d %H:%M:%S')
    # # 指定期間で読み込む
    # time_granularity = '15min'
    # _ , hourly_kanban_count = compute_hourly_specific_checkpoint_kanbansu_data_by_hinban(hinban_info, target_column,
    #  start_datetime_for_input, end_datetime_for_input, time_granularity, flag_useDataBase, kojo)
    # # 実行結果の確認
    # #st.dataframe(hourly_kanban_count)

    # # OUT
    # # 文字型に戻す
    # start_datetime_for_output = start_datetime_for_show.strftime('%Y-%m-%d %H:%M:%S')
    # end_datetime_for_output = end_datetime_for_calc.strftime('%Y-%m-%d %H:%M:%S')
    # #　指定期間で読み込む
    # time_granularity = '15min'
    # tehai_data = compute_hourly_tehai_data_by_hinban(hinban_info, start_datetime_for_output, end_datetime_for_output, time_granularity,
    #  flag_useDataBase, kojo)
    # tehai_data['日量数（箱数）/稼働時間'] = tehai_data['日量数（箱数）'] / (16.5*4)
    # tehai_data['月末までの最大日量数（箱数）/稼働時間'] = tehai_data['月末までの最大日量数（箱数）'] / (16.5*4)
    # tehai_data['日量数/稼働時間'] = tehai_data['日量数'] / (16.5*4)
    # tehai_data['月末までの最大日量数/稼働時間'] = tehai_data['月末までの最大日量数'] / (16.5*4)
    # # '日付' と '日量数（箱数）' の列のみを抽出
    # filtered_tehai_data = tehai_data[['日時','収容数','日量数（箱数）/稼働時間','日量数/稼働時間','月末までの最大日量数/稼働時間','月末までの最大日量数（箱数）/稼働時間','納入LT(H)','設計値MIN','設計値MAX']]
    # # 実行結果の確認
    # #st.dataframe(filtered_tehai_data)

    # # 稼働フラグ
    # # 稼働フラグデータの読み込み
    # start_datetime_for_kado = start_datetime_for_show.strftime('%Y-%m-%d %H:%M:%S')
    # end_datetime_for_kado = end_datetime_for_calc.strftime('%Y-%m-%d %H:%M:%S')
    # kado_df = get_kado_schedule_from_172_20_113_185(start_datetime_for_kado, end_datetime_for_kado, day_col='計画(昼)', night_col='計画(夜)',time_granularity='15min')
    # #st.dataframe(kado_df)

    #　並列処理（マルチプロセス）
    def run_parallel_processing(process_number, hinban_info, target_column,  start_datetime_for_calc, end_datetime_for_calc, flag_useDataBase, kojo):

        if process_number == 0:

            # 在庫
            # 文字型に戻す
            #! 15分単位なので変換
            start_datetime_for_zaiko = start_datetime_for_calc.replace(minute=0, second=0).strftime('%Y-%m-%d %H:%M:%S')
            end_datetime_for_zaiko = start_datetime_for_calc.replace(minute=0, second=0).strftime('%Y-%m-%d %H:%M:%S')
            # 指摘期間で読み込む
            zaiko_df = compute_hourly_buhin_zaiko_data_by_hinban(hinban_info,
            start_datetime_for_zaiko, end_datetime_for_zaiko,flag_useDataBase, kojo)
            # 必要な列のみ抽出
            zaiko_extracted = zaiko_df[['日時', '在庫数（箱）','現在在庫（台）']]
            #! 15分単位の時刻に修正
            zaiko_extracted.loc[zaiko_extracted.index[0], '日時'] = start_datetime_for_calc
            # 実行結果の確認
            #st.dataframe(zaiko_extracted)

            # IN準備
            #todo 時間遅れあるから前の時間を開始とする
            #todo 更新日時で取っているから幅を見る必要ある
            #! 近い時刻で更新されたものは後ろの時刻で更新されるから前後両方で見る必要がある
            start_datetime_for_input = start_datetime_for_calc - timedelta(hours=24*10)
            end_datetime_for_input = end_datetime_for_calc + timedelta(hours=24*10)
            # 文字型に戻す
            start_datetime_for_input = start_datetime_for_input.strftime('%Y-%m-%d %H:%M:%S')
            end_datetime_for_input = end_datetime_for_input.strftime('%Y-%m-%d %H:%M:%S')
            # 指定期間で読み込む
            time_granularity = '15min'
            _ , hourly_kanban_count = compute_hourly_specific_checkpoint_kanbansu_data_by_hinban(hinban_info, target_column,
            start_datetime_for_input, end_datetime_for_input, time_granularity, flag_useDataBase, kojo)
            # 実行結果の確認
            #st.dataframe(hourly_kanban_count)

            return (zaiko_extracted, hourly_kanban_count)

        elif process_number == 1:

            # OUT
            # 文字型に戻す
            start_datetime_for_output = start_datetime_for_show.strftime('%Y-%m-%d %H:%M:%S')
            end_datetime_for_output = end_datetime_for_calc.strftime('%Y-%m-%d %H:%M:%S')
            #　指定期間で読み込む
            time_granularity = '15min'
            tehai_data = compute_hourly_tehai_data_by_hinban(hinban_info, start_datetime_for_output, end_datetime_for_output, time_granularity,
            flag_useDataBase, kojo)
            #st.dataframe(tehai_data)
            tehai_data['日量数（箱数）/稼働時間'] = tehai_data['日量数（箱数）'] / (16.5*4)
            tehai_data['月末までの最大日量数（箱数）/稼働時間'] = tehai_data['月末までの最大日量数（箱数）'] / (16.5*4)
            tehai_data['日量数/稼働時間'] = tehai_data['日量数'] / (16.5*4)
            tehai_data['月末までの最大日量数/稼働時間'] = tehai_data['月末までの最大日量数'] / (16.5*4)
            # '日付' と '日量数（箱数）' の列のみを抽出
            filtered_tehai_data = tehai_data[['日時','収容数','日量数（箱数）/稼働時間','日量数/稼働時間','月末までの最大日量数/稼働時間','月末までの最大日量数（箱数）/稼働時間','納入LT(H)','設計値MIN','設計値MAX']]
            # 実行結果の確認
            #st.dataframe(filtered_tehai_data)

            return filtered_tehai_data

        elif process_number == 2: 

            # 稼働フラグ
            # 稼働フラグデータの読み込み
            start_datetime_for_kado = start_datetime_for_show.strftime('%Y-%m-%d %H:%M:%S')
            end_datetime_for_kado = end_datetime_for_calc.strftime('%Y-%m-%d %H:%M:%S')
            kado_df = get_kado_schedule_from_172_20_113_185(start_datetime_for_kado, end_datetime_for_kado, day_col='計画(昼)', night_col='計画(夜)',time_granularity='15min')
            #st.dataframe(kado_df)

            return kado_df

    n_jobs = 3

    # # 並列処理の実行
    # results_parallel = Parallel(n_jobs=n_jobs)(
    #     delayed(run_parallel_processing)(process_number, hinban_info, target_column,  start_datetime_for_calc, end_datetime_for_calc, flag_useDataBase, kojo) for process_number in range(3)
    # )

    # 処理時間テスト
    # 並列処理テスト
    # start_time_parallel = time.time()
    # results_parallel = Parallel(n_jobs=n_jobs)(
    #     delayed(run_parallel_processing)(process_number, hinban_info, target_column, start_datetime_for_calc, end_datetime_for_calc, flag_useDataBase, kojo) for process_number in range(3)
    # )
    # parallel_time = time.time() - start_time_parallel

    # 逐次処理テスト
    start_time_sequential = time.time()
    results_sequential = [
        run_parallel_processing(i, hinban_info, target_column, start_datetime_for_calc, end_datetime_for_calc, flag_useDataBase, kojo) for i in range(3)
    ]
    sequential_time = time.time() - start_time_sequential
    results_parallel = results_sequential

    #print(f"並列処理時間: {parallel_time:.2f}秒")
    #print(f"逐次処理時間: {sequential_time:.2f}秒")
    #print(f"速度向上率: {sequential_time/parallel_time:.2f}倍")

    zaiko_extracted, hourly_kanban_count = results_parallel[0]
    filtered_tehai_data = results_parallel[1]
    kado_df = results_parallel[2]

    # IN
    # パラメータ設定
    past_hours = int(filtered_tehai_data['納入LT(H)'].unique()[0])
    # 工場到着予定かんばん数の計算
    column_name = target_column + "のかんばん数"
    # ○時間前のかんばん数を追加する
    hourly_kanban_count_full = hourly_kanban_count.copy()
    if past_hours != 0:
        past_hours = past_hours*4#! リミット計算は15分単位のため
        hourly_kanban_count_full['工場到着予定かんばん数'] = hourly_kanban_count[column_name].shift(past_hours)
    else:
        hourly_kanban_count_full['工場到着予定かんばん数'] = hourly_kanban_count[column_name]
    # 欠損値（最初のリードタイムの時間分）を0で埋める
    hourly_kanban_count_full['工場到着予定かんばん数'] = hourly_kanban_count_full['工場到着予定かんばん数'].fillna(0).astype(int)
    
    # 在庫予測
    inventory_df_adjusted_hako, merged_df_hako, inventory_df_adjusted_buhin, merged_df_buhin, inventory_df_adjusted_hako_and_buhin, merged_df_hako_and_buhin = run_simulation(
        zaiko_extracted,hourly_kanban_count_full,filtered_tehai_data,kado_df,
        start_datetime_for_calc,end_datetime_for_calc,start_datetime_for_show,
        column_name,mode,out_parameter,
        selected_zaiko_hako,selected_zaiko_buhin)
    
    return inventory_df_adjusted_hako, merged_df_hako, inventory_df_adjusted_buhin, merged_df_buhin, inventory_df_adjusted_hako_and_buhin, merged_df_hako_and_buhin
    
# MARK: 【方法２】シミュレーションに必要な変数を準備し、シミュレーションを行う共通関数（全品番一括でダウンロードする）
# todo 何度も読み込むと負荷が大きくなるかもしれないので
#! 今は在庫予測専用で使用
@st.cache_data
def setup_and_run_simulation_fast(
    hinban_info,
    kojo,
    flag_useDataBase,
    start_datetime_for_calc, # 開始日時（datetime オブジェクト）
    end_datetime_for_calc, # 終了日時（datetime オブジェクト）
    start_datetime_for_show,# 結果を見せる開始日時（datetime オブジェクト）
    target_column,
    mode,
    out_parameter,
    selected_zaiko_hako = None,
    selected_zaiko_buhin = None
):

    #　並列処理（マルチプロセス）
    def run_parallel_processing(process_number, hinban_info, target_column,  start_datetime_for_calc, end_datetime_for_calc, flag_useDataBase, kojo):

        # 品番情報設定
        hinban = hinban_info[0]
        seibishitsu = hinban_info[1]

        if process_number == 0:

            # 在庫（全品番）
            start_datetime_for_zaiko = start_datetime_for_calc.strftime('%Y-%m-%d %H:%M:%S')
            end_datetime_for_zaiko = start_datetime_for_calc.strftime('%Y-%m-%d %H:%M:%S')
            # 指定期間で読み込む
            zaiko_all_df = compute_hourly_buhin_zaiko_data_by_all_hinban(start_datetime_for_zaiko, end_datetime_for_zaiko, flag_useDataBase, kojo)
            #st.dataframe(zaiko_all_df)
            # 品番抽出
            zaiko_df = zaiko_all_df[(zaiko_all_df['品番'] == hinban) & (zaiko_all_df['整備室コード'] == seibishitsu)]
            #st.dataframe(zaiko_df)
            # 必要な列のみ抽出
            zaiko_extracted = zaiko_df[['日時', '在庫数（箱）','現在在庫（台）']]

            # IN準備（全品番）
            #todo 時間遅れあるから前の時間を開始とする
            #todo 更新日時で取っているから幅を見る必要ある
            #! 近い時刻で更新されたものは後ろの時刻で更新されるから前後両方で見る必要がある
            start_datetime_for_input = start_datetime_for_calc - timedelta(hours=24*10)
            end_datetime_for_input = end_datetime_for_calc + timedelta(hours=24*10)
            # 文字型に戻す
            start_datetime_for_input = start_datetime_for_input.strftime('%Y-%m-%d %H:%M:%S')
            end_datetime_for_input = end_datetime_for_input.strftime('%Y-%m-%d %H:%M:%S')
            # 指定期間で読み込む
            _, df_full = compute_hourly_specific_checkpoint_kanbansu_data_by_all_hinban(target_column, start_datetime_for_input, end_datetime_for_input, flag_useDataBase, kojo)
            #st.dataframe(df_full)
            # 品番抽出
            hourly_kanban_count = df_full[(df_full['品番'] == hinban) & (df_full['整備室コード'] == seibishitsu)]
            #st.dataframe(hourly_kanban_count)
            # 全ての列のNoneを0に置換
            hourly_kanban_count = hourly_kanban_count.fillna(0)
            #st.dataframe(hourly_kanban_count)

            return (zaiko_extracted, hourly_kanban_count)

        elif process_number == 1:

            # OUT
            # 文字型に戻す
            start_datetime_for_output = start_datetime_for_show.strftime('%Y-%m-%d %H:%M:%S')
            end_datetime_for_output = end_datetime_for_calc.strftime('%Y-%m-%d %H:%M:%S')
            #　指定期間で読み込む
            time_granularity = 'h'
            tehai_data = compute_hourly_tehai_data_by_hinban(hinban_info, start_datetime_for_output, end_datetime_for_output, time_granularity,
            flag_useDataBase, kojo)
            tehai_data['日量数（箱数）/稼働時間'] = tehai_data['日量数（箱数）'] / 16.5
            tehai_data['月末までの最大日量数（箱数）/稼働時間'] = tehai_data['月末までの最大日量数（箱数）'] / 16.5
            tehai_data['日量数/稼働時間'] = tehai_data['日量数'] / 16.5
            tehai_data['月末までの最大日量数/稼働時間'] = tehai_data['月末までの最大日量数'] / 16.5
            # '日付' と '日量数（箱数）' の列のみを抽出
            filtered_tehai_data = tehai_data[['日時','収容数','日量数（箱数）/稼働時間','日量数/稼働時間','月末までの最大日量数/稼働時間','月末までの最大日量数（箱数）/稼働時間','納入LT(H)','設計値MIN','設計値MAX']]
            #st.dataframe(filtered_tehai_data)

            return filtered_tehai_data

        elif process_number == 2:

            # 稼働フラグ
            # 稼働フラグデータの読み込み
            start_datetime_for_kado = start_datetime_for_show.strftime('%Y-%m-%d %H:%M:%S')
            end_datetime_for_kado = end_datetime_for_calc.strftime('%Y-%m-%d %H:%M:%S')
            kado_df = get_kado_schedule_from_172_20_113_185(start_datetime_for_kado, end_datetime_for_kado, day_col='計画(昼)', night_col='計画(夜)', time_granularity='h')
            #st.dataframe(kado_df)

            return kado_df

    n_jobs = 3

    # 並列処理の実行
    # results_parallel = Parallel(n_jobs=n_jobs)(
    #     delayed(run_parallel_processing)(process_number, hinban_info, target_column,  start_datetime_for_calc, end_datetime_for_calc, flag_useDataBase, kojo) for process_number in range(3)
    # )

    # 逐次処理テスト
    start_time_sequential = time.time()
    results_sequential = [
        run_parallel_processing(process_number, hinban_info, target_column,  start_datetime_for_calc, end_datetime_for_calc, flag_useDataBase, kojo) for process_number in range(3)
    ]
    sequential_time = time.time() - start_time_sequential
    results_parallel = results_sequential

    zaiko_extracted, hourly_kanban_count = results_parallel[0]
    filtered_tehai_data = results_parallel[1]
    kado_df = results_parallel[2]
    
    # IN
    # パラメータ設定
    past_hours = int(filtered_tehai_data['納入LT(H)'].unique()[0])
    # 工場到着予定かんばん数の計算
    column_name = target_column + "のかんばん数"
    # ○時間前のかんばん数を追加する
    hourly_kanban_count_full = hourly_kanban_count.copy()
    hourly_kanban_count_full['工場到着予定かんばん数'] = hourly_kanban_count[column_name].shift(past_hours)
    # 欠損値（最初のリードタイムの時間分）を0で埋める
    hourly_kanban_count_full['工場到着予定かんばん数'] = hourly_kanban_count_full['工場到着予定かんばん数'].fillna(0).astype(int)
    #
    if hourly_kanban_count_full is None or hourly_kanban_count_full.empty: 
        # 日時範囲の作成（1時間間隔）
        date_range = pd.date_range(start=start_datetime_for_show, end=end_datetime_for_calc, freq='h')
        # データフレーム作成
        hourly_kanban_count_full = pd.DataFrame({
            '日時': date_range,
            column_name: 0,  # デフォルトで1を設定
            '工場到着予定かんばん数':0
        })
    
    # 在庫予測
    inventory_df_adjusted_hako, merged_df_hako, inventory_df_adjusted_buhin, merged_df_buhin, inventory_df_adjusted_hako_and_buhin, merged_df_hako_and_buhin = run_simulation(
        zaiko_extracted,hourly_kanban_count_full,filtered_tehai_data,kado_df,
        start_datetime_for_calc,end_datetime_for_calc,start_datetime_for_show,
        column_name,mode,out_parameter,
        selected_zaiko_hako,selected_zaiko_buhin)

    #st.dataframe(inventory_df_adjusted_hako)

    return inventory_df_adjusted_hako, merged_df_hako

# MARK: 在庫リミット計算（バックエンド）
def compute_zaiko_limit(hinban_info, target_column, start_datetime, selected_zaiko_hako, selected_zaiko_buhin, out_parameter, kojo, flag_useDataBase):

    # 時間設定
    start_datetime = datetime.strptime(start_datetime, '%Y-%m-%d %H:%M:%S')
    start_datetime_for_calc = start_datetime
    end_datetime_for_calc = start_datetime + timedelta(hours=24)
    start_datetime_for_show = start_datetime - timedelta(hours=6)

    # 在庫シミュレーション関係
    calc_mode = "リミット計算"
    inventory_df_adjusted_hako, merged_df_hako, inventory_df_adjusted_buhin, merged_df_buhin, inventory_df_adjusted_hako_and_buhin, merged_df_hako_and_buhin = setup_and_run_simulation(
        hinban_info,
        kojo,
        flag_useDataBase,
        start_datetime_for_calc, # 開始日時（datetime オブジェクト）
        end_datetime_for_calc, # 終了日時（datetime オブジェクト）
        start_datetime_for_show,# 結果を見せる開始日時（datetime オブジェクト）
        target_column,
        calc_mode,
        out_parameter,
        selected_zaiko_hako = selected_zaiko_hako,
        selected_zaiko_buhin = selected_zaiko_buhin
    )

    #-------------------------------------------------------------ここから描画（将来的には分割したい）

    # タブの作成
    tab1, tab2, tab3 = st.tabs(["箱換算", "部品換算", "箱＋部品換算"])

    #st.dataframe(inventory_df_adjusted_hako)
    #st.dataframe(inventory_df_adjusted_buhin)

    with tab1:

        inventory_df_adjusted = inventory_df_adjusted_hako
        merged_df = merged_df_hako

        # 最初の時間のデータ（実際のデータ）とそれ以降の予測データに分割
        actual_data = inventory_df_adjusted.iloc[0:1]  # 最初の1時間分は実際のデータ
        forecast_data = inventory_df_adjusted.iloc[1:]  # それ以降は予測データ

        # グラフの作成
        fig = go.Figure()

        # 実際のデータを青色で描画
        fig.add_trace(go.Bar(
            x=actual_data['日時'], 
            y=actual_data['在庫数（箱）'], 
            name='実績', 
            marker_color='blue', 
            opacity=0.3
        ))

        # 予測データをオレンジ色で追加描画
        fig.add_trace(go.Bar(
            x=forecast_data['日時'], 
            y=forecast_data['在庫数（箱）'], 
            name='予測', 
            marker_color='orange', 
            opacity=0.3
        ))

        # x軸を1時間ごとに表示する設定
        fig.update_layout(
            title='予測結果',  # ここでタイトルを設定
            xaxis_title='日時',  # x軸タイトル
            yaxis_title='在庫数（箱）',  # y軸タイトル
            xaxis=dict(
                tickformat="%Y-%m-%d %H:%M",  # 日時のフォーマットを指定
                dtick=3600000  # 1時間ごとに表示 (3600000ミリ秒 = 1時間)
            ),
            barmode='group'  # 複数のバーをグループ化
        )

        # 設計値MINを破線で追加描画
        fig.add_trace(go.Scatter(
            x=inventory_df_adjusted['日時'],  # または forecast_data['日時'] を使用
            y=inventory_df_adjusted['設計値MIN'],  # 設計値MINの値を繰り返し
            name='設計値MIN',
            line=dict(
                color='orange',
                dash='dash'  # 破線スタイルを指定
            )
        ))

        # 設計値MINを破線で追加描画
        fig.add_trace(go.Scatter(
            x=inventory_df_adjusted['日時'],  # または forecast_data['日時'] を使用
            y=inventory_df_adjusted['設計値MAX'],  # 設計値MINの値を繰り返し
            name='設計値MAX',
            line=dict(
                color='green',
                dash='dash'  # 破線スタイルを指定
            )
        ))

        # グラフをStreamlitで表示
        st.plotly_chart(fig)

        # 新しい列「備考」を追加し、start_timeに基づいて「過去」「未来」と表示
        merged_df['※注釈                                                                               '] = merged_df['日時'].apply(
            lambda x: 'あなたはこの時間を選択しました' if x == start_datetime else ('過去' if x < start_datetime else '未来')
        )

        # '日時'列でstart_timeに一致する行をハイライト
        def highlight_start_time(row):
            return ['background-color: yellow' if row['日時'] == start_datetime else '' for _ in row]
        
        st.code(f"📝 計算式：未来の在庫数 = 在庫数 + 工場到着予定かんばん数 - 日量箱数/稼働時間")

        # 注釈を追加（例としてstart_timeを表示）
        st.markdown(f"")
        st.markdown(f"")
        st.markdown(f"**下の表で予測の内容を確認できます。**")

        # 条件に該当する（過去の在庫数）行の在庫数を "-" にする
        merged_df.loc[
            (merged_df['日時'] >= start_datetime_for_show) & 
            (merged_df['日時'] < start_datetime), 
            '在庫数（箱）'
        ] = "-"

        # '日時'列でstart_timeに一致する行をハイライトして表示
        st.dataframe(merged_df.style.apply(highlight_start_time, axis=1))

    with tab2:

        inventory_df_adjusted = inventory_df_adjusted_buhin
        merged_df = merged_df_buhin

        # 最初の時間のデータ（実際のデータ）とそれ以降の予測データに分割
        actual_data = inventory_df_adjusted.iloc[0:1]  # 最初の1時間分は実際のデータ
        forecast_data = inventory_df_adjusted.iloc[1:]  # それ以降は予測データ

        # グラフの作成
        fig = go.Figure()

        # 実際のデータを青色で描画
        fig.add_trace(go.Bar(
            x=actual_data['日時'], 
            y=actual_data['在庫数（部品数）'], 
            name='実績', 
            marker_color='blue', 
            opacity=0.3
        ))

        # 予測データをオレンジ色で追加描画
        fig.add_trace(go.Bar(
            x=forecast_data['日時'], 
            y=forecast_data['在庫数（部品数）'], 
            name='予測', 
            marker_color='orange', 
            opacity=0.3
        ))

        # x軸を1時間ごとに表示する設定
        fig.update_layout(
            title='予測結果',  # ここでタイトルを設定
            xaxis_title='日時',  # x軸タイトル
            yaxis_title='在庫数（部品数）',  # y軸タイトル
            xaxis=dict(
                tickformat="%Y-%m-%d %H:%M",  # 日時のフォーマットを指定
                dtick=3600000  # 1時間ごとに表示 (3600000ミリ秒 = 1時間)
            ),
            barmode='group'  # 複数のバーをグループ化
        )

        # 設計値MINを破線で追加描画
        fig.add_trace(go.Scatter(
            x=inventory_df_adjusted['日時'],  # または forecast_data['日時'] を使用
            y=inventory_df_adjusted['設計値MIN'],  # 設計値MINの値を繰り返し
            name='設計値MIN',
            line=dict(
                color='orange',
                dash='dash'  # 破線スタイルを指定
            )
        ))

        # 設計値MINを破線で追加描画
        fig.add_trace(go.Scatter(
            x=inventory_df_adjusted['日時'],  # または forecast_data['日時'] を使用
            y=inventory_df_adjusted['設計値MAX'],  # 設計値MINの値を繰り返し
            name='設計値MAX',
            line=dict(
                color='green',
                dash='dash'  # 破線スタイルを指定
            )
        ))

        # グラフをStreamlitで表示
        st.plotly_chart(fig)

        # 新しい列「備考」を追加し、start_timeに基づいて「過去」「未来」と表示
        merged_df['※注釈                                                                               '] = merged_df['日時'].apply(
            lambda x: 'あなたはこの時間を選択しました' if x == start_datetime else ('過去' if x < start_datetime else '未来')
        )

        # '日時'列でstart_timeに一致する行をハイライト
        def highlight_start_time(row):
            return ['background-color: yellow' if row['日時'] == start_datetime else '' for _ in row]
        
        st.code(f"📝 計算式：未来の在庫数 = 在庫数 + 工場到着予定かんばん数 - 日量箱数/稼働時間")

        # 注釈を追加（例としてstart_timeを表示）
        st.markdown(f"")
        st.markdown(f"")
        st.markdown(f"**下の表で予測の内容を確認できます。**")

        # 条件に該当する（過去の在庫数）行の在庫数を "-" にする
        merged_df.loc[
            (merged_df['日時'] >= start_datetime_for_show) & 
            (merged_df['日時'] < start_datetime), 
            '在庫数（部品数）'
        ] = "-"

        # '日時'列でstart_timeに一致する行をハイライトして表示
        st.dataframe(merged_df.style.apply(highlight_start_time, axis=1))
    
    with tab3:
        
        inventory_df_adjusted = inventory_df_adjusted_hako_and_buhin
        merged_df = merged_df_hako_and_buhin

        # 最初の時間のデータ（実際のデータ）とそれ以降の予測データに分割
        actual_data = inventory_df_adjusted.iloc[0:1]  # 最初の1時間分は実際のデータ
        forecast_data = inventory_df_adjusted.iloc[1:]  # それ以降は予測データ

        # グラフの作成
        fig = go.Figure()

        # 実際のデータを青色で描画
        fig.add_trace(go.Bar(
            x=actual_data['日時'], 
            y=actual_data['在庫数（部品数）'], 
            name='実績', 
            marker_color='blue', 
            opacity=0.3
        ))

        # 予測データをオレンジ色で追加描画
        fig.add_trace(go.Bar(
            x=forecast_data['日時'], 
            y=forecast_data['在庫数（部品数）'], 
            name='予測', 
            marker_color='orange', 
            opacity=0.3
        ))

        # x軸を1時間ごとに表示する設定
        fig.update_layout(
            title='予測結果',  # ここでタイトルを設定
            xaxis_title='日時',  # x軸タイトル
            yaxis_title='在庫数（部品数）',  # y軸タイトル
            xaxis=dict(
                tickformat="%Y-%m-%d %H:%M",  # 日時のフォーマットを指定
                dtick=3600000  # 1時間ごとに表示 (3600000ミリ秒 = 1時間)
            ),
            barmode='group'  # 複数のバーをグループ化
        )

        # 設計値MINを破線で追加描画
        fig.add_trace(go.Scatter(
            x=inventory_df_adjusted['日時'],  # または forecast_data['日時'] を使用
            y=inventory_df_adjusted['設計値MIN'],  # 設計値MINの値を繰り返し
            name='設計値MIN',
            line=dict(
                color='orange',
                dash='dash'  # 破線スタイルを指定
            )
        ))

        # 設計値MINを破線で追加描画
        fig.add_trace(go.Scatter(
            x=inventory_df_adjusted['日時'],  # または forecast_data['日時'] を使用
            y=inventory_df_adjusted['設計値MAX'],  # 設計値MINの値を繰り返し
            name='設計値MAX',
            line=dict(
                color='green',
                dash='dash'  # 破線スタイルを指定
            )
        ))

        # グラフをStreamlitで表示
        st.plotly_chart(fig)

        # 新しい列「備考」を追加し、start_timeに基づいて「過去」「未来」と表示
        merged_df['※注釈                                                                               '] = merged_df['日時'].apply(
            lambda x: 'あなたはこの時間を選択しました' if x == start_datetime else ('過去' if x < start_datetime else '未来')
        )

        # '日時'列でstart_timeに一致する行をハイライト
        def highlight_start_time(row):
            return ['background-color: yellow' if row['日時'] == start_datetime else '' for _ in row]
        
        st.code(f"📝 計算式：未来の在庫数 = 在庫数 + 工場到着予定かんばん数 - 日量箱数/稼働時間")

        # 注釈を追加（例としてstart_timeを表示）
        st.markdown(f"")
        st.markdown(f"")
        st.markdown(f"**下の表で予測の内容を確認できます。**")

        # 条件に該当する（過去の在庫数）行の在庫数を "-" にする
        merged_df.loc[
            (merged_df['日時'] >= start_datetime_for_show) & 
            (merged_df['日時'] < start_datetime), 
            '在庫数（部品数）'
        ] = "-"

        # '日時'列でstart_timeに一致する行をハイライトして表示
        st.dataframe(merged_df.style.apply(highlight_start_time, axis=1))

    return 0

# MARK：在庫リミット計算表示（フロント）
def show_results_of_zaiko_limit(hinban_info, target_column, start_datetime, selected_zaiko_hako, selected_zaiko_buhin, out_parameter, kojo, flag_useDataBase):
    #st.write(kojo)
    start_time = time.time()
    # 品番情報
    #! 15分単位を0分に直して
    flag_display = 1
    start_datetime = pd.to_datetime(start_datetime).replace(minute=0, second=0).strftime('%Y-%m-%d %H:%M:%S')
    get_hinban_info_detail(hinban_info, start_datetime, flag_display,flag_useDataBase, kojo)
    compute_zaiko_limit(hinban_info, target_column, start_datetime, selected_zaiko_hako, selected_zaiko_buhin, out_parameter, kojo, flag_useDataBase)
    resultstime = time.time() - start_time
    print(resultstime)

# MARK: 在庫予測（バックエンド）
def compute_future_zaiko(target_column, start_datetime, run_mode, out_parameter, kojo, flag_useDataBase):

    # 時間設定
    start_datetime = datetime.strptime(start_datetime, '%Y-%m-%d %H:%M:%S')
    start_datetime_for_calc = start_datetime
    end_datetime_for_calc = start_datetime + timedelta(hours=24)
    start_datetime_for_show = start_datetime - timedelta(hours=6)

    #　ユニークな品番_整備室の組み合わせを抽出
    # todo 半年等で見ると600品番ぐらいある
    # todo 納入便がNULLになりエラーでおちる
    unique_hinbans = get_hinban_master()[:5]
    #st.write(unique_hinbans)

    # 空のリストを作成
    hinban_list = []
    data_list = []
    hinban_info = ["", ""]

    # ユニークな品番の組み合わせの数だけ処理を行う
    for unique_hinban in unique_hinbans:

        # 最初の _ で 2 つに分割
        hinban_info[0], hinban_info[1] = unique_hinban.split("_", 1)
        #st.write(hinban_info[0], hinban_info[1])

        try:
            # 在庫シミュレーション関係
            #todo 仕入先ダイヤと紐づかずエラー出る
            mode = "在庫予測"
            zaiko_actuals_and_forecast_df, merged_df = setup_and_run_simulation_fast(
                hinban_info,
                kojo,
                flag_useDataBase,
                start_datetime_for_calc, # 開始日時（datetime オブジェクト）
                end_datetime_for_calc, # 終了日時（datetime オブジェクト）
                start_datetime_for_show,# 結果を見せる開始日時（datetime オブジェクト）
                target_column,
                mode,
                out_parameter,
                selected_zaiko_hako=None,
                selected_zaiko_buhin=None
            )
            #st.dataframe(zaiko_actuals_and_forecast_df)
            #st.dataframe(merged_df)
        except Exception as e:
            # logger.error(f"Error processing hinban {unique_hinban}: {str(e)}")
            # logger.error(traceback.format_exc())  # スタックトレースを出力
            # st.write("test")
            continue

        #st.dataframe(zaiko_actuals_and_forecast_df)

        temp_df = zaiko_actuals_and_forecast_df

        # 判定
        temp_df["下限割れ"] = (temp_df["在庫数（箱）"] < temp_df["設計値MIN"]).astype(int)
        temp_df["上限越え"] = (temp_df["在庫数（箱）"] > temp_df["設計値MAX"]).astype(int)
        temp_df["在庫0"] = (temp_df["在庫数（箱）"] < 0).astype(int)

        # 各項目の合計を計算
        total_lower_limit = temp_df["下限割れ"].sum()
        total_upper_exceed = temp_df["上限越え"].sum()
        total_stock_zero = temp_df["在庫0"].sum()

        # 条件分岐でOK/NGに変換
        total_lower_limit = "NG" if total_lower_limit > 0 else "OK"
        total_upper_exceed = "NG" if total_upper_exceed > 0 else "OK"
        total_stock_zero = "NG" if total_stock_zero > 0 else "OK"

        # Matplotlibでプロット作成
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(temp_df["日時"], temp_df["在庫数（箱）"], label="在庫数（箱）", marker="o")
        ax.fill_between(temp_df["日時"],
         temp_df["設計値MIN"], temp_df["設計値MAX"], color="lightgray", alpha=0.5, label="設計値範囲 (MIN-MAX)")
        #これはいらないかも
        #ax.axhline(y=basedata_filtered["設計値MIN"].iloc[0], color="blue", linestyle="--", label="設計値MIN")
        #ax.axhline(y=basedata_filtered["設計値MAX"].iloc[0], color="red", linestyle="--", label="設計値MAX")

        # ---- グラフの装飾 ----
        ax.set_title("在庫予測結果と設計値（基準線）との比較", fontsize=14)
        ax.set_xlabel("日時", fontsize=12)
        ax.set_ylabel("在庫数（箱）", fontsize=12)
        ax.legend()
        ax.grid(True)
        plt.xticks(rotation=45)

        # 日時列をdatetime型に変換
        temp_df['日時'] = pd.to_datetime(temp_df['日時'])

        # 一番古い時刻を基準時刻（現在時刻）とする
        base_time = temp_df['日時'].min()

        # 基準時刻からの経過時間 (時間単位) を計算
        temp_df['経過時間(時間)'] = (temp_df['日時'] - base_time).dt.total_seconds() / 3600

        # 設計値MINを割る最初の時間
        time_min = temp_df.loc[temp_df['在庫数（箱）'] < temp_df['設計値MIN'], '経過時間(時間)'].min()

        # 設計値MAXを割る最初の時間
        time_max = temp_df.loc[temp_df['在庫数（箱）'] > temp_df['設計値MAX'], '経過時間(時間)'].min()

        # 在庫が0より小さくなる最初の時間
        time_zero = temp_df.loc[temp_df['在庫数（箱）'] < 0, '経過時間(時間)'].min()

        # ---- PNGファイルとして保存 ----
        save_dir = "outputs/在庫予測結果"
        os.makedirs(save_dir, exist_ok=True)
        output_file = f"{save_dir}/{unique_hinban}.png"
        fig.savefig(output_file, format="png", dpi=300, bbox_inches="tight")

        flag_display = 0
        hinban_indo_detail_df = get_hinban_info_detail(hinban_info, start_datetime, flag_display,flag_useDataBase, kojo)

        # 必要データだけ準備
        hinban_list.append(output_file)
        unique_hinmei = hinban_indo_detail_df['品名'].iloc[0]
        unique_shiresaki = hinban_indo_detail_df['仕入先名'].iloc[0]
        unique_shiresaki_kojo = hinban_indo_detail_df['仕入先工場名'].iloc[0]
        data_list.append({"品番_整備室": unique_hinban, "品名": unique_hinmei,
                           "仕入先名": unique_shiresaki, "発送工場名": unique_shiresaki_kojo,
                           "下限割れ":total_lower_limit,"上限越え":total_upper_exceed,"欠品":total_stock_zero,
                           "下限割れまでの時間":time_min,"上限越えまでの時間":time_max,"欠品までの時間":time_zero})

    # ローカルの PNG ファイルを Base64 エンコードする関数
    def img_to_base64(file_path):
        with open(file_path, "rb") as f:
            data = f.read()
        # Base64 エンコードして文字列に変換
        return base64.b64encode(data).decode("utf-8")

    # DataFrame に変換
    df_A = pd.DataFrame(data_list)
    # 画像を Base64 変換
    base64_images = [img_to_base64(p) for p in hinban_list]
    # DataFrame に変換
    df_B = pd.DataFrame(base64_images, columns=["画像base64"])

    #edited_df = st.data_editor(df_A, num_rows="dynamic")

    # DataFrame を統合（横方向に結合）
    data = pd.concat([df_A, df_B], axis=1)

    df = pd.DataFrame(data)
    #st.dataframe(df)

    # dfが空っぽだとエラー出る
    if len(df) != 0:

        #st.dataframe(df)

        #import csv

        st.divider()

        # 最後の列を除く
        df_excluded_last = df.iloc[:, :-1]

        # 3つの列それぞれについて、NGなら1、OKなら0に変換
        df_excluded_last['下限割れ'] = (df_excluded_last['下限割れ'] == 'NG').astype(int)
        df_excluded_last['上限越え'] = (df_excluded_last['上限越え'] == 'NG').astype(int)
        df_excluded_last['欠品'] = (df_excluded_last['欠品'] == 'NG').astype(int)

        # 例：'品番'列を分割する場合
        df_excluded_last[['品番', '整備室']] = df_excluded_last['品番_整備室'].str.split('_', expand=True)
        df_excluded_last['実行日時'] = start_datetime

        df_excluded_last = df_excluded_last.drop('品番_整備室', axis=1)

        # CSV文字列に変換（Shift_JISでエンコード）
        csv_data = df_excluded_last.to_csv(index=False, encoding='cp932')#, quoting=csv.QUOTE_ALL)

        # CSVをバイナリに変換 → Base64エンコード
        b64_encoded = base64.b64encode(csv_data.encode('cp932')).decode()

        # data URI スキームの文字列を作成
        # Shift_JIS で解釈されるよう charset=shift_jis も付与
        csv_uri = f"data:text/csv;charset=shift_jis;base64,{b64_encoded}"

        # カスタムHTMLボタン（例）
        custom_button = f"""
            <a download="在庫予測結果.csv" href="{csv_uri}">
                <button style="
                    background-color: #4CAF50;
                    border: none;
                    color: white;
                    padding: 12px 24px;
                    text-align: center;
                    text-decoration: none;
                    display: inline-block;
                    font-size: 16px;
                    border-radius: 8px;
                    cursor: pointer;
                ">
                📥 結果をダウンロードする
                </button>
            </a>
        """

        if run_mode == "手動実行":

            st.markdown(custom_button, unsafe_allow_html=True)

        def style_ok_ng(value):
            """OK/NG文字列を色付きバッジHTMLに変換"""
            if value == "OK":
                return """<span class="badge-ok">OK</span>"""
            elif value == "NG":
                return """<span class="badge-ng">NG</span>"""
            else:
                return str(value)

        #HTML 組み立て
        html_code = """
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="utf-8">
            <title>DataTables with Default Filters and Sorting</title>

            <!-- ▼ DataTables用CSS (CDN) ▼ -->
            <link rel="stylesheet" type="text/css"
                href="https://cdn.datatables.net/1.13.4/css/jquery.dataTables.css"/>
            <script type="text/javascript"
                    src="https://cdn.jsdelivr.net/npm/jquery@3.6.4/dist/jquery.min.js"></script>
            <script type="text/javascript"
                    src="https://cdn.datatables.net/1.13.4/js/jquery.dataTables.js"></script>

            <style>
            /* テーブルの基本スタイル */
            table {
                width: 100%;
                border-collapse: collapse;
            }
            th, td {
                border: 1px solid #ccc;
                padding: 8px;
                text-align: center;
            }

            /* OK/NGバッジのスタイル */
            .badge-ok {
                display: inline-block;
                padding: 5px 10px;
                color: #fff;
                background-color: #00aaff;
                border-radius: 5px;
            }
            .badge-ng {
                display: inline-block;
                padding: 5px 10px;
                color: #fff;
                background-color: #ff4444;
                border-radius: 5px;
            }

            /* トグルボタン */
            .toggle-button {
                padding: 8px 16px;
                font-size: 16px;
                background-color: #008CBA;
                color: white;
                border: none;
                border-radius: 6px;
                cursor: pointer;
            }
            .toggle-button:hover {
                background-color: #006F9A;
            }

            /* トグル表示部分（画像）は初期非表示 */
            .hidden-content {
                display: none;
                margin-top: 8px;
            }

            /* ソートセレクトボックスのスタイル */
            .filter-select {
                margin: 10px 0;
                padding: 6px 12px;
                font-size: 14px;
                border-radius: 4px;
                border: 1px solid #ccc;
            }
            </style>

            <script>
            $(document).ready(function() {
                // DataTables の初期化
                var table = $('#myTable').DataTable({
                    paging: true,
                    searching: true,
                    ordering: true,
                    pageLength: 10,
                    lengthMenu: [10, 20, 30],
                    order: [[5, 'asc']],  // デフォルトで「欠品までの時間」を降順にソート
                    columnDefs: [
                        {
                            targets: [7, 9, 5], // 数値列（下限割れ、上限越え、欠品までの時間）
                            render: function(data, type, row) {
                                if (type === 'sort' || type === 'type') {
                                    // NaN を -Infinity に変換してソート可能に
                                    return data === null || data === '' ? Infinity : parseFloat(data);
                                }
                                return data; // 表示時はそのまま
                            }
                        }
                    ]
                });

                // デフォルトフィルタ適用
                table.column(4).search('NG').draw();  // 欠品列を NG でフィルタ
                table.column(6).search('NG').draw();  // 下限割れ列を NG でフィルタ
                table.column(8).search('').draw();   // 上限越え列はフィルタなし（すべて）

                // 列単位フィルタの処理
                function doFilter() {
                    var valKekin = $('#filter_kekin').val();
                    var valKagen = $('#filter_kagen').val();
                    var valJougen = $('#filter_jougen').val();
                    table.column(4).search(valKekin).draw();
                    table.column(6).search(valKagen).draw();
                    table.column(8).search(valJougen).draw();
                }

                // フィルタが変更されたら doFilter を実行
                $('#filter_kekin').on('change', doFilter);
                $('#filter_kagen').on('change', doFilter);
                $('#filter_jougen').on('change', doFilter);

                // ソート条件の変更イベント
                $('#sort-order').on('change', function() {
                    var columnIndex = $(this).val();  // セレクトボックスの値を取得
                    table.order([columnIndex, 'asc']).draw();  // 指定列で降順ソート
                });
            });

            // トグル機能の実装
            function toggleImage(id) {
                var elem = document.getElementById(id);
                if (elem.style.display === 'none' || elem.style.display === '') {
                    elem.style.display = 'block';
                } else {
                    elem.style.display = 'none';
                }
            }
            </script>
        </head>
        <body>
            <!-- ▼ 列単位フィルタUI: 欠品, 下限割れ, 上限越え ▼ -->
            <div class="filter-boxes">
                <label>欠品:
                    <select id="filter_kekin" class="filter-select">
                        <option value="">(すべて)</option>
                        <option value="OK">OKのみ</option>
                        <option value="NG" selected>NGのみ</option>
                    </select>
                </label>

                <label>下限割れ:
                    <select id="filter_kagen" class="filter-select">
                        <option value="">(すべて)</option>
                        <option value="OK">OKのみ</option>
                        <option value="NG" selected>NGのみ</option>
                    </select>
                </label>

                <label>上限越え:
                    <select id="filter_jougen" class="filter-select">
                        <option value="" selected>(すべて)</option>
                        <option value="OK">OKのみ</option>
                        <option value="NG">NGのみ</option>
                    </select>
                </label>
            </div>

            <!-- ▼ ソート用セレクトボックスを追加 -->
            <div>
                <label for="sort-order">並び替え条件:</label>
                <select id="sort-order" class="filter-select">
                    <option value="6">下限割れまでの時間（大きい順）</option>
                    <option value="8">上限越えまでの時間（大きい順）</option>
                    <option value="4" selected>欠品までの時間（大きい順）</option>
                </select>
            </div>

            <!-- ▼ DataTables 対応テーブル -->
            <table id="myTable" class="display">
                <thead>
                    <tr>
                        <th>品番_整備室</th>
                        <th>品名</th>
                        <th>仕入先名</th>
                        <th>仕入先工場名</th>
                        <th>欠品</th>
                        <th>欠品までの時間</th>
                        <th>下限割れ</th>
                        <th>下限割れまでの時間</th>
                        <th>上限越え</th>
                        <th>上限越えまでの時間</th>
                        <th>グラフ</th>
                    </tr>
                </thead>
                <tbody>
        """

        # DataFrame の行をループして HTML に変換
        # for i, row in df.iterrows():
        #     html_code += f"""
        #     <tr>
        #         <td>{row['品番_整備室']}</td>
        #         <td>{row['品名']}</td>
        #         <td>{row['仕入先名']}</td>
        #         <td>{row['発送工場名']}</td>
        #         <td>{row['下限割れまでの時間']}</td>
        #         <td>{row['上限越えまでの時間']}</td>
        #         <td>{row['欠品までの時間']}</td>
        #         <td>{style_ok_ng(row['欠品'])}</td>
        #         <td>{style_ok_ng(row['下限割れ'])}</td>
        #         <td>{style_ok_ng(row['上限越え'])}</td>
        #         <td>
        #             <button class="toggle-button" onclick="toggleImage('hidden-content-{i}')">表示</button>
        #             <div id="hidden-content-{i}" class="hidden-content">
        #                 <img src="data:image/png;base64,{row['画像base64']}" style="max-width: 200px;">
        #             </div>
        #         </td>
        #     </tr>
        #     """
        for i, row in df.iterrows():
            html_code += f"""
            <tr>
                <td>{row['品番_整備室']}</td>
                <td>{row['品名']}</td>
                <td>{row['仕入先名']}</td>
                <td>{row['発送工場名']}</td>
                <td>{style_ok_ng(row['欠品'])}</td>
                <td>{row['欠品までの時間']}</td>
                <td>{style_ok_ng(row['下限割れ'])}</td>
                <td>{row['下限割れまでの時間']}</td>
                <td>{style_ok_ng(row['上限越え'])}</td>
                <td>{row['上限越えまでの時間']}</td>
                <td>
                    <button class="toggle-button" onclick="toggleImage('hidden-content-{i}')">表示</button>
                    <div id="hidden-content-{i}" class="hidden-content">
                        <img src="data:image/png;base64,{row['画像base64']}" style="max-width: 200px;">
                    </div>
                </td>
            </tr>
            """

        html_code += """
                </tbody>
            </table>
        </body>
        </html>
        """

        if run_mode == "手動実行":

            # 4) Streamlit で表示
            st.components.v1.html(html_code, height=1000, scrolling=True)

        #st.dataframe(df)

    else:
        st.write("すべての品番を取得できませんでした。選択した日時のデータは存在しません")
        df_excluded_last = 0

    return df_excluded_last

# MARK: 在庫予測表示（フロント）
def show_results_of_future_zaiko(target_column,  start_datetime, run_mode, out_parameter, kojo, flag_useDataBase):
    compute_future_zaiko(target_column,  start_datetime, run_mode, out_parameter, kojo, flag_useDataBase)

# MARK: 単独テスト用
if __name__ == "__main__":

    kojo = 'anjo1'
    hinban_info = ['3559850A010', '1Y']
    start_datetime = '2025-02-01 00:00:00'
    end_datetime = '2025-03-12 09:00:00'
    target_column = '納入予定日時'
    flag_useDataBase = 1
    selected_zaiko = 10

    # 在庫リミット計算
    out_parameter = "日量を採用する"
    #df = compute_zaiko_limit(hinban_info, target_column, start_datetime, selected_zaiko, out_parameter, kojo, flag_useDataBase)
    #print(df)

    # 在庫予測
    out_parameter = "日量を採用する"
    run_mode = "手動実行"
    df = compute_future_zaiko(target_column, start_datetime, run_mode, out_parameter, kojo, flag_useDataBase)
    #print(df)