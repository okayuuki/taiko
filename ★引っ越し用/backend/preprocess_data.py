import pandas as pd
import streamlit as st
from scipy.stats import trim_mean
import plotly.graph_objects as go

#MARK: データ統合
def merge_data():

    #日時列を昇順にしておくこと

    # todo --------------------------------
    # CSVファイルのパスを指定
    file_path = '統合テーブル本番.csv'  
    # CSVファイルを読み込む
    merged_df = pd.read_csv(file_path, encoding='shift_jis')
    # None や NaN をすべて 0 に置き換える
    merged_df = merged_df.fillna(0)
    # todo --------------------------------

    #データ統合

    #返す

    return merged_df

# MARK:目的変数と説明変数の決定
def compute_features_and_target():

    # targetの計算
    def compute_target_variable(df, datetime_column, input_column, output_column, target_column, method, trim_ratio):
        
        """
        指定された集計方法（平均・中央値・トリム平均）に基づいて、
        各時刻（hour）ごとの在庫数の「いつもの傾向（目的変数）」を計算し、
        指定の列に出力する関数。

        処理の流れ：
        1. datetime列をdatetime型に変換し、そこからhour（時刻）を抽出する。
        2. hour（時刻）単位で在庫数をグルーピングする。
        3. 指定された手法（method）で各hourの代表値を算出する：
        - 'mean'：平均
        - 'median'：中央値
        - 'trim_mean'：上下一定割合を除外して平均
        4. 各行の時刻（hour）に応じて、算出した「いつもの在庫数」をoutput_columnに記録する。
        5. input_column - output_column の差を取り、目的変数（target_column）として記録。

        Parameters:
            df (pd.DataFrame): 対象データフレーム（日時と在庫数を含む）
            datetime_column (str): 日時列の列名（datetime型である必要があるが、内部で変換も実施）
            input_column (str): 現在の在庫数が記録されている列
            output_column (str): 「いつもの在庫数」を格納する列
            target_column (str): 差分としての目的変数を格納する列
            method (str): 集計方法。'mean', 'median', 'trim_mean' のいずれかを指定
            trim_ratio (float): トリム平均を使う場合に上下で除外する割合（例：0.1 → 上下10%ずつ除く）

        Returns:
            pd.DataFrame: 新しい列（output_column）に目的変数を追加したデータフレーム

        Example:
            入力データ：

                日時               | 在庫数
                -------------------|--------
                2025-03-29 09:00   |   50
                2025-03-28 09:00   |   55
                2025-03-27 09:00   |   53
                2025-03-29 15:00   |   42
                2025-03-28 15:00   |   45

            method='median' の場合：
                - 9時の中央値 = 53 → output_column = 53
                - 15時の中央値 = 43.5 → output_column = 43.5
                - 目的変数（target_column）＝ 在庫数 − output_column

            結果：
                在庫数 50 → 差分 = -3（= 50 - 53）
                在庫数 42 → 差分 = -1.5（= 42 - 43.5）
        """

        df = df.copy()
        df[datetime_column] = pd.to_datetime(df[datetime_column], errors='coerce')
        df['hour'] = df[datetime_column].dt.hour

        # 時刻ごとに処理を分岐
        grouped = df.groupby('hour')[input_column]

        if method == 'mean':
            hour_values = grouped.mean().to_dict()

        elif method == 'median':
            hour_values = grouped.median().to_dict()

        elif method == 'trim_mean':
            # 各 group を個別に処理して、トリム平均を計算
            hour_values = {
                hour: trim_mean(values, proportiontocut=trim_ratio)
                for hour, values in grouped
            }
        else:
            raise ValueError(f"不正なmethod指定: {method}. 'mean', 'median', 'trim_mean' のいずれかにしてください。")

        # 各行の hour に応じて「いつもの在庫数」を設定
        df[output_column] = df['hour'].map(hour_values)

        # 補助列を削除
        df.drop(columns=['hour'], inplace=True)

        # 目的変数を計算（現在の在庫数との差分）
        df[target_column] = df[input_column] - df[output_column]

        return df

    # feature_No1の計算
    # 実績（入庫実績）と設計（入庫予定）のデータを±X時間のタイムトレランス内でスナップ処理しマッチングさせる
    # 納入予定のある行が、近くの実績がある行にスナップされるようにする
    def compute_nyuuko_yotei_kanbansu_by_snapping_with_tolerance( df, kado_column, input_column,
                                                                  shiftted_column, lt_column, base_column, snap_column, time_tolerance):

        # 入庫予定かんばん数列の追加
        def compute_nyuuko_yotei_kanbansu( df, kado_column, input_column, shiftted_column, lt_column):

            """
            入庫予定かんばん数（shiftted_column）を計算してDataFrameに追加する関数
            shiftted_columnの値は、input_columnの値をリードタイムの値分先に進んだ行にシフトした値

            処理の流れ：
            1. input_column（例：発注かんばん数）が0でない行のみを対象とする。
            2. 各対象行について、リードタイム（lt_column）を取得し、
            3. 稼働フラグ（kado_column）が1の行をリードタイム件数分カウントしながら進める。
            4. リードタイム後の行が存在すれば、その行のshiftted_columnにinput_columnの値を加算する。

            Parameters:
                df (pd.DataFrame): 入出庫データを含むデータフレーム
                kado_column (str): 稼働フラグ列名（1=稼働、0=非稼働）
                input_column (str): 入庫元データ列名（例：発注かんばん数）
                shiftted_column (str): 出力列名（計算結果として追加する列）
                lt_column (str): リードタイム（日数）を示す列名

            Returns:
                pd.DataFrame: 入庫予定かんばん数列を追加したデータフレーム

            Example:
                以下のようなDataFrameがあったとする（リードタイム = 3）:

                    index | 稼働フラグ  | 発注かんばん数  | 納入予定かんばん数
                    ------|------------|----------------|-------------------
                    0     |     1      |      10        |        0
                    1     |     0      |       0        |        0
                    2     |     1      |       0        |        0
                    3     |     1      |       0        |        0
                    4     |     1      |       0        |       10  ← ここに納入予定が記録される！

                → index=0 の発注（10）は、リードタイム3行分の稼働である index=4 にスライドされ、
                その行の「納入予定かんばん数」に加算される。

                shiftted_column（納入予定）は、input_column（発注）より
                後ろの行に記録される点に注意
            """

            # shiftted_column列を初期化
            df[shiftted_column] = 0

            # 全行を走査して、input_column列の値 ≠ 0 の行のみ処理する
            for idx in df.index:

                # input_column列の値 = 0 ならその行はスキップする（入庫予定かんばん数の計算に関係がないため）
                # mergedの方でNoneやNanを0にできているはずだけど、念のため、NoneやNanも条件に含める
                # ここで0やNone、NaNをスキップできないと、後の処理が期待通りにならないので、注意すること
                if df.at[idx, input_column] == 0 or pd.isna(df.at[idx, input_column]):
                    continue

                # input_column列の値 ≠ 0 なら、リードタイムの値を確認する
                # lead_time = リードタイムの値
                lead_time = df.at[idx, lt_column]

                count = 0
                cursor = idx #リードタイム後のインデックス
                # リードタイム日数分だけ「稼働フラグが1の行」をカウントして進める
                # count = lead_time が基本終了条件
                # cursor + 1 < len(df) で データの末尾に到達しないように制限
                while count < lead_time and cursor + 1 < len(df):
                    cursor += 1
                    # 稼働フラグ列の値が1なら数を数える
                    if df.at[cursor, kado_column] == 1:
                        count += 1
                #st.write(df.at[idx, "日時"],df.at[idx, input_column],count)

                # リードタイム先が範囲内であればinput_column列の値をlead_time分シフトして、shiftted_column列の値に加算する
                if count == lead_time:
                    df.at[cursor, shiftted_column] += df.at[idx, input_column]

            return df
        
        # ±X時間のタイムトレランス内でスナップ処理
        def snap_with_tolerance(df, kado_column, shiftted_column, base_column, snap_column, time_tolerance):

            """
            ±X時間（稼働時間）以内にbase_columnの値がある場合に、shiftted_columnの値をスナップ（吸着）させる関数。

            処理の流れ：
            1. shiftted_column（納入予定かんばん数）に値がある行のみ対象とする。
            2. その行を中心として、稼働フラグが1の前後time_tolerance件分の稼働行を取得する。
            3. その範囲内に base_column（納入実績）が0でない行があれば、その最初の行に planned 値をスナップする。
            4. 実績が見つからなければ、自分自身の行に planned を保持する。

            Parameters:
                df (pd.DataFrame): 処理対象のデータフレーム
                kado_column (str): 稼働フラグ列（1=稼働、0=非稼働）
                shiftted_column (str): 納入予定かんばん数の列（スナップ対象）
                base_column (str): 納入実績かんばん数の列（スナップ先の基準）
                snap_column (str): 結果を格納する列（スナップ後の納入予定）
                time_tolerance (int): 許容する稼働日数（前後X件の稼働日）

            Returns:
                pd.DataFrame: snap_columnにスナップ結果を格納したデータフレーム

            Example:
                以下のようなデータを想定（time_tolerance=2）:

                    index | 稼働フラグ  | 納入予定　| 納入実績 | スナップ結果
                    ------|------------|----------|----------|--------------
                    0     |     1      |    5     |    0     |      0
                    1     |     1      |    0     |    4     |      5  ← index=0の予定がここにスナップ！
                    2     |     1      |    0     |    0     |      0
                    3     |     1      |    3     |    0     |      3  ← 実績がなければ自分自身に残る

                → スナップの対象は、納入予定がある行のみ（shiftted_column ≠ 0）
                → 前後の稼働行に納入実績があるかを調べ、最初に見つかった行へ移す
                → 見つからなければ、自身の行にそのまま残す
            """
            
            # 新しい列を0で初期化
            df[snap_column] = 0

            # 稼働日のインデックス一覧
            kado_indices = df[df[kado_column] == 1].index.tolist()

            for idx in df.index:
                planned = df.at[idx, shiftted_column]

                # 納入予定がない場合はスキップ
                if planned == 0:
                    continue

                # 現在の行が稼働日リストの何番目かを取得（見つからない場合はスキップ）
                try:
                    kado_pos = kado_indices.index(idx)
                except ValueError:
                    continue  # 稼働フラグ=0の行は対象外

                # 前後X件の稼働インデックス（自分を除く）
                nearby_kado_indices = kado_indices[max(0, kado_pos - time_tolerance):kado_pos] + \
                                    kado_indices[kado_pos + 1:kado_pos + 1 + time_tolerance]

                # 前後の稼働行の中で、納入かんばん数 ≠ 0 の行を探す（上から順に）
                matched_idx = None
                for near_idx in nearby_kado_indices:
                    if df.at[near_idx, base_column] != 0:
                        matched_idx = near_idx
                        break  # 一番最初に見つけた行でOK

                # 見つかればその行にコピー、見つからなければ自分にコピー
                if matched_idx is not None:
                    df.at[matched_idx, snap_column] += planned
                else:
                    df.at[idx, snap_column] += planned

            return df

        # 入庫予定かんばん数列の追加
        df = compute_nyuuko_yotei_kanbansu(df, kado_column, input_column, shiftted_column, lt_column)

        # 入庫かんばん数_スナップ済の追加
        df = snap_with_tolerance(df, kado_column, shiftted_column, base_column, snap_column, time_tolerance)

        # 既存dfに新しい列を追加して返す
        return df

    # feature_No2の計算
    # 直近のX時間の着工数を活用して、直近の生産状況を定量化する
    def quantify_recent_chakkousuu_status_and_trend(df, kado_column,
                                                              chakkou_column, window,
                                                              recent_chakkousuu_status_and_trend_column):
        
        """
        稼働していた直近X時間分の着工数から、直近の生産状況（着工傾向）を
        稼働フラグを重みとした加重平均で定量化する関数。

        処理の流れ：
        1. 各行について、その行までのデータを時系列で遡って取得する。
        2. その中から「稼働フラグ > 0」の行のみを抽出する。
        3. 抽出された稼働行の中から直近 window 件分を取得する。
        4. 着工数に稼働フラグを重みとして加重平均を計算し、
        結果を新しい列として現在の行に記録する。

        Parameters:
            df (pd.DataFrame): 入力データフレーム。kado_column（稼働フラグ）と
                            chakkou_column（着工数）を含む必要がある。
            kado_column (str): 稼働フラグ列の名前（例：1 = 完全稼働、0.5 = 半稼働、0 = 非稼働）。
            chakkou_column (str): 着工数を示す列名。
            window (int): 稼働時間ベースで何件分の履歴を使って傾向を算出するか。
            recent_chakkousuu_status_and_trend_column (str): 結果を格納する新しい列名。

        Returns:
            pd.DataFrame: 各行に着工傾向（加重平均）を示す新しい列を追加したDataFrame。

        Example:
            入力データ（window=3）の場合：

                index | 時刻   | 稼働フラグ | 着工数 | 着工傾向
                ------|--------|-----------|--------|----------
                0     | 8:00   |   1.0     |   10   |   10.0
                1     | 9:00   |   0.0     |    0   |   10.0
                2     | 10:00  |   0.5     |    6   |    8.0
                3     | 11:00  |   1.0     |    9   |    8.3
                4     | 12:00  |   1.0     |    8   |    8.5

            → 着工傾向列（加重平均）は、着工数 × 稼働フラグ の加重平均で算出される。
            例：index=3 のとき、直近3件（1.0×10, 0.5×6, 1.0×9）の加重平均は：
                (10×1 + 6×0.5 + 9×1) / (1 + 0.5 + 1) = 23 / 2.5 = 9.2

            ※ 稼働フラグが0でも着工数が0以外の場合は無視される（加重0なので影響なし）。
        """

        # 平均バージョン
        # df = df.copy()
        # df[recent_chakkousuu_status_and_trend_column] = None

        # for idx in df.index:

        #     # 現在の行までの範囲を取得（過去）
        #     df_up_to_now = df.loc[:idx]

        #     # 稼働していた行のみを抽出
        #     kado_rows = df_up_to_now[df_up_to_now[kado_column] != 0]

        #     # 直近の稼働X件だけを取得
        #     recent_kado = kado_rows.tail(window)

        #     # chakkou_columnの平均を計算
        #     mean_value = recent_kado[chakkou_column].mean()

        #     # 新しい列に代入
        #     df.at[idx, recent_chakkousuu_status_and_trend_column] = mean_value

        # return df

        # 稼働フラグとの加重平均バージョン
        df = df.copy()
        df[recent_chakkousuu_status_and_trend_column] = None

        for idx in df.index:

            # 現在の行まで取得
            df_up_to_now = df.loc[:idx]

            # 稼働行のみを抽出
            kado_rows = df_up_to_now[df_up_to_now[kado_column] != 0]

            # 直近window件の稼働行を取得
            recent_kado = kado_rows.tail(window)

            if not recent_kado.empty:
                # 加重平均を計算（稼働フラグ × 着工数） / 稼働フラグの合計
                weights = recent_kado[kado_column]
                values = recent_kado[chakkou_column]
                weighted_mean = (weights * values).sum() / weights.sum()
            else:
                weighted_mean = 0

            df.at[idx, recent_chakkousuu_status_and_trend_column] = weighted_mean

        return df

    # feature_No3の計算
    # 在庫水準
    # 現在庫は、過去にどれだけ発注され、どれだけ消費（回収）されたかの差分の累積によって形成される
    def compute_zaiko_level(df, kado_column, lt_column, input_column, window,
                            output_column):
        
        """
        稼働日ベースでLT時間分さかのぼり、そこからwindow件の移動平均を計算する。

        Parameters:
            df (pd.DataFrame): 対象のデータフレーム
            lt_column (str): LT（さかのぼる稼働時間）が入っている列名
            kado_column (str): 稼働フラグの列名（1=稼働、0=非稼働）
            input_column (str): 生産台数の列名
            window (int): 平均をとる稼働時間の行数
            output_column (str): 結果を保存する新しい列名
        """

        #todo 手配の方で計算してもいいかも
        df["かんばん回転日数"] = (df["サイクル間隔"] * (df["サイクル情報"] + 1)) / df["サイクル回数"]

        # 結果を保存する列を初期化
        df[output_column] = None

        # 稼働フラグが1の行のインデックスをリスト化
        kado_indices = df[df[kado_column] == 1].index.tolist()

        # 全行をループ
        for current_index in df.index:

            # LTの値を取得（何件分さかのぼるか）
            lt_value = df.at[current_index, lt_column]

            # LTが欠損または数値でない場合はスキップ
            if pd.isna(lt_value) or not isinstance(lt_value, (int, float)):
                continue

            # 現在の行が稼働日インデックスの何番目かを取得
            try:
                current_kado_pos = kado_indices.index(current_index)
            except ValueError:
                continue  # 万一見つからなければスキップ

            # LT件分だけさかのぼる（位置ベース）
            start_kado_pos = current_kado_pos - int(lt_value)
            if start_kado_pos < 0:
                continue  # 範囲外ならスキップ

            # LT位置から window 件分の稼働日を取得
            target_kado_indices = kado_indices[start_kado_pos : start_kado_pos + window]

            # 平均を計算
            if target_kado_indices:
                production_values = df.loc[target_kado_indices, input_column]
                average_value = production_values.mean()
                df.at[current_index, output_column] = average_value

        return df
    
    # 結果を表示する汎用関数
    def plot_result(df, datetime_column,
        value_columns,  # list型: 表示したい列（複数OK）
        flag_show, graph_title, yaxis_title,
        kado_column = None):
        
        """
        指定した日時列と複数の値列をPlotly+Streamlitで表示する汎用関数。

        Parameters:
            df (pd.DataFrame): 対象データ
            datetime_column (str): 時系列軸となる列（datetime型推奨）
            value_columns (list[str]): 可視化したい複数の列名（在庫数・いつも・差分など）
            flag_show (bool): 表示するかどうかのフラグ
            graph_title (str): グラフタイトル（任意指定可）
            yaxis_title (str): Y軸タイトル（任意指定可）
            kado_column (str or None): 稼働フラグの列名（任意）。指定されれば表に表示されます。
        """

        if not flag_show:
            return  # フラグがFalseなら何も表示しない

        # データの抽出と表示
        display_columns = [datetime_column] + value_columns
        if kado_column and kado_column in df.columns:
            display_columns.append(kado_column)

        temp_df = df[display_columns]
        st.dataframe(temp_df)

        # グラフ作成
        fig = go.Figure()

        # 各列をトレースとして追加
        dash_styles = ['solid', 'dot', 'dash', 'longdash', 'dashdot']  # 複数用
        for i, col in enumerate(value_columns):
            fig.add_trace(go.Scatter(
                x=temp_df[datetime_column],
                y=temp_df[col],
                mode='lines+markers',
                name=col,
                line=dict(dash=dash_styles[i % len(dash_styles)])  # 見やすく複数対応
            ))

        # レイアウト調整（ホバーモード・タイトルなど）
        fig.update_layout(
            title=graph_title,
            xaxis_title=datetime_column,
            yaxis_title=yaxis_title,
            hovermode='x unified',  # ← ホバーで同一日時の値をまとめて表示
            legend_title='凡例',
            template='plotly_white'
        )

        # グラフ表示
        st.plotly_chart(fig, use_container_width=True)
    
    
    # 統合データの読み込み
    merged_df = merge_data()
    st.dataframe(merged_df)

    # ★使用する列名が変わる可能性があるため変数で定義する

    # 結果を描画するか判定するためのフラグ変数
    # 描画する：True、しない：False
    flag_show = True

    # 目的変数の設定
    target_datetime_column = '日時'
    target_input_column = '在庫数（箱）'
    target_output_column = 'いつもの在庫数（箱）'
    target_column = 'target_在庫数（箱）-いつもの在庫数（箱）'
    target_method = 'median'
    target_trim_ratio = 0.1 #トリム平均を使う場合の切り捨て比率（例：0.1で上下10%カット）
    features_df = compute_target_variable(merged_df, target_datetime_column, target_input_column, target_output_column, target_column,
                            target_method, target_trim_ratio)
    # 結果の確認
    # 結果確認する列をリストとして定義
    value_columns = [target_input_column,target_output_column,target_column]
    # 結果を出力
    plot_result( features_df, target_datetime_column, value_columns, flag_show,
                 graph_title='在庫数の推移（実測・いつも・差分）', yaxis_title = '在庫数（箱）')

    # 説明変数の計算

    # 入庫予定かんばん数_スナップ済を計算する
    feature_No1_kado_column = '稼働フラグ'
    feature_No1_input_column = '納入予定日時のかんばん数'
    feature_No1_shiftted_column = '入庫予定かんばん数'
    feature_No1_lt_column = '納入LT(H)'
    feature_No1_base_column = "入庫（箱）" #todo 所在管理で再計算した方がいい？
    feature_No1_snap_column = "feature_No1_入庫予定かんばん数_スナップ済"
    feature_No1_time_tolerance = 1
    features_df = compute_nyuuko_yotei_kanbansu_by_snapping_with_tolerance( features_df, feature_No1_kado_column,
                                                                            feature_No1_input_column, feature_No1_shiftted_column,feature_No1_lt_column,
                                                                              feature_No1_base_column, feature_No1_snap_column, feature_No1_time_tolerance)
    # 結果の確認
    # 結果確認する列をリストとして定義
    value_columns = [feature_No1_input_column, feature_No1_shiftted_column, feature_No1_base_column, feature_No1_snap_column]
    # 結果を出力
    plot_result( features_df, target_datetime_column, value_columns, flag_show,
                 graph_title = 'IN関係の推移', yaxis_title = 'IN関係', kado_column = feature_No1_kado_column)

    

    # 物流センターから部品置き場の間の滞留かんばん数を計算する

    

    # 直近のX時間の着工数を活用して、直近の生産状況を定量化する
    feature_No2_kado_column = '稼働フラグ'
    feature_No2_chakkou_column = '生産台数'
    feature_No2_recent_chakkousuu_status_and_trend_column = 'feature_No2_最近の着工数の状況'
    feature_No2_window = 8
    features_df = quantify_recent_chakkousuu_status_and_trend(features_df, feature_No2_kado_column,
                                                              feature_No2_chakkou_column, feature_No2_window,
                                                              feature_No2_recent_chakkousuu_status_and_trend_column)
    # 結果の確認
    # 結果確認する列をリストとして定義
    value_columns = [feature_No2_chakkou_column,feature_No2_recent_chakkousuu_status_and_trend_column]
    # 結果を出力
    plot_result( features_df, target_datetime_column, value_columns, flag_show,
                 graph_title = '最近の着工数の推移', yaxis_title = '着工数', kado_column = feature_No2_kado_column)

    
    #todo 発注数と回収数がいる
    #todo 仕入先の稼働も入るから、アイシンの稼働フラグで計算するの難しいな
    # 在庫水準の計算
    feature_No3_kado_column = '稼働フラグ'
    feature_No3_lt_column = 'かんばん回転日数'
    feature_No3_input_column = '納入予定日時のかんばん数' #todo　仮
    feature_No3_output_column = 'feature_No3_過去のかんばん状況'
    feature_No3_window = 24*5
    features_df = compute_zaiko_level(features_df, feature_No3_kado_column, feature_No3_lt_column,
                                       feature_No3_input_column, feature_No3_window,
                                       feature_No3_output_column)
    # 結果の確認
    features_df_temp = features_df[['日時', feature_No3_kado_column, feature_No3_input_column,
                                     feature_No3_output_column]]
    st.dataframe(features_df_temp)
    # 結果の確認
    # 結果確認する列をリストとして定義
    value_columns = [feature_No3_input_column, feature_No3_output_column]
    # 結果を出力
    plot_result( features_df, target_datetime_column, value_columns, flag_show,
                 graph_title = '過去のかんばん推移', yaxis_title = 'かんばん数', kado_column = feature_No3_kado_column)

    
    return features_df

#MARK: 単独テスト
if __name__ == "__main__":
    
    print("test")

    df = compute_features_and_target()
    print(df)