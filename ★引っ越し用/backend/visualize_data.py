
import streamlit as st
import datetime as dt
import pandas as pd
import os
import json

# 自作ライブラリのimport（備忘：関数として認識されないときは、vscodeを再起動する）
from get_data import compute_hourly_buhin_zaiko_data_by_all_hinban

# 最終成果物ディレクトリー
#FINAL_OUTPUTS_PATH = 'outputs'
# ★相対パスで読み込み
# 現在のファイルの絶対パスを取得
current_dir = os.path.dirname(os.path.abspath(__file__))
# 目的のファイルの絶対パスを作成
FINAL_OUTPUTS_PATH = os.path.join(current_dir,"outputs")

# 設定ファイルの絶対パスを作成
CONFIG_PATH = os.path.join(current_dir, "..", "..", "configs", "settings.json")

# 日本語があるため、UTF-8でファイルを読み込む
with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
    config = json.load(f)
#　対象工場の読み込み
selected_data = config['selected_data']
selecte_kojo = selected_data["kojo"]
# 仕入先ダイヤファイルの読み込み
active_data_paths= config['active_data_path']
TEHAI_DATA_NAME =  active_data_paths[selecte_kojo]
#!

# 作成物の名前定義
#TEHAI_DATA_NAME = '手配必要数&手配運用情報テーブル.csv'

# 手配系データのフルパス
TEHAI_DATA_PATH = os.path.join( FINAL_OUTPUTS_PATH, TEHAI_DATA_NAME)

# ステップ０の処理、下限割れor上限越え品番の表示
def show_abnormal_results( selected_date, selected_time, flag_useDataBase, kojo):

    # 連続時間を計算
    def calculate_max_consecutive_time(selected_datetime, data, time_column, flag_column, group_columns):

        """
        品番と受入場所ごとの最大連続時間と対象時間を計算する関数。

        Parameters:
            data (pd.DataFrame): 処理対象のデータフレーム。
            time_column (str): 日時を示す列名。
            flag_column (str): 異常を示すフラグ列名。
            group_columns (list): グループ化の基準となる列名（例: 品番, 受入場所）。

        Returns:
            pd.DataFrame: 最大連続時間と対象時間を含む結果データフレーム。
        """

        results = []

        for group_keys, group in data.groupby(group_columns):
            group = group.sort_values(by=time_column)  # 日時でソート
            group['連続フラグ'] = (group[flag_column] != group[flag_column].shift()).cumsum()
            max_consecutive_time = 0
            max_time_range = ""
            going = ""

            # 下限割れ、上限越えしている区間
            for _, sub_group in group[group[flag_column] == 1].groupby('連続フラグ'):
                # 連続する区間の開始と終了を取得し、時間差を計算
                start_time = sub_group[time_column].min()
                end_time = sub_group[time_column].max()
                consecutive_time = (end_time - start_time).total_seconds() / 3600

                if consecutive_time > max_consecutive_time:
                    max_consecutive_time = consecutive_time
                    max_time_range = f"{start_time} ~ {end_time}"
                    if end_time == selected_datetime:
                        going = "進行中"
                    else:
                        going = "解消済"

            # 結果をリストに保存
            # result = {col: val for col, val in zip(group_columns, group_keys)}
            # result['品番'] = "_".join(map(str, group_keys))
            # result['連続時間（h）'] = int(max_consecutive_time)
            # result['対象時間'] = max_time_range
            result = {
                '品番':"_".join(map(str, group_keys)),
                '連続時間（h）' : int(max_consecutive_time),
                '対象時間': max_time_range,
                'ステータス':going,
            }
            results.append(result)

        # 結果をデータフレーム化して降順にソート
        results_df = pd.DataFrame(results).sort_values(by='連続時間（h）', ascending=False)
        return results_df
    
    # ランキング表示
    def create_abnormal_hinban_ranking(df_min: pd.DataFrame, df_max: pd.DataFrame):

        """
        2つのデータフレームを横並びで表示し、複数行がある場合はスクロールバーで閲覧できるようにする

        Parameters
        ----------
        df_min : pd.DataFrame
            1つ目（下限）のデータフレーム
        df_max : pd.DataFrame
            2つ目（上限）のデータフレーム
        """

        # 順位列を追加（1から始まる連番）
        df_min["危険順位"] = range(1, len(df_min) + 1)
        df_max["危険順位"] = range(1, len(df_max) + 1)

        # 順位列を一番左に移動
        columns = ["危険順位"] + [col for col in df_min.columns if col != "危険順位"]
        df_min = df_min[columns]
        columns = ["危険順位"] + [col for col in df_max.columns if col != "危険順位"]
        df_max = df_max[columns]

        # テーブルHTML部分
        df_min_html = df_min.to_html(index=False, classes="my-table", table_id="table1")
        df_max_html = df_max.to_html(index=False, classes="my-table", table_id="table2")

        # 条件を変数で指定
        column_name = "連続時間（h）"
        low_threshold = 3
        high_threshold = 8
        max_threshold = 9

        # CSS部分
        custom_css = """
        <style>
        .parent-container {
            display: flex;
            gap: 20px;
        }
        .table-container {
            flex: 1; 
            max-height: 500px;
            overflow-y: auto;
            border: 0px solid #ccc; /* 枠線 */
            padding: 8px;
            box-sizing: border-box;
        }
        table.my-table {
            border-collapse: collapse;
            width: 100%;
        }
        table.my-table th, table.my-table td {
            border: 1px solid #ddd;
            padding: 8px;
            text-align: left;
        }
        #table1 th {
            background-color: #9BD8FF; /* テーブル1のヘッダー行（薄い青） */
            color: #000; /* ヘッダーの文字色（黒） */
        }

        #table2 th {
            background-color: #FFB99B; /* テーブル2のヘッダー行（薄い赤） */
            color: #000; /* ヘッダーの文字色（黒） */
        }
        .table-container h3 {
            font-size: 30px;/* 表タイトルの文字サイズ */
        }
        </style>
        """

        # JavaScript（列名 & 閾値引数対応）
        # highlightRedIfOver(テーブルID, 列名, 閾値) を定義
        script = f"""
                <script>
                function highlightByValueRange(tableId, colName, lowThreshold, highThreshold, maxThreshold) {{
                    const table = document.getElementById(tableId);
                    if (!table) return;

                    // ヘッダ行の <th> を走査して、該当列のインデックスを見つける
                    const headerCells = table.getElementsByTagName('th');
                    let colIndex = -1;
                    for (let i = 0; i < headerCells.length; i++) {{
                        if (headerCells[i].innerText.trim() === colName) {{
                            colIndex = i;
                            break;
                        }}
                    }}
                    // 見つからなかったら何もしない
                    if (colIndex === -1) return;

                    // データ行をループして条件に応じた色を設定
                    const rows = table.getElementsByTagName('tr');
                    for (let r = 1; r < rows.length; r++) {{
                        const cells = rows[r].getElementsByTagName('td');
                        if (!cells[colIndex]) continue;

                        // 数値にパースして比較
                        let val = parseFloat(cells[colIndex].innerText);
                        if (!isNaN(val)) {{
                            if (val >= lowThreshold && val <= highThreshold) {{
                                cells[colIndex].style.color = 'orange';
                            }} else if (val >= maxThreshold) {{
                                cells[colIndex].style.color = 'red';
                            }}
                        }}
                    }}
                }}

                // ウィンドウ読み込み時に実行
                window.onload = function() {{
                    highlightByValueRange('table1', '{column_name}', {low_threshold}, {high_threshold}, {max_threshold});
                    highlightByValueRange('table2', '{column_name}', {low_threshold}, {high_threshold}, {max_threshold});
                }};
                </script>
        """

        # HTML全体
        combined_html = f"""
        <div class="parent-container">
            <div class="table-container">
                <h3>📉 下限割れ品番リスト</h3>
                {df_min_html}
            </div>
            <div class="table-container">
                <h3>📈 上限越え品番リスト</h3>
                {df_max_html}
            </div>
        </div>
        {script}
        """

        # 上で定義したCSS + HTMLを1つにまとめる
        html_content = custom_css + combined_html

        # st.components.v1.htmlを使ってHTML+JSを埋め込む
        # なぜ？
        # Streamlitのst.markdownやst.writeでHTML文字列を埋め込む場合、unsafe_allow_html=Trueを指定しても、
        # JavaScriptタグ（<script>）はサニタイズされて実行されないため、色変更処理の部分が動作しないから
        # Streamlit が用意している st.components.v1.html を使うと、HTML＋JavaScriptをまとめて実行できる
        st.components.v1.html(html_content, height=500)

    # 日付（YYYYMMDD）と時間（HH）を統合して日時変数を作成
    selected_datetime = dt.datetime.combine(selected_date, dt.datetime.strptime(selected_time, "%H:%M").time())
    # 実行結果の表示
    st.sidebar.code(f"新たに選択された日時: {selected_datetime}")

    # ステップ0のタイトル表示
    st.header("異常品番リスト")

    # 処理の説明
    st.write("**ステップ０の処理を受け付けました。下限割れor上限越えしている品番をリストアップします。常時上限越え品番は基準の定義や設計値の設定に問題があるため、除外しています。**")
    
    # 選択日時表示
    st.metric(label="選択日時", value=selected_datetime.strftime("%Y-%m-%d %H:%M"))

    # 探索時間前を設定
    # 選択した時間～過去24時間を見
    selected_datetime_start = (selected_datetime - dt.timedelta(hours=24)).strftime('%Y-%m-%d %H:%M:%S')
    selected_datetime_end = selected_datetime.strftime('%Y-%m-%d %H:%M:%S')

    # 自動ラックの在庫データを読み込み
    zaiko_df = compute_hourly_buhin_zaiko_data_by_all_hinban(selected_datetime_start, selected_datetime_end, flag_useDataBase, kojo)

    # 実行結果の確認
    # 開始時間と終了時間を取得
    #min_datetime = zaiko_df['日時'].min()
    #max_datetime = zaiko_df['日時'].max()
    #st.write(min_datetime, max_datetime)

    # 日付列を作成
    zaiko_df['日付'] = zaiko_df['日時'].dt.date

    # Activedata
    Activedata = pd.read_csv(TEHAI_DATA_PATH, encoding='shift_jis')
    #st.dataframe(Activedata)

    # 両方のデータフレームで '日付' 列を datetime 型に統一
    zaiko_df['日付'] = pd.to_datetime(zaiko_df['日付'])
    Activedata['日付'] = pd.to_datetime(Activedata['日付'])

    # 日時でデータフレームを結合
    zaiko_df = pd.merge(zaiko_df, Activedata, on=['品番','整備室コード','日付'])
    #
    #st.dataframe(zaiko_df)

    #! 常に上限を超えている品番を削除
    # 在庫数の中央値を計算（品番、整備室コードごと）
    zaiko_median = zaiko_df.groupby(['品番', '整備室コード'])['在庫数（箱）'].median().reset_index()
    zaiko_median.columns = ['品番', '整備室コード', '在庫数中央値']

    # 元のデータフレームに中央値をマージ
    zaiko_df = pd.merge(zaiko_df, zaiko_median, on=['品番', '整備室コード'])

    # 設計値MAXとの比較と除外処理（品番、整備室コードごと）
    # 各品番・整備室コードの組み合わせについて、すべての時点で設計値MAXを超えているものを特定
    exceed_max = zaiko_df.groupby(['品番', '整備室コード']).apply(
        lambda x: (x['在庫数（箱）'] > x['設計値MAX']).all()
    ).reset_index()
    exceed_max.columns = ['品番', '整備室コード', '常時超過']

    # 常時超過していない組み合わせのみを抽出
    valid_combinations = exceed_max[~exceed_max['常時超過']][['品番', '整備室コード']]
    zaiko_df = pd.merge(zaiko_df, valid_combinations, on=['品番', '整備室コード'])

    # 特定の時間帯のデータを抽出
    zaiko_df = zaiko_df[(zaiko_df['日時'] >= selected_datetime_start) & (zaiko_df['日時'] <= selected_datetime_end)]

    #column = ['日時','品番','受入場所','在庫数（箱）','設計値MIN','設計値MAX']
    #st.dataframe(zaiko_df[column].head(20000))

    data = zaiko_df

    # データフレームの初期処理: 新しい列「下限割れ」を作成
    data['下限割れ'] = 0
    data['上限越え'] = 0

    #todo temp（不等ピッチ考慮できていないので）
    data['設計値MAX'] = data['設計値MAX'] + 1

    # 「在庫数（箱）」が「設計値MIN」を下回っている場合、「下限割れ」を1に設定
    data.loc[data['在庫数（箱）'] < data['設計値MIN'], '下限割れ'] = 1
    data.loc[data['在庫数（箱）'] > data['設計値MAX'], '上限越え'] = 1

    # 日付列をdatetime型に変換
    data['日時'] = pd.to_datetime(data['日時'])

    # 結果をデータフレーム化
    # 関数を使用して計算
    results_min_df = calculate_max_consecutive_time(
        selected_datetime = selected_datetime,
        data=data,  # 入力データ
        time_column='日時',  # 日時列
        flag_column='下限割れ',  # フラグ列
        group_columns=['品番', '整備室コード']  # グループ化の基準列
    )
    results_min_df = results_min_df.sort_values(by='連続時間（h）', ascending=False).reset_index(drop=True)

    results_max_df = calculate_max_consecutive_time(
        selected_datetime = selected_datetime,
        data=data,  # 入力データ
        time_column='日時',  # 日時列
        flag_column='上限越え',  # フラグ列
        group_columns=['品番', '整備室コード']  # グループ化の基準列
    )
    results_max_df = results_max_df.sort_values(by='連続時間（h）', ascending=False).reset_index(drop=True)

    create_abnormal_hinban_ranking(results_min_df, results_max_df)

    # テキストエリアの表示
    st.markdown("""
        <div style="background-color: #ffffff; padding: 10px; border-radius: 5px;">
        📌 <strong>コラムの説明</strong><br>
        <ul>
        <li><strong>連続時間（h）：</strong> 下限割れ or 上限越えしていた連続時間（hour）を表しています</li>
        <li><strong>対象時間：</strong> 下限割れ or 上限越えしていた期間を表しています</li>
        <li><strong>ステータス：</strong> 「進行中」 ⇒ 選択した時刻でも異常発生中、「解消済」 ⇒ 選択した時刻では異常解消済み</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)


# MARK: 単独テスト用
if __name__ == "__main__":

    kojo = 'anjo1'
    hinban_info = ['3559850A010', '1Y']
    start_datetime = '2025-02-01 00:00:00'
    end_datetime = '2025-03-12 09:00:00'
    target_column = '納入予定日時'
    flag_useDataBase = 1
    selected_zaiko = 10

    