#!/usr/bin/env python3

import pandas as pd

def make_e_from_supplierfactory_to_seibishitu( input_filename, input_encoding, output_filename):
    
    """
    仕入先工場(仕入先名_仕入先工場名) → 整備室コード のエッジを作成

    手順:
    1) 「仕入先名」と「仕入先工場名」を「_」で連結し「仕入先名_仕入先工場名」列を作成
    2) 「仕入先名_仕入先工場名」と「品番_整備室コード」を抽出
    3) 「~id」「label(~label)」を追加
       - ~id: "supplierfactory_part_" + 連番(1始まり)
       - ~label: "PRODUCES_PART"
    出力: data/edges/e_from_supplierfactory_to_part.csv (UTF-8)
    """

    # 読み込み
    df = pd.read_csv(input_filename, encoding=input_encoding)

    # 必須カラムが無い場合の簡易補完
    if "整備室コード" not in df.columns:
        df["整備室コード"] = "1Y"
    if "仕入先名" not in df.columns:
        df["仕入先名"] = ""
    if "仕入先工場名" not in df.columns:
        df["仕入先工場名"] = ""

    # 1) 仕入先名_仕入先工場名 を作成
    df["仕入先名_仕入先工場名"] = (
        df["仕入先名"].astype(str).str.strip().fillna("")
        + "_"
        + df["仕入先工場名"].astype(str).str.strip().fillna("")
    )

    # 2) 必要列を抽出（連番用にインデックスをリセット）
    result_df = df[["仕入先名_仕入先工場名", "整備室コード"]].copy().reset_index(drop=True)

    # 3) 列の追加
    result_df["~id"] = "supplierfactory_seibishitu_" + (result_df.index + 1).astype(str)
    result_df["~label"] = "SUPPLIES_PART_TO"

    # 列並びと列名変更
    result_df = result_df[["~id", "仕入先名_仕入先工場名", "~label", "整備室コード"]]
    result_df = result_df.rename(columns={
        "仕入先名_仕入先工場名": "~from",
        "整備室コード": "~to",
    })

    # 出力（インデックスは不要）
    result_df.to_csv(output_filename, encoding="utf-8", index=False)

def make_e_from_seibishitu_to_part( input_filename, input_encoding, output_filename):

    """
    整備室 → 部品(品番_整備室コード) のエッジを生成します。

    処理内容:
    1) 「品番」と「整備室コード」を「_」で連結し「品番_整備室コード」列を作成
    2) 「整備室コード」「品番_整備室コード」を抽出（連番付与のためにインデックスをリセット）
    3) 追加列を作成
       - ~id: "part_kanban_" + 連番(1始まり)
       - ~label: "MANAGES_PART"
    4) 列名をリネームし、出力順を整形
       - 「整備室コード」→「~from」, 「品番_整備室コード」→「~to」
       - 出力列: ["~id", "~from", "~label", "~to"]

    入出力:
    - 入力CSV: data/所在管理_1.csv（Shift-JIS想定）
    - 出力CSV: data/edges/e_from_seibishitu_to_part.csv（UTF-8, index なし）
    """

    # データ読み込み
    df = pd.read_csv(input_filename, encoding=input_encoding)

    if "整備室コード" not in df.columns:
        df["整備室コード"] = "1Y"

    # 1) 品番_整備室コード の作成
    df["品番_整備室コード"] = (
        df["品番"].astype(str).str.strip().fillna("")
        + "_"
        + df["整備室コード"].astype(str).str.strip().fillna("")
    )

    # 2) 必要列を抽出（行番号をリセットして連番を作りやすくする）
    result_df = df[["整備室コード", "品番_整備室コード"]].copy().reset_index(drop=True)

    # 3) 列の追加
    #   ~id: 連番を付与
    result_df["~id"] = "seibishitu_part_" + (result_df.index + 1).astype(str)
    #   label: "MANAGES_PART"を指定
    result_df["~label"] = "MANAGES_PART"

    # 並びかえ
    result_df = result_df[["~id", "整備室コード", "~label", "品番_整備室コード"]]

    # 列名変更
    result_df = result_df.rename(columns={"整備室コード": "~from"})
    result_df = result_df.rename(columns={"品番_整備室コード": "~to"})

    # 保存（インデックス列は不要）
    result_df.to_csv(output_filename, encoding="utf-8", index=False)

def make_e_from_part_to_kanban( input_filename, input_encoding, output_filename):

    """
    部品(品番_整備室コード) → かんばん(かんばんシリアル) のエッジを生成します。

    処理内容:
    1) 「品番」と「整備室コード」を「_」で連結し「品番_整備室コード」列を作成
    2) 「品番_整備室コード」「かんばんシリアル」を抽出（連番付与のためにインデックスをリセット）
    3) 追加列を作成
       - ~id: "part_kanban_" + 連番(1始まり)
       - ~label: "HAS_KANBAN"
    4) 列名をリネームし、出力順を整形
       - 「品番_整備室コード」→「~from」, 「かんばんシリアル」→「~to」
       - 出力列: ["~id", "~from", "~label", "~to"]

    入出力:
    - 入力CSV: data/所在管理_1.csv（Shift-JIS想定）
    - 出力CSV: data/edges/e_from_part_to_kanban.csv（UTF-8, index なし）
    """

    # データ読み込み
    df = pd.read_csv(input_filename, encoding=input_encoding)

    if "整備室コード" not in df.columns:
        df["整備室コード"] = "1Y"

    # 1) 品番_整備室コード の作成
    df["品番_整備室コード"] = (
        df["品番"].astype(str).str.strip().fillna("")
        + "_"
        + df["整備室コード"].astype(str).str.strip().fillna("")
    )

    # 2) 必要列を抽出（行番号をリセットして連番を作りやすくする）
    result_df = df[["品番_整備室コード", "かんばんシリアル"]].copy().reset_index(drop=True)

    # 3) 列の追加
    #   ~id: 連番を付与
    result_df["~id"] = "part_kanban_" + (result_df.index + 1).astype(str)
    #   label: "HAS_KANBAN"を指定
    result_df["~label"] = "HAS_KANBAN"

    # 並びかえ
    result_df = result_df[["~id", "品番_整備室コード", "~label", "かんばんシリアル"]]

    # 列名変更
    result_df = result_df.rename(columns={"品番_整備室コード": "~from"})
    result_df = result_df.rename(columns={"かんばんシリアル": "~to"})

    # 保存（インデックス列は不要）
    result_df.to_csv(output_filename, encoding="utf-8", index=False)

if __name__ == "__main__":

    # 入力ファイル
    input_filename = "data/所在管理_1.csv"
    input_encoding = "shift-jis"

    # 仕入先名_仕入先工場名 → 整備室コード のエッジファイルを作成
    output_filename = "data/edges/e_from_supplierfactory_to_seibishitu.csv"
    make_e_from_supplierfactory_to_seibishitu( input_filename, input_encoding, output_filename)

    # 整備室コード → 部品（品番_整備室コード） のエッジファイルを作成
    output_filename = "data/edges/e_from_seibishitu_to_part.csv.csv"
    make_e_from_seibishitu_to_part( input_filename, input_encoding, output_filename)

    # 部品（品番_整備室コード）→ かんばん（かんばんシリアル）のエッジファイルを作成
    output_filename = "data/edges/e_from_part_to_kanban.csv"
    make_e_from_part_to_kanban( input_filename, input_encoding, output_filename)

    #　確認
    filename = "data/edges/e_from_supplierfactory_to_seibishitu.csv"
    encoding = "utf-8"
    df = pd.read_csv(filename, encoding=encoding)
    print(df)

