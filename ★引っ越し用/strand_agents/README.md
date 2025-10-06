# AWSコスト管理エージェント (app.py)

This repository ships a single CLI program app.py that wires the Strands Agent framework
with AWS Bedrock Claude 3.5 Sonnet, several utility tools, and the AWS Knowledge MCP
endpoint. The agent is designed to answer questions about AWS cost utilisation while
surfacing high-spend services and proposing optimisation ideas.

## 前提条件
- Python 3.10 以上
- strands-agent 系列パッケージおよび mcp クライアントが利用可能であること
- Tavily 検索用の API キー (環境変数 TAVILY_API_KEY)
- AWS Bedrock へアクセス可能な認証情報 (環境変数、共有クレデンシャルファイルなど)
- Knowledge MCP エンドポイント (https://knowledge-mcp.global.api.aws) へ到達できるネットワーク

## 環境変数
- AWS_REGION (任意): Bedrock モデルを呼び出すリージョン。省略時は ap-northeast-1
- TAVILY_API_KEY (必須): Tavily 検索ツールが外部情報を取得する際に使用

## 使い方
1. 必要な依存ライブラリをインストールします。
       pip install -r requirements.txt  # 例: プロジェクトでrequirements.txtを管理している場合
2. Bedrock と MCP にアクセス可能な認証情報を設定します。
       aws configure  # ローカル実行時にプロファイルを設定していない場合
3. TAVILY_API_KEY を環境変数として設定します。
4. プログラムを起動します。
       python app.py
5. プロンプトに従い質問を入力すると、以下のツールを併用しながら回答が生成されます。

## 統合ツール
- use_aws : AWS Cost Explorer などのAPIを介して費用データを取得
- tavily_search : Web検索からコスト削減アイデアを収集
- current_time : 応答内に現在日時を掲載
- mcp.list_tools_sync() : Knowledge MCP 経由でAWSドキュメントを検索

## ファイル構成
- app.py : Bedrockモデル設定、MCPクライアント初期化、ユーザーCLIインターフェースを備えたメインスクリプト

## 開発メモ
- run_agent() 関数が Agent の組み立てと対話処理を担います。
- 追加ツールやプロンプトの調整は SYSTEM_PROMPT および tools 配列を更新してください。
- 実行前にネットワークや認証設定の疎通確認を行ってください。

