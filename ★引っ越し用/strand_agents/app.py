"""AWSコスト最適化のためにBedrockエージェントとMCPツールを組み合わせて動作させるモジュール。"""
from strands import Agent
from strands.models import BedrockModel
from strands_tools import use_aws, current_time
from strands_tools.tavily import tavily_search
from strands.tools.mcp import MCPClient
from mcp.client.streamable_http import streamablehttp_client
import os

# Knowledge MCP エンドポイントへ接続するクライアントを初期化
mcp = MCPClient(
    lambda: streamablehttp_client("https://knowledge-mcp.global.api.aws")
)

# Bedrock Claude 3.5 Sonnet モデル設定
model = BedrockModel(
    model_id="apac.anthropic.claude-3-5-sonnet-20240620-v1:0",
    temperature=0.3,
    region_name=os.getenv("AWS_REGION", "ap-northeast-1"),
)

# エージェントへ渡すシステムプロンプト。利用可能なツールや応答方針を明示する。
SYSTEM_PROMPT = """
あなたはAWSのコスト管理アシスタントです。
あなたの役割は：
- 利用可能なツールを使用して正確な情報を提供する
- 特に利用料の高いサービスをハイライトする
- サービスの説明は不要でサービス名とコストだけをシンプルに出力
- どのサービスを掘り下げるか・コストを削減したいかユーザに尋ねる
- サービス(Service)単位や項目(UsageType)単位で見やすくまとめる
- コスト削減案を尋ねられたらtavily_search()を活用しつつ提案する
- 回答出力はマークダウン形式でよいが、太字(**)や表形式や絵文字は使用しない
以下のツールにアクセスできます：
1. use_aws() - AWSのAPIを使用して情報を取得する
2. tavily_search() - 最新情報をWebで検索する
3. current_time() - 現在日時を取得する
4. mcp.list_tools_sync() - AWSドキュメントを検索
""".strip()


def run_agent(user_input: str) -> None:
    """ユーザーの質問を受け取り、MCP経由のツールと共にエージェントを実行する。"""
    # MCP クライアントとのコンテキスト内でツール一覧を取得し、Agentへ渡す
    with mcp:
        mcp_tools = mcp.list_tools_sync()
        agent = Agent(
            model=model,
            tools=[
                use_aws,      # AWS API からのコスト情報取得
                current_time, # 応答内で現在日時を提示
                tavily_search, # コスト削減策の外部調査
                mcp_tools,    # AWSドキュメント検索のためのMCPツール群
            ],
            system_prompt=SYSTEM_PROMPT,
        )
        agent(user_input)


def main() -> None:
    """CLI入力を受け付けてエージェントを起動する。"""
    # 入力が空の場合は記事で紹介された想定質問をデフォルトとして使用
    user_input = input("質問を入力してください（例: 今月のコスト感をまとめて）> ").strip()
    if not user_input:
        user_input = "今月のコスト感をまとめて"
    run_agent(user_input)


if __name__ == "__main__":
    main()
