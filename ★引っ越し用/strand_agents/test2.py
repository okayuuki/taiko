# app.py
from strands import Agent
from strands.models import BedrockModel
from strands_tools import use_aws, current_time
from strands_tools.tavily import tavily_search
from strands.tools.mcp import MCPClient
from mcp.client.streamable_http import streamablehttp_client
import os

# MCPクライアントを作成
mcp = MCPClient(
    lambda: streamablehttp_client("https://knowledge-mcp.global.api.aws")
)

# Claude 3.7 Sonnet on Bedrock (例: us-west-2)
model = BedrockModel(
    model_id="apac.anthropic.claude-3-5-sonnet-20240620-v1:0",
    temperature=0.3,
    region_name=os.getenv("AWS_REGION", "ap-northeast-1"),
)

# 記事で紹介のポリシーを反映したシステムプロンプト
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
3. current_time() - 現在日時を取得する
4. mcp.list_tools_sync() - AWSドキュメントを検索
""".strip()

with mcp:

    mcp_tools = mcp.list_tools_sync()  # ← ここは with の中で呼ぶ

    agent = Agent(
        model=model,
        tools=[
            use_aws,
            current_time,
            #tavily_search,
            mcp_tools
        ],
        system_prompt=SYSTEM_PROMPT,
    )

    agent("Bedrock Agentcoreのランタイムってどんな機能？一言で説明して。")