# app.py
from strands import Agent
from strands.models import BedrockModel
from strands_tools import use_aws, current_time
from strands_tools.tavily import tavily_search
import os

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
2. tavily_search() - 最新情報をWebで検索する
3. current_time() - 現在日時を取得する
""".strip()

agent = Agent(
    model=model,
    tools=[
        use_aws,
        current_time,
        tavily_search
    ],
    system_prompt=SYSTEM_PROMPT,
)

def main():
    # 起動時のデフォルト質問（記事にならう）
    user_input = input("質問を入力してください（例: 今月のコスト感をまとめて）> ").strip()
    if not user_input:
        user_input = "今月のコスト感をまとめて"
    agent(user_input)
    # resp = agent(user_input)
    # print("\n---- Agent Response ----\n")
    # print(resp)

if __name__ == "__main__":
    main()
