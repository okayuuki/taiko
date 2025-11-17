# 必要なライブラリをインポート
from strands import Agent
from strands.tools.mcp.mcp_client import MCPClient
from mcp.client.streamable_http import streamablehttp_client
from bedrock_agentcore.runtime import BedrockAgentCoreApp

# AgentCoreランタイム用のAPIサーバーを作成
app = BedrockAgentCoreApp()

# エージェント呼び出し関数を、APIサーバーのエントリーポイントに設定
@app.entrypoint
async def invoke_agent(payload, context):

    # フロントエンドで入力されたプロンプトとAPIキーを取得
    prompt = payload.get("prompt")
    tavily_api_key = payload.get("tavily_api_key")

    ### この中が通常のStrandsのコード ----------------------------------
    # Tavily MCPサーバーを設定
    mcp = MCPClient(lambda: streamablehttp_client(
        f"https://mcp.tavily.com/mcp/?tavilyApiKey={tavily_api_key}"
    ))

    # MCPクライアントを起動したまま、エージェントを呼び出し
    with mcp:
        agent = Agent(
            model="anthropic.claude-3-5-sonnet-20240620-v1:0",
            tools=mcp.list_tools_sync()
        )

        # エージェントの応答をストリーミングで取得
        stream = agent.stream_async(prompt)
        async for event in stream:
            yield event
    ### ------------------------------------------------------------

# APIサーバーを起動
app.run()