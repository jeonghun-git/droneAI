import asyncio
from typing import Optional, Dict
from contextlib import AsyncExitStack

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from openai import OpenAI
from dotenv import load_dotenv
import json
import os

load_dotenv()  # load environment variables from .env

MODEL = "deepseek/deepseek-chat-v3-0324"

# 기본 서버 설정 (하위 호환성을 위해 유지)
DEFAULT_SERVERS_CONFIG = {
    "context7": {"command": "npx", "args": ["-y", "@upstash/context7-mcp"]},
    "filesystem": {
        "command": "npx",
        "args": ["-y", "@modelcontextprotocol/server-filesystem", "/Users/jeonghun"],
        "env": None,
        ## mcp_server.json 을 사용하도록
    },
}


def convert_tool_format(tool):
    converted_tool = {
        "type": "function",
        "function": {
            "name": tool.name,
            "description": tool.description,
            "parameters": {
                "type": "object",
                "properties": tool.inputSchema["properties"],
                "required": tool.inputSchema["required"],
            },
        },
    }
    return converted_tool


class MCPClient:
    def __init__(self, servers_config=None):
        self.sessions: Dict[str, ClientSession] = {}  # 다중 세션 관리
        self.exit_stack = AsyncExitStack()
        self.available_tools = []  # 모든 서버의 도구 통합
        # 외부에서 설정을 받거나 기본값 사용
        self.servers_config = servers_config or DEFAULT_SERVERS_CONFIG

    async def connect_to_servers(self):
        """여러 MCP 서버에 연결"""
        print("MCP 서버들에 연결 중...")

        for server_name, config in self.servers_config.items():
            try:
                print(f"{server_name} 서버 연결 시도...")
                server_params = StdioServerParameters(**config)
                stdio_transport = await self.exit_stack.enter_async_context(
                    stdio_client(server_params)
                )
                stdio, write = stdio_transport
                session = await self.exit_stack.enter_async_context(
                    ClientSession(stdio, write)
                )

                await session.initialize()

                # 서버별 도구 목록 가져오기
                response = await session.list_tools()
                tools = [convert_tool_format(tool) for tool in response.tools]

                # 도구 이름에 서버 prefix 추가 (충돌 방지)
                for tool in tools:
                    tool["function"][
                        "name"
                    ] = f"{server_name}_{tool['function']['name']}"

                self.sessions[server_name] = session
                self.available_tools.extend(tools)

                print(f"✓ {server_name} 서버 연결 완료 - 도구 {len(tools)}개 등록")

            except Exception as e:
                print(f"✗ {server_name} 서버 연결 실패: {e}")
                continue

        print(f"\n총 {len(self.available_tools)}개의 도구가 사용 가능합니다.")

    async def execute_tool_call(self, tool_name: str, tool_args: dict) -> str:
        """도구 실행 - 서버별로 라우팅"""
        server_name = tool_name.split("_")[0]
        actual_tool_name = "_".join(tool_name.split("_")[1:])

        if server_name not in self.sessions:
            return f"서버 '{server_name}'를 찾을 수 없습니다."

        try:
            session = self.sessions[server_name]
            result = await session.call_tool(actual_tool_name, tool_args)
            return str(result.content) if hasattr(result, "content") else str(result)
        except Exception as e:
            return f"도구 실행 오류: {e}"

    async def cleanup(self):
        await self.exit_stack.aclose()


# 테스트용 메인 함수 (독립 실행 시에만 사용)
async def main():
    client = MCPClient()
    try:
        await client.connect_to_servers()
        if client.sessions:
            print("MCP 클라이언트 테스트 모드")
            print("사용 가능한 도구들:")
            for tool in client.available_tools:
                print(f"- {tool['function']['name']}")
        else:
            print("연결된 MCP 서버가 없습니다.")
    finally:
        await client.cleanup()


if __name__ == "__main__":
    import sys

    asyncio.run(main())
