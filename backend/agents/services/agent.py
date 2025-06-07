import json
import re
import os
import time
import asyncio
import uuid
from openai import OpenAI
from dotenv import load_dotenv
from ..tools.search_tools import TOOL_MAPPING
from ..tools.mcp_client import MCPClient


class AIAgent:
    def __init__(
        self,
        model: str = None,
        tools=None,
        endpoint: str = "https://openrouter.ai/api/v1",
        system_prompt: str = None,
        is_chat_agent: bool = False,
        mcp_use: bool = False,
        mcp_config_path: str = "mcp_servers.json",
    ):
        load_dotenv()
        self.model = model
        self.system_prompt = system_prompt
        self.is_chat_agent = is_chat_agent
        self.mcp_use = mcp_use
        self.mcp_client = None
        self.mcp_tools = []
        self.server_names = []

        self.client = OpenAI(base_url=endpoint, api_key=os.getenv("OPENROUTER_API_KEY"))
        self.base_history = [{"role": "system", "content": system_prompt}]
        self.history = self.base_history.copy()

        # 도구 로딩 로직
        self.tools = self._prepare_tools(tools, mcp_config_path)

    def _prepare_tools(self, existing_tools, mcp_config_path):
        """도구들을 조건에 따라 준비"""
        combined_tools = []

        if existing_tools:
            combined_tools.extend(existing_tools)

        if self.mcp_use:
            try:
                self._init_mcp_async(mcp_config_path)
            except Exception as e:
                print(f"MCP 초기화 실패: {e}")

        return combined_tools

    def _init_mcp_async(self, config_path):
        self.mcp_config_path = config_path

    async def _ensure_mcp_initialized(self):
        """MCP 클라이언트가 초기화되지 않았다면 초기화"""
        if self.mcp_use and self.mcp_client is None:
            try:
                # JSON 설정 파일 로드
                with open(self.mcp_config_path, "r", encoding="utf-8") as f:
                    config = json.load(f)

                # 환경 변수 치환
                config_str = json.dumps(config)
                for key, value in os.environ.items():
                    config_str = config_str.replace(f"${{{key}}}", value)
                config = json.loads(config_str)

                self.mcp_client = MCPClient()
                self.mcp_client.servers_config = config["mcpServers"]

                await self.mcp_client.connect_to_servers()
                self.mcp_tools = self.mcp_client.available_tools
                self.server_names = list(config["mcpServers"].keys())

                if self.mcp_tools:
                    self.tools.extend(self.mcp_tools)

                print(f"MCP 초기화 완료: {len(self.mcp_tools)}개 도구 로드됨")

            except Exception as e:
                print(f"MCP 초기화 실패: {e}")
                self.mcp_use = False

    def _is_mcp_tool(self, tool_name):
        if not self.mcp_use or not self.server_names:
            return False
        return any(tool_name.startswith(f"{server}_") for server in self.server_names)

    async def text_response(self, user_prompt, context_info=None):
        await self._ensure_mcp_initialized()

        if context_info:
            enhanced_prompt = f"{user_prompt}\n\n컨텍스트 정보: {context_info}"
        else:
            enhanced_prompt = user_prompt

        self.history.append({"role": "user", "content": enhanced_prompt})

        if len(self.history) > 10:
            self.history = [self.history[0]] + self.history[-8:]

        params = {
            "model": self.model,
            "messages": self.history,
            "stream": True,
        }

        if self.tools:
            params["tools"] = self.tools
            params["tool_choice"] = "auto"

        response = self.client.chat.completions.create(**params)

        final_response = ""
        function_name = ""
        function_args = ""
        json_buffer = ""

        for chunk in response:
            delta = chunk.choices[0].delta

            if delta.content:
                print(delta.content, end="", flush=True)
                final_response += delta.content

            elif delta.tool_calls:
                tool_call = delta.tool_calls[0]

                if tool_call.function.name:
                    function_name += tool_call.function.name
                    print(f"\nAction: {tool_call.function.name}", end="", flush=True)

                if tool_call.function.arguments:
                    json_buffer += tool_call.function.arguments
                    print(tool_call.function.arguments, end="", flush=True)

                    if (
                        json_buffer.count("{") == json_buffer.count("}")
                        and json_buffer.count("{") > 0
                    ):
                        function_args = json_buffer
                        json_buffer = ""

        if final_response:
            self.history.append({"role": "assistant", "content": final_response})
            return final_response
        elif function_name:
            # UUID로 tool_call_id 생성
            tool_call_id = f"call_{uuid.uuid4().hex[:12]}"

            # tool_calls 포함된 assistant 메시지 추가
            self.history.append(
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": tool_call_id,
                            "type": "function",
                            "function": {
                                "name": function_name,
                                "arguments": function_args,
                            },
                        }
                    ],
                }
            )

            return (function_name, function_args, tool_call_id)
        else:
            return ""

    async def get_tool_response(self, *args):
        if len(args) != 3:
            return "도구 호출 오류"

        tool_name, raw_args, tool_call_id = args

        # JSON 유효성 검사 강화
        try:
            tool_args = json.loads(raw_args)
        except json.JSONDecodeError:
            last_brace = raw_args.rfind("}")
            if last_brace != -1:
                try:
                    tool_args = json.loads(raw_args[: last_brace + 1])
                except:
                    match = re.search(r'"query":\s*"([^"]*)"', raw_args)
                    if match:
                        tool_args = {"query": match.group(1)}
                    else:
                        tool_args = {"query": raw_args}
            else:
                tool_args = {"query": raw_args}

        # MCP 도구 실행
        if self.mcp_use and self._is_mcp_tool(tool_name):
            await self._ensure_mcp_initialized()
            if self.mcp_client:
                tool_result = await self.mcp_client.execute_tool_call(
                    tool_name, tool_args
                )
            else:
                tool_result = "MCP 클라이언트 초기화 실패"

        # 기존 TOOL_MAPPING 도구 실행
        else:
            if tool_name in TOOL_MAPPING:
                tool_result = TOOL_MAPPING[tool_name](**tool_args)
            else:
                tool_result = f"도구 '{tool_name}'를 찾을 수 없습니다."

        # 도구 결과를 히스토리에 추가 (tool 메시지로)
        self.history.append(
            {
                "role": "tool",
                "tool_call_id": tool_call_id,
                "name": tool_name,
                "content": tool_result,
            }
        )

        print(f"\n도구 실행 완료. 응답 생성 중...\n")

        # LLM에 다시 요청해서 최종 응답 생성
        follow_up_response = self.client.chat.completions.create(
            model=self.model, messages=self.history, stream=True, max_tokens=1000
        )

        # 스트리밍으로 최종 응답 출력
        final_text = ""
        for chunk in follow_up_response:
            delta = chunk.choices[0].delta
            if delta.content:
                print(delta.content, end="", flush=True)
                final_text += delta.content

        # 최종 응답을 히스토리에 추가
        if final_text:
            self.history.append({"role": "assistant", "content": final_text})

        return final_text


# 테스트용 비동기 메인 함수
async def main():
    agent = AIAgent(
        model="openai/gpt-4.1-mini",
        system_prompt="You are a helpful assistant.",
        mcp_use=True,
    )

    print("AI Agent 시작됨! (MCP 활성화)")
    print("종료하려면 'quit' 입력")

    while True:
        try:
            user_prompt = input("\nUser: ")
            if user_prompt.lower() in ["quit", "exit"]:
                break

            response = await agent.text_response(user_prompt)

            # 도구 호출이 필요한 경우
            if isinstance(response, tuple):
                tool_result = await agent.get_tool_response(*response)
                print(f"\n도구 실행 결과: {tool_result}")

        except KeyboardInterrupt:
            print("\n프로그램을 종료합니다.")
            break
        except Exception as e:
            print(f"\n오류 발생: {e}")


if __name__ == "__main__":
    asyncio.run(main())
