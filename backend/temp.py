import asyncio
import json
import os
from typing import Optional, Dict, List, Any
from contextlib import AsyncExitStack

import httpx
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

MODEL = "deepseek/deepseek-v3"
CHUTES_ENDPOINT = "https://chutes.ai/app/chute/154ad01c-a431-5744-83c8-651215124360"

# DeepSeek V3 모델용 특수 토큰 상수
TOOL_CALLS_BEGIN = "<｜tool▁call▁begin｜>"
TOOL_CALL_BEGIN = "<｜tool▁call▁begin｜>"
TOOL_SEP = " "
TOOL_CALL_END = " "
TOOL_CALLS_END = " "
TOOL_OUTPUTS_BEGIN = " "
TOOL_OUTPUT_BEGIN = " "
TOOL_OUTPUT_END = " "
TOOL_OUTPUTS_END = " "
END_OF_SENTENCE = " "
USER = " "
ASSISTANT = " "

MCP_SERVERS = {
    "filesystem": {
        "command": "npx",
        "args": ["-y", "@modelcontextprotocol/server-filesystem", "/home"],
        "env": None
    },
    "brave_search": {
        "command": "npx",
        "args": ["-y", "@modelcontextprotocol/server-brave-search"],
        "env": {"BRAVE_API_KEY": os.getenv("BRAVE_API_KEY")}
    },
    "github": {
        "command": "npx", 
        "args": ["-y", "@modelcontextprotocol/server-github"],
        "env": {"GITHUB_PERSONAL_ACCESS_TOKEN": os.getenv("GITHUB_TOKEN")}
    },
    "postgres": {
        "command": "npx",
        "args": ["-y", "@modelcontextprotocol/server-postgres"],
        "env": {"DATABASE_URL": os.getenv("DATABASE_URL")}
    }
}

def convert_tool_format(tool):
    """도구 정의를 OpenAI 호환 형식으로 변환합니다."""
    converted_tool = {
        "type": "function",
        "function": {
            "name": tool.name,
            "description": tool.description,
            "parameters": {
                "type": "object",
                "properties": tool.inputSchema.get("properties", {}),
                "required": tool.inputSchema.get("required", [])
            }
        }
    }
    return converted_tool

def format_deepseek_system_prompt(tools):
    """DeepSeek V3 모델용 시스템 프롬프트를 포맷팅합니다."""
    system_prompt = "You are a helpful assistant with tool calling capabilities. "
    system_prompt += f"When a tool call is needed, you MUST use the following format to issue the call:\n"
    system_prompt += f"{TOOL_CALLS_BEGIN}{TOOL_CALL_BEGIN}function{TOOL_SEP}FUNCTION_NAME\n"
    system_prompt += "```json\n{\"param1\": \"value1\", \"param2\": \"value2\"}\n```"
    system_prompt += f"{TOOL_CALL_END}{TOOL_CALLS_END}\n\n"
    system_prompt += "Make sure the JSON is valid."
    system_prompt += "## Tools\n\n### Function\n\nYou have the following functions available:\n\n"
    
    for tool in tools:
        system_prompt += f"- `{tool['function']['name']}`:\n```json\n{json.dumps(tool, indent=2)}\n```\n"
    
    return system_prompt

def format_deepseek_messages(messages):
    """DeepSeek V3 모델용 메시지를 포맷팅합니다."""
    formatted_messages = []
    
    # 시스템 메시지 찾기
    system_content = None
    for msg in messages:
        if msg["role"] == "system":
            system_content = msg["content"]
            break
    
    # 시스템 메시지가 없으면 첫 번째 메시지로 추가
    if system_content:
        formatted_messages.append({"role": "system", "content": system_content})
    
    # 나머지 메시지 추가
    for msg in messages:
        if msg["role"] == "system":
            continue  # 시스템 메시지는 이미 처리함
        
        if msg["role"] == "user":
            formatted_content = f"{USER}{msg['content']}{ASSISTANT}"
            formatted_messages.append({"role": "user", "content": formatted_content})
        
        elif msg["role"] == "assistant":
            if "tool_calls" in msg and msg["tool_calls"]:
                # 도구 호출 메시지 포맷팅
                content = msg.get("content", "")
                tool_calls_text = TOOL_CALLS_BEGIN
                
                for i, tool_call in enumerate(msg["tool_calls"]):
                    func_name = tool_call["function"]["name"]
                    arguments = tool_call["function"]["arguments"]
                    
                    tool_call_text = f"{TOOL_CALL_BEGIN}function{TOOL_SEP}{func_name}\n"
                    tool_call_text += f"```json\n{arguments}\n```{TOOL_CALL_END}"
                    
                    if i == 0 and content:
                        tool_calls_text += content
                    
                    tool_calls_text += tool_call_text
                
                tool_calls_text += f"{TOOL_CALLS_END}{END_OF_SENTENCE}"
                formatted_messages.append({"role": "assistant", "content": tool_calls_text})
            
            else:
                # 일반 응답 메시지
                content = msg["content"]
                if content:
                    formatted_messages.append({"role": "assistant", "content": f"{content}{END_OF_SENTENCE}"})
        
        elif msg["role"] == "tool":
            # 도구 응답 메시지
            prefix = "Use the results below to formulate an answer to the user question unless additional information is needed."
            tool_output = f"{TOOL_OUTPUTS_BEGIN}{TOOL_OUTPUT_BEGIN}{msg['content']}{TOOL_OUTPUT_END}"
            formatted_messages.append({"role": "system", "content": f"{prefix} {tool_output}"})
    
    return formatted_messages

async def call_chutes_api(messages, tools=None):
    """Chutes API를 직접 호출하여 tool calling을 수행합니다."""
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {os.getenv('CHUTES_API_KEY')}"
    }
    
    # DeepSeek 모델용 메시지 포맷팅
    if tools:
        # 시스템 프롬프트 추가
        system_prompt = format_deepseek_system_prompt(tools)
        has_system = any(msg["role"] == "system" for msg in messages)
        
        if not has_system:
            messages = [{"role": "system", "content": system_prompt}] + messages
        else:
            # 기존 시스템 메시지 업데이트
            for i, msg in enumerate(messages):
                if msg["role"] == "system":
                    messages[i]["content"] = system_prompt + "\n\n" + msg["content"]
                    break
    
    # DeepSeek 모델용으로 메시지 변환
    formatted_messages = format_deepseek_messages(messages)
    
    payload = {
        "model": MODEL,
        "messages": formatted_messages,
        "max_tokens": 2000
    }
    
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{CHUTES_ENDPOINT}/api/v1/chat",
                headers=headers,
                json=payload,
                timeout=60.0
            )
            
            if response.status_code == 200:
                return response.json()
            else:
                return {"error": f"API 호출 실패: {response.status_code}", "details": response.text}
    except Exception as e:
        return {"error": f"API 요청 오류: {str(e)}"}

class MCPClient:
    def __init__(self):
        self.sessions: Dict[str, ClientSession] = {}
        self.exit_stack = AsyncExitStack()
        self.openai = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=os.getenv("OPENROUTER_API_KEY")
        )
        self.messages = []
        self.available_tools = []
        self.use_chutes_api = True

    async def connect_to_servers(self):
        print("MCP 서버들에 연결 중...")
        
        for server_name, config in MCP_SERVERS.items():
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
                
                response = await session.list_tools()
                tools = [convert_tool_format(tool) for tool in response.tools]
                
                for tool in tools:
                    tool["function"]["name"] = f"{server_name}_{tool['function']['name']}"
                
                self.sessions[server_name] = session
                self.available_tools.extend(tools)
                
                print(f"✓ {server_name} 서버 연결 완료 - 도구 {len(tools)}개 등록")
                
            except Exception as e:
                print(f"✗ {server_name} 서버 연결 실패: {e}")
                continue

        print(f"\n총 {len(self.available_tools)}개의 도구가 사용 가능합니다.")

    async def execute_tool_call(self, tool_name: str, tool_args: Dict[str, Any]) -> str:
        server_name = tool_name.split('_')[0]
        actual_tool_name = '_'.join(tool_name.split('_')[1:])
        
        if server_name not in self.sessions:
            return f"서버 '{server_name}'를 찾을 수 없습니다."
        
        try:
            session = self.sessions[server_name]
            result = await session.call_tool(actual_tool_name, tool_args)
            return str(result.content) if hasattr(result, 'content') else str(result)
        except Exception as e:
            return f"도구 실행 오류: {e}"

    async def parse_deepseek_response(self, response_content):
        """DeepSeek 모델의 응답에서 도구 호출 정보를 파싱합니다."""
        if TOOL_CALLS_BEGIN not in response_content:
            return None
            
        try:
            # 도구 호출 부분 추출
            tool_calls_text = response_content.split(TOOL_CALLS_BEGIN)[1].split(TOOL_CALLS_END)[0]
            tool_calls = []
            
            # 각 도구 호출 파싱
            for tool_call_text in tool_calls_text.split(TOOL_CALL_BEGIN)[1:]:
                if TOOL_CALL_END not in tool_call_text:
                    continue
                    
                parts = tool_call_text.split(TOOL_SEP)
                if len(parts) < 2:
                    continue
                    
                tool_type = parts[0].strip()
                remaining = parts[1].split(TOOL_CALL_END)[0]
                
                # 함수 이름과 인자 추출
                func_name_parts = remaining.split("\n```json", 1)
                if len(func_name_parts) < 2:
                    continue
                    
                func_name = func_name_parts[0].strip()
                arguments = func_name_parts[1].strip().rstrip("```").strip()
                
                tool_calls.append({
                    "type": tool_type,
                    "function": {
                        "name": func_name,
                        "arguments": arguments
                    },
                    "id": f"call_{len(tool_calls)}"
                })
            
            return tool_calls
        except Exception as e:
            print(f"도구 호출 파싱 오류: {e}")
            return None

    async def process_query(self, query: str) -> str:
        self.messages.append({
            "role": "user", 
            "content": query
        })

        try:
            if self.use_chutes_api:
                # Chutes API를 직접 호출하여 도구 호출 수행
                response_data = await call_chutes_api(self.messages, self.available_tools)
                
                if "error" in response_data:
                    return f"Chutes API 오류: {response_data['error']}"
                
                assistant_message = response_data["choices"][0]["message"]
                response_content = assistant_message["content"]
                
                # DeepSeek 특수 포맷에서 도구 호출 파싱
                tool_calls = await self.parse_deepseek_response(response_content)
                
                if tool_calls:
                    # 도구 호출 정보를 메시지에 추가
                    assistant_message_with_tools = {
                        "role": "assistant",
                        "content": response_content.split(TOOL_CALLS_BEGIN)[0].strip(),
                        "tool_calls": tool_calls
                    }
                    self.messages.append(assistant_message_with_tools)
                    
                    # 각 도구 호출 실행
                    for tool_call in tool_calls:
                        tool_name = tool_call["function"]["name"]
                        tool_args = json.loads(tool_call["function"]["arguments"]) if tool_call["function"]["arguments"] else {}
                        
                        print(f"🔧 도구 실행: {tool_name}")
                        result = await self.execute_tool_call(tool_name, tool_args)
                        
                        self.messages.append({
                            "role": "tool",
                            "tool_call_id": tool_call["id"],
                            "name": tool_name,
                            "content": result
                        })
                    
                    # 후속 응답을 위해 Chutes API 다시 호출
                    follow_up_data = await call_chutes_api(self.messages)
                    
                    if "error" in follow_up_data:
                        return f"Chutes API 후속 호출 오류: {follow_up_data['error']}"
                    
                    follow_up_content = follow_up_data["choices"][0]["message"]["content"]
                    # 특수 토큰 제거
                    final_content = follow_up_content.replace(END_OF_SENTENCE, "")
                    
                    # 마지막 응답 메시지 추가
                    self.messages.append({
                        "role": "assistant",
                        "content": final_content
                    })
                    
                    return final_content
                else:
                    # 도구 호출 없는 일반 응답
                    clean_content = response_content.replace(END_OF_SENTENCE, "")
                    self.messages.append({
                        "role": "assistant", 
                        "content": clean_content
                    })
                    return clean_content
            else:
                # OpenRouter API 사용
                response = self.openai.chat.completions.create(
                    model=MODEL,
                    tools=self.available_tools,
                    messages=self.messages,
                    max_tokens=2000
                )
                
                message = response.choices[0].message
                self.messages.append(message.model_dump())
                
                if message.tool_calls:
                    for tool_call in message.tool_calls:
                        tool_name = tool_call.function.name
                        tool_args = json.loads(tool_call.function.arguments) if tool_call.function.arguments else {}
                        
                        print(f"🔧 도구 실행: {tool_name}")
                        result = await self.execute_tool_call(tool_name, tool_args)
                        
                        self.messages.append({
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "name": tool_name,
                            "content": result
                        })
                    
                    follow_up_response = self.openai.chat.completions.create(
                        model=MODEL,
                        messages=self.messages,
                        max_tokens=2000
                    )
                    
                    return follow_up_response.choices[0].message.content
                else:
                    return message.content
            
        except Exception as e:
            return f"오류가 발생했습니다: {e}"

    async def chat_loop(self):
        print("\n🤖 DeepSeek-V3 MCP 클라이언트가 시작되었습니다!")
        print("질문을 입력하거나 'quit'를 입력하여 종료하세요.")
        print("사용 API:", "Chutes API" if self.use_chutes_api else "OpenRouter API")
        print("사용 가능한 서버:", list(self.sessions.keys()))
        
        while True:
            try:
                query = input("\n💬 질문: ").strip()
                
                if query.lower() in ['quit', 'exit', '종료']:
                    print("클라이언트를 종료합니다.")
                    break
                    
                if query.lower() == 'switch':
                    self.use_chutes_api = not self.use_chutes_api
                    print(f"API 변경됨: {'Chutes API' if self.use_chutes_api else 'OpenRouter API'}")
                    continue
                
                if not query:
                    continue
                
                print("🤔 처리 중...")
                result = await self.process_query(query)
                print("\n📝 응답:")
                print(result)
                
            except KeyboardInterrupt:
                print("\n\n클라이언트를 종료합니다.")
                break
            except Exception as e:
                print(f"오류: {e}")

    async def cleanup(self):
        await self.exit_stack.aclose()

async def main():
    client = MCPClient()
    try:
        await client.connect_to_servers()
        if client.sessions:
            await client.chat_loop()
        else:
            print("연결된 MCP 서버가 없습니다. 환경 변수를 확인해주세요.")
    finally:
        await client.cleanup()

if __name__ == "__main__":
    asyncio.run(main()) 