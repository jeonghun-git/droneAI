from openai import OpenAI
import os
from dotenv import load_dotenv
import json
import requests
import time
from bs4 import BeautifulSoup
import re
from rich.console import Console
from rich.markdown import Markdown
from rich.live import Live

# Rich Console 초기화
console = Console()

def clean_text(text):
    cleaned_text = re.sub(r'\s+', ' ', text)
    cleaned_text = cleaned_text.strip()
    return cleaned_text

def enhanced_search(query: str):
    global last_search_context
    
    # 검색어 정보 업데이트
    last_search_context["query"] = query
    if "엔믹스" in query or "NMIXX" in query:
        last_search_context["topic"] = "엔믹스"
    elif "테슬라" in query:
        last_search_context["topic"] = "테슬라"
    
    print(f"🔍 검색 쿼리: {query}")
    response = requests.get(f"https://search.naver.com/search.naver?where=nexearch&sm=top_hty&fbm=0&ie=utf8&query={query}")

    if response.status_code == 200:
        soup = BeautifulSoup(response.content, 'lxml')
        main_pack = soup.find(id='main_pack')
        
        if main_pack:
            text_only = main_pack.get_text(separator='\n').strip()
            cleaned_text = clean_text(text_only)
            
            if len(cleaned_text) > 1500:
                cleaned_text = cleaned_text[:1500] + "..."
            
            return cleaned_text
        else:
            return "검색 결과를 찾을 수 없습니다."
    else:
        return f"검색 요청 실패. 상태 코드: {response.status_code}"

def routing(agent: str):
    return agent

TOOL_MAPPING = {
    "search": enhanced_search,
    "routing": routing,
}

TOOLS = [{
    "type": "function",
    "function": {
        "name": "search",
        "description": "Search the web for a given query.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The query to search for."
                }
            },
            "required": ["query"],
            "additionalProperties": False
        },
        "strict": True
    }
}]

ROUTING = [{
    "type": "function",
    "function": {
        "name": "routing",
        "description": "Route the user's request to the appropriate agent.",
        "parameters": {
            "type": "object",
            "properties": {
                "agent": {
                    "type": "string",
                    "enum": ["chat", "tool"],
                    "description": "The agent to route the user's request to.",
                }
            },
            "required": ["agent"],
            "additionalProperties": False,
        },
        "strict": True
    }
}]

# 전역 변수로 마지막 검색어 추적
last_search_context = {"query": None, "topic": None}

class AIAgent:
    def __init__(self, model: str = None, tools=None, endpoint: str = "https://openrouter.ai/api/v1", system_prompt: str = None, is_chat_agent: bool = False, **kwargs):
        load_dotenv()
        self.model = model
        self.system_prompt = system_prompt
        self.tools = tools
        self.is_chat_agent = is_chat_agent
        self.client = OpenAI(base_url=endpoint, api_key=os.getenv("OPENROUTER_API_KEY"))
        self.base_history = [{"role": "system", "content": system_prompt}]
        self.history = self.base_history.copy()
        self.last_search_query = None
        
    def text_response(self, user_prompt, context_info=None):
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
        
        # chat_agent인 경우 Live 컨텍스트 매니저 사용
        if self.is_chat_agent:
            with Live(Markdown(""), refresh_per_second=10, console=console) as live:
                for chunk in response:
                    delta = chunk.choices[0].delta
                    
                    if delta.content:
                        final_response += delta.content
                        # 실시간으로 마크다운 업데이트
                        live.update(Markdown(final_response))
                        
        else:
            # 일반 에이전트는 기존 방식 유지
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
                        
                        if json_buffer.count('{') == json_buffer.count('}') and json_buffer.count('{') > 0:
                            function_args = json_buffer
                            json_buffer = ""
        
        if final_response:
            self.history.append({"role": "assistant", "content": final_response})
            return final_response
        elif function_name:
            return (function_name, function_args)
        else:
            return ""

    def get_tool_response(self, *args):
        if len(args) != 2:
            return "도구 호출 오류"
            
        tool_name = args[0]
        raw_args = args[1]
        
        # JSON 유효성 검사 강화
        try:
            tool_args = json.loads(raw_args)
        except json.JSONDecodeError:
            # 마지막 중괄호 찾아서 잘라내기 시도
            last_brace = raw_args.rfind('}')
            if last_brace != -1:
                try:
                    tool_args = json.loads(raw_args[:last_brace+1])
                except:
                    # 정규식으로 query 값 추출
                    match = re.search(r'"query":\s*"([^"]*)"', raw_args)
                    if match:
                        tool_args = {"query": match.group(1)}
                    else:
                        tool_args = {"query": raw_args}
            else:
                tool_args = {"query": raw_args}
        
        # 검색어 저장
        if tool_name == "search":
            self.last_search_query = tool_args.get("query")

        tool_result = TOOL_MAPPING[tool_name](**tool_args)
        
        print(f"\nObservation: 검색 결과를 받았습니다.")
        print(f"Thought: 검색 결과를 바탕으로 최종 답변을 제공하겠습니다.")
        print(f"Final Answer: ")
        
        # 검색 결과를 히스토리에 추가
        self.history.append({
            "role": "assistant", 
            "content": f"검색 결과: {tool_result[:500]}..."
        })
        
        return tool_result

if __name__ == "__main__":
    route_agent = AIAgent(
        model="openai/gpt-4.1-mini",
        tools=ROUTING,
        system_prompt="""반드시 다음 JSON 형식으로만 응답하세요:
{"agent":"chat"} 또는 {"agent":"tool"}

현재 정보나 검색이 필요한 경우 "tool"을, 일반적인 대화는 "chat"을 선택하세요."""
    )

    chat_agent = AIAgent(
        model="deepseek/deepseek-chat-v3-0324:free",
        tools=None,
        is_chat_agent=True,  # 실시간 마크다운 렌더링 활성화
        endpoint="https://openrouter.ai/api/v1",
        system_prompt=f"""당신은 친근한 AI 어시스턴트입니다. 
답변을 마크다운 형식으로 작성하세요:

- **굵은 글씨**로 중요한 정보 강조
- `백틱`으로 특정 용어나 숫자 표시
- ## 제목으로 섹션 구분
- 📅 🎂 💡 🔍 등 이모지 활용
- > 인용문으로 부가 설명
- 목록은 - 또는 1. 사용

현재 시간: {time.strftime('%Y-%m-%d %H:%M:%S')}

예시:
## 🎂 생일 정보
**장규진**의 생일은 `2006년 5월 26일`입니다.

> 💡 **참고**: 현재 만 19세입니다."""
    )
    
    tool_agent = AIAgent(
        model="openai/gpt-4.1-mini",
        tools=TOOLS, 
        endpoint="https://openrouter.ai/api/v1",
        system_prompt=f"""당신은 사용자의 요청에 따라 웹 검색을 수행하는 AI 에이전트입니다.

검색어 생성 규칙:
1. 사용자가 구체적인 검색어를 제공하면 그대로 사용
2. "다시 알아봐줘", "재검색해줘" 등의 요청이면 이전 검색 주제를 참고하여 검색어 생성
3. 애매한 요청이면 사용자에게 명확히 질문
4. 시간 정보가 필요한 경우 현재 시간을 참고하여 검색어 생성

예시:
- "규진이 생일 몇일 남았지?" → "엔믹스 장규진 생일 남은 일수"
- "테슬라 주가 얼마야?" → "테슬라 실시간 주가"

현재 시간: {time.strftime('%Y-%m-%d %H:%M:%S')}"""
    )
    
    while True:
        try:
            user_prompt = input("\nYou: ")
            
            if user_prompt == "exit":
                break   
            
            print(f"\n🎯 사용자 요청: {user_prompt}")
            
            # 라우팅 결정
            route_result = route_agent.text_response(user_prompt)
            
            if isinstance(route_result, tuple):
                route = route_agent.get_tool_response(*route_result)
            else:
                route = route_result
            
            print(f"\n[라우팅: {route}]\n")
            
            if "chat" in route:
                chat_agent.text_response(user_prompt)
            elif "tool" in route:
                # 이전 검색 컨텍스트 정보 제공
                context_info = f"이전 검색 주제: {last_search_context.get('topic', '없음')}, 마지막 검색어: {last_search_context.get('query', '없음')}"
                
                tool_result = tool_agent.text_response(user_prompt, context_info=context_info)
                if isinstance(tool_result, tuple):
                    search_result = tool_agent.get_tool_response(*tool_result)
                    
                    summary = search_result[:800] if len(search_result) > 800 else search_result
                    final_response = chat_agent.text_response(
                        f"""검색 결과를 바탕으로 마크다운 형식으로 답변해주세요:

검색 결과: {summary}

사용자 질문: {user_prompt}

마크다운 형식으로 구조화하여 답변하세요:
- 제목은 ## 사용
- 중요 정보는 **굵게**
- 날짜/숫자는 `백틱`으로 감싸기
- 이모지 활용하여 시각적 효과
- > 인용문으로 부가 설명"""
                    )
                    
        except KeyboardInterrupt:
            print("\n👋 프로그램을 종료합니다.")
            break
        except Exception as e:
            print(f"\n⚠️ 오류 발생: {str(e)}")
            continue
