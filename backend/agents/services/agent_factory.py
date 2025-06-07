import time
from .agent import AIAgent
from ..tools.search_tools import TOOLS, ROUTING

# 에이전트 팩토리: 다양한 타입의 에이전트를 생성
def create_route_agent():
    """라우팅 에이전트 생성"""
    return AIAgent(
        model="openai/gpt-4.1-mini",
        tools=ROUTING,
        system_prompt="""사용자의 요청을 분석하여 적절한 에이전트로 라우팅하세요.

- 현재 정보나 검색이 필요한 경우: tool
- 일반적인 대화나 인사: chat

반드시 routing 함수를 호출하여 응답하세요."""
    )

def create_chat_agent():
    """대화 에이전트 생성"""
    return AIAgent(
        model="deepseek/deepseek-chat-v3-0324:free",
        tools=None,
        is_chat_agent=True,
        system_prompt=f"""You are a helpful assistant.
        
        
        적극적으로 대화를 진행하세요.

        _혹시 대화 중에 날씨 관련 질문이 있으면, 테이블 형식으로 응답하세요._
        _날씨 관련 질문이 없으면, 날씨 언급자체를 하지말고, 일반적인 대화를 진행하세요._
        
        ex) 
        (참고로, 혹시라도 "날씨" 관련 질문이 있으면 바로 테이블로 깔끔하게 알려줄게! 아니면 자연스럽게 재밌는 주제로 대화 이어갈게요✨)
        => 이러한 언급 조차 X

        Current time: {time.strftime('%Y-%m-%d %H:%M:%S')}"""
        )

def create_tool_agent():
    """도구 에이전트 생성"""
    return AIAgent(
        model="openai/gpt-4.1-mini",
        tools=TOOLS,
        system_prompt=f"""당신은 사용자의 요청에 따라 웹 검색을 수행하는 AI 에이전트입니다.

검색어 생성 규칙:
1. 사용자가 구체적인 검색어를 제공하면 그대로 사용
2. "다시 알아봐줘", "재검색해줘" 등의 요청이면 이전 검색 주제를 참고하여 검색어 생성
3. 애매한 요청이면 사용자에게 명확히 질문

현재 시간: {time.strftime('%Y-%m-%d %H:%M:%S')}"""
    ) 