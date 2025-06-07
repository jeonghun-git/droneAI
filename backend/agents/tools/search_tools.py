import requests
from bs4 import BeautifulSoup
import re
from ..utils.text_utils import clean_text

# 전역 변수
last_search_context = {"query": None, "topic": None}

def enhanced_search(query: str):
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
    """라우팅 함수"""
    return agent

# 도구 정의
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