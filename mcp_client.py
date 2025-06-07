# DeepSeek V3 모델용 특수 토큰 상수
TOOL_CALLS_BEGIN = "<｜tool▁call▁begin｜>"
TOOL_CALL_BEGIN = "<｜tool▁call▁begin｜>"
TOOL_SEP = ""
TOOL_CALL_END = "<｜tool▁call▁end｜>"
TOOL_CALLS_END = "<｜tool▁calls▁end｜>"
TOOL_OUTPUTS_BEGIN = "<｜tool▁outputs▁begin｜>"
TOOL_OUTPUT_BEGIN = ""
TOOL_OUTPUT_END = "구"
TOOL_OUTPUTS_END = "<｜tool▁outputs▁end｜>"
END_OF_SENTENCE = ""
USER = ""
ASSISTANT = "×"

import os
import httpx

async def call_chutes_api(messages, tools=None):
    """Chutes API를 직접 호출하여 tool calling을 수행합니다."""
    api_token = os.getenv("CHUTES_API_KEY")
    
    headers = {
        "Authorization": f"Bearer {api_token}",
        "Content-Type": "application/json"
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
        "model": "deepseek-ai/DeepSeek-V3-0324",
        "messages": formatted_messages,
        "max_tokens": 2000,
        "temperature": 0.7
    }
    
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{CHUTES_ENDPOINT}/chat/completions",
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