import json
import time
from ..models.schemas import ChatCompletionChunk, Choice, Delta
from ..tools.search_tools import enhanced_search

async def stream_agent_response(agent, prompt):
    """에이전트 응답을 실시간으로 스트리밍"""
    # 기존 히스토리에 사용자 메시지 추가
    agent.history.append({"role": "user", "content": prompt})
    
    if len(agent.history) > 10:
        agent.history = [agent.history[0]] + agent.history[-8:]
    
    params = {
        "model": agent.model,
        "messages": agent.history,
        "stream": True,
    }
    
    if agent.tools:
        params["tools"] = agent.tools
        params["tool_choice"] = "auto"
    
    try:
        response = agent.client.chat.completions.create(**params)
        
        accumulated_response = ""
        function_name = ""
        function_args = ""
        json_buffer = ""
        has_content = False
        
        for chunk in response:
            if chunk.choices and len(chunk.choices) > 0:
                delta = chunk.choices[0].delta
                
                if delta.content:
                    accumulated_response += delta.content
                    has_content = True
                    yield delta.content  # 실시간으로 청크 전송
                    
                elif delta.tool_calls:
                    tool_call = delta.tool_calls[0]
                    
                    if tool_call.function.name:
                        function_name += tool_call.function.name
                        yield f"\n🔧 도구 실행: {tool_call.function.name}\n"
                        
                    if tool_call.function.arguments:
                        json_buffer += tool_call.function.arguments
                        
                        if json_buffer.count('{') == json_buffer.count('}') and json_buffer.count('{') > 0:
                            function_args = json_buffer
                            json_buffer = ""
                
                # 스트림 완료 확인
                if chunk.choices[0].finish_reason == "stop":
                    break
        
        # 응답을 히스토리에 추가
        if accumulated_response:
            agent.history.append({"role": "assistant", "content": accumulated_response})
        elif function_name and function_args:
            # 도구 호출 결과 처리
            tool_result = agent.get_tool_response(function_name, function_args)
            yield f"\n📊 검색 결과를 받았습니다.\n\n"
        
        # 스트림이 비어있는 경우 기본 응답
        if not has_content and not function_name:
            yield "응답을 생성할 수 없습니다."
            
    except Exception as e:
        yield f"에이전트 오류: {str(e)}"

async def process_and_stream_response(user_prompt: str, completion_id: str, created_time: int, model: str, chat_agent):
    """실시간으로 에이전트 응답을 스트리밍"""
    try:
        print(f"\n🎯 사용자 요청: {user_prompt}")
        
        # 임시로 chat으로 고정 (디버깅용)
        route = "chat"
        print(f"\n[라우팅: {route}] (임시 고정)\n")
        
        if route == "chat":
            # 채팅 에이전트 응답을 실시간 스트리밍
            async for chunk_text in stream_agent_response(chat_agent, user_prompt):
                if chunk_text and chunk_text.strip():
                    chunk = ChatCompletionChunk(
                        id=completion_id,
                        object="chat.completion.chunk",
                        created=created_time,
                        model=model,
                        choices=[
                            Choice(
                                index=0,
                                delta=Delta(content=chunk_text),
                                finish_reason=None
                            )
                        ]
                    )
                    yield f"data: {chunk.model_dump_json()}\n\n"
            
    except Exception as e:
        print(f"스트림 오류: {str(e)}")
        error_text = f"오류가 발생했습니다: {str(e)}"
        chunk = ChatCompletionChunk(
            id=completion_id,
            object="chat.completion.chunk",
            created=created_time,
            model=model,
            choices=[
                Choice(
                    index=0,
                    delta=Delta(content=error_text),
                    finish_reason="stop"
                )
            ]
        )
        yield f"data: {chunk.model_dump_json()}\n\n" 