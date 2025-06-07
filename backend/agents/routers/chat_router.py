import json
import time
import uuid
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from ..models.schemas import ChatCompletionRequest, ChatCompletionResponse, ChatCompletionChunk, Choice, Delta, Message, Usage
from ..services.agent_factory import create_route_agent, create_chat_agent, create_tool_agent
from ..services.streaming import process_and_stream_response, stream_agent_response
from ..tools.search_tools import enhanced_search

router = APIRouter(prefix="/api", tags=["chat"])

# 에이전트 초기화
route_agent = create_route_agent()
chat_agent = create_chat_agent()
tool_agent = create_tool_agent()

@router.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    try:
        user_message = request.messages[-1].content if request.messages else ""
        completion_id = f"chatcmpl-{str(uuid.uuid4())}"
        created_time = int(time.time())
        
        print(f"사용자: {user_message}")
        
        async def ai_stream_generator():
            try:
                print("AI 에이전트 시작")
                
                # LLM 기반 라우팅
                print("라우팅 결정 중...")
                route_agent.history = route_agent.base_history.copy()
                route_agent.history.append({"role": "user", "content": user_message})
                
                route_response = route_agent.client.chat.completions.create(
                    model=route_agent.model,
                    messages=route_agent.history,
                    tools=route_agent.tools,
                    tool_choice="auto",
                    stream=False
                )
                
                route = "chat"  # 기본값
                
                # 도구 호출 결과 확인
                if route_response.choices[0].message.tool_calls:
                    tool_call = route_response.choices[0].message.tool_calls[0]
                    if tool_call.function.name == "routing":
                        try:
                            args = json.loads(tool_call.function.arguments)
                            route = args.get("agent", "chat")
                        except:
                            route = "chat"
                else:
                    # 텍스트 응답에서 추출
                    content = route_response.choices[0].message.content.lower()
                    if "tool" in content:
                        route = "tool"
                
                print(f"📍 라우팅 결과: {route}")
                
                if route == "chat":
                    # 채팅 에이전트 직접 호출
                    # 요청의 messages 사용 (대화 히스토리 유지)
                    if request.messages:
                        # 시스템 프롬프트 유지를 위해 base_history의 첫 번째 메시지 사용
                        chat_agent.history = [chat_agent.base_history[0]]
                        # 사용자가 보낸 전체 대화 내역 추가
                        for msg in request.messages:
                            chat_agent.history.append({"role": msg.role, "content": msg.content})
                    else:
                        # 메시지가 없으면 새 대화 시작
                        chat_agent.history = chat_agent.base_history.copy()
                        chat_agent.history.append({"role": "user", "content": user_message})
                    
                    print(f"대화 히스토리 길이: {len(chat_agent.history)}")
                    
                    response = chat_agent.client.chat.completions.create(
                        model=chat_agent.model,
                        messages=chat_agent.history,
                        stream=True
                    )
                    
                    accumulated_response = ""
                    for chunk in response:
                        if chunk.choices and len(chunk.choices) > 0:
                            delta = chunk.choices[0].delta
                            if delta.content:
                                accumulated_response += delta.content
                                
                                chunk_data = {
                                    "id": completion_id,
                                    "object": "chat.completion.chunk",
                                    "created": created_time,
                                    "model": request.model,
                                    "choices": [
                                        {
                                            "index": 0,
                                            "delta": {
                                                "content": delta.content
                                            },
                                            "finish_reason": None
                                        }
                                    ]
                                }
                                yield f"data: {json.dumps(chunk_data, ensure_ascii=False)}\n\n"
                            
                            if chunk.choices[0].finish_reason == "stop":
                                break
                    
                    # 히스토리에 응답 추가
                    if accumulated_response:
                        chat_agent.history.append({"role": "assistant", "content": accumulated_response})
                
                else:
                    # 검색 쿼리 생성
                    print("🔧 검색 쿼리 생성 중...")
                    tool_agent.history = tool_agent.base_history.copy()
                    tool_agent.history.append({"role": "user", "content": f"다음 질문에 대한 최적의 검색어를 생성해주세요: {user_message}"})
                    
                    search_query_response = tool_agent.client.chat.completions.create(
                        model=tool_agent.model,
                        messages=tool_agent.history,
                        stream=False
                    )
                    
                    # 검색어 추출
                    search_query = search_query_response.choices[0].message.content
                    search_query = search_query.strip().strip('"').strip("'")
                    if "검색어:" in search_query:
                        search_query = search_query.split("검색어:")[-1].strip()
                    
                    print(f"🔍 생성된 검색 쿼리: {search_query}")
                    
                    # 검색 실행
                    search_result = enhanced_search(search_query)
                    print(f"📊 검색 완료: {len(search_result)} 글자")
                    
                    # 검색 결과를 바탕으로 최종 답변 생성
                    final_prompt = f"""검색 결과를 바탕으로 사용자의 질문에 답변해주세요:

검색 결과: {search_result}

사용자 질문: {user_message}"""
                    
                    # 히스토리 관리 개선
                    if request.messages:
                        # 시스템 프롬프트 유지
                        chat_agent.history = [chat_agent.base_history[0]]
                        # 이전 대화 내역 유지 (마지막 사용자 메시지는 검색 결과와 함께 대체)
                        for i, msg in enumerate(request.messages):
                            if i < len(request.messages) - 1:
                                chat_agent.history.append({"role": msg.role, "content": msg.content})
                        # 검색 결과가 포함된 프롬프트 추가
                        chat_agent.history.append({"role": "user", "content": final_prompt})
                    else:
                        chat_agent.history = chat_agent.base_history.copy()
                        chat_agent.history.append({"role": "user", "content": final_prompt})
                    
                    print(f"검색 모드 히스토리 길이: {len(chat_agent.history)}")
                    
                    response = chat_agent.client.chat.completions.create(
                        model=chat_agent.model,
                        messages=chat_agent.history,
                        stream=True
                    )
                    
                    accumulated_response = ""
                    for chunk in response:
                        if chunk.choices and len(chunk.choices) > 0:
                            delta = chunk.choices[0].delta
                            if delta.content:
                                accumulated_response += delta.content
                                
                                chunk_data = {
                                    "id": completion_id,
                                    "object": "chat.completion.chunk",
                                    "created": created_time,
                                    "model": request.model,
                                    "choices": [
                                        {
                                            "index": 0,
                                            "delta": {
                                                "content": delta.content
                                            },
                                            "finish_reason": None
                                        }
                                    ]
                                }
                                yield f"data: {json.dumps(chunk_data, ensure_ascii=False)}\n\n"
                            
                            if chunk.choices[0].finish_reason == "stop":
                                break
                    chat_agent.history.append({"role": "assistant", "content": accumulated_response})
                
                # 종료 청크
                final_chunk = {
                    "id": completion_id,
                    "object": "chat.completion.chunk",
                    "created": created_time,
                    "model": request.model,
                    "choices": [
                        {
                            "index": 0,
                            "delta": {},
                            "finish_reason": "stop"
                        }
                    ]
                }
                yield f"data: {json.dumps(final_chunk, ensure_ascii=False)}\n\n"
                yield "data: [DONE]\n\n"
                
                print("✅ AI 스트림 완료")
                
            except Exception as e:
                print(f"❌ AI 스트림 오류: {str(e)}")
                error_chunk = {
                    "id": completion_id,
                    "object": "chat.completion.chunk",
                    "created": created_time,
                    "model": request.model,
                    "choices": [
                        {
                            "index": 0,
                            "delta": {
                                "content": f"오류가 발생했습니다: {str(e)}"
                            },
                            "finish_reason": "stop"
                        }
                    ]
                }
                yield f"data: {json.dumps(error_chunk, ensure_ascii=False)}\n\n"
                yield "data: [DONE]\n\n"
        
        return StreamingResponse(
            ai_stream_generator(), 
            media_type="text/plain",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "Content-Type": "text/plain",
                "Access-Control-Allow-Origin": "*",
                "Access-Control-Allow-Headers": "*"
            }
        )
            
    except Exception as e:
        print(f"❌ 전체 오류: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/ask/custom")
async def ask_custom(request: ChatCompletionRequest):
    """jh-chat 커스텀 엔드포인트 - OpenAI 호환 형식"""
    try:
        user_message = request.messages[-1].content if request.messages else ""
        completion_id = f"chatcmpl-{str(uuid.uuid4())}"
        created_time = int(time.time())
        
        async def stream_generator():
            try:
                print("🚀 스트림 시작")
                chunk_count = 0
                
                # 대화 히스토리 설정
                if request.messages:
                    # 시스템 프롬프트 유지
                    chat_agent.history = [chat_agent.base_history[0]]
                    # 사용자가 보낸 모든 메시지 추가
                    for msg in request.messages:
                        chat_agent.history.append({"role": msg.role, "content": msg.content})
                
                # 실시간 스트리밍 응답
                async for chunk_data in process_and_stream_response(user_message, completion_id, created_time, request.model, chat_agent):
                    if chunk_data and chunk_data.strip():
                        chunk_count += 1
                        print(f"📤 청크 {chunk_count}: {len(chunk_data)} bytes")
                        yield chunk_data
                
                print(f"✅ 스트림 완료 (총 {chunk_count}개 청크)")
                
                # 스트림 종료
                final_chunk = ChatCompletionChunk(
                    id=completion_id,
                    object="chat.completion.chunk",
                    created=created_time,
                    model=request.model,
                    choices=[
                        Choice(
                            index=0,
                            delta=Delta(),
                            finish_reason="stop"
                        )
                    ]
                )
                yield f"data: {final_chunk.model_dump_json()}\n\n"
                yield "data: [DONE]\n\n"
                print("🏁 스트림 종료 신호 전송")
                
            except Exception as e:
                print(f"❌ 스트림 제너레이터 오류: {str(e)}")
                error_chunk = ChatCompletionChunk(
                    id=completion_id,
                    object="chat.completion.chunk",
                    created=created_time,
                    model=request.model,
                    choices=[
                        Choice(
                            index=0,
                            delta=Delta(content=f"오류가 발생했습니다: {str(e)}"),
                            finish_reason="stop"
                        )
                    ]
                )
                yield f"data: {error_chunk.model_dump_json()}\n\n"
                yield "data: [DONE]\n\n"
        
        return StreamingResponse(
            stream_generator(), 
            media_type="text/plain",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "Content-Type": "text/plain",
                "Access-Control-Allow-Origin": "*",
                "Access-Control-Allow-Headers": "*"
            }
        )
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/ask/custom/abort")
async def abort_custom_chat():
    """jh-chat 커스텀 엔드포인트 중단 요청 처리"""
    return {"message": "Chat aborted successfully"}

@router.get("/v1/models")
async def list_models():
    return {
        "object": "list",
        "data": [
            {
                "id": "Searching",
                "object": "model",
                "created": int(time.time()),
                "owned_by": "custom"
            }
        ]
    } 