from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from .routers import chat_router

app = FastAPI()

# CORS 설정 추가
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 라우터 포함
app.include_router(chat_router.router)

# 요청 로깅 미들웨어
@app.middleware("http")
async def log_requests(request: Request, call_next):
    print(f"🔍 요청: {request.method} {request.url.path}")
    response = await call_next(request)
    print(f"✅ 응답: {response.status_code}")
    return response

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 