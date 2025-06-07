from pprint import pprint
import geocoder

import httpx
import asyncio
import json

# 현재 위치의 위경도 조회
g = geocoder.ip("me")
latitude, longitude = g.latlng
print(f"현재 위치: {latitude}, {longitude}")


API_KEY = "1fe5b7f4350252cf9fe61fb5641bf724"

radius = 3000

category = "FD6"


async def fetch_places():
    url = f"https://dapi.kakao.com/v2/local/search/category.json?category_group_code={category}&x={longitude}&y={latitude}&radius={radius}"
    headers = {"Authorization": f"KakaoAK {API_KEY}"}

    async with httpx.AsyncClient() as client:
        response = await client.get(url, headers=headers)
        data = response.json()

    # 결과 처리
    if data.get("documents"):
        print(json.dumps(data["documents"], indent=2, ensure_ascii=False))
        print(data.keys())

    else:
        print("검색 결과가 없습니다.")


# 비동기 함수 호출
asyncio.run(fetch_places())
