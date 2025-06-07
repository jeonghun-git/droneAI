import re

def clean_text(text):
    """텍스트 정리: 여러 공백을 하나로 압축하고 앞뒤 공백 제거"""
    cleaned_text = re.sub(r'\s+', ' ', text)
    cleaned_text = cleaned_text.strip()
    return cleaned_text

def optimize_search_query(user_question: str) -> str:
    """사용자 질문을 검색에 최적화된 쿼리로 변환"""
    # 기본적인 전처리
    query = user_question.strip()
    
    # 공백 정리
    query = " ".join(query.split())
    
    return query if query else user_question 