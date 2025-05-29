# Python 기반 슬림 이미지
FROM python:3.10-slim

# 시스템 패키지 설치 (Prophet 필수 의존성 포함)
RUN apt-get update && apt-get install -y \
    build-essential \
    libpython3-dev \
    libatlas-base-dev \
    libopenblas-dev \
    liblapack-dev \
    git \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# 작업 디렉토리 설정
WORKDIR /app

# 의존성 복사 및 설치
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 전체 소스 복사
COPY . .

# 모델 저장 폴더 생성 (없으면 에러 방지)
RUN mkdir -p models

# 환경변수 파일 복사는 CI 또는 실행 시 볼륨으로 주입할 것

# 기본 포트
EXPOSE 10283

# FastAPI 앱 실행 (gunicorn 대체 가능)
CMD ["uvicorn", "power_predict_api:app", "--host", "0.0.0.0", "--port", "10283"]
