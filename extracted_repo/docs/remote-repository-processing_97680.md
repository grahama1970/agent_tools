# 원격 저장소 처리

## 기본 사용법

공개 저장소 처리:
```bash
# 전체 URL 사용
repomix --remote https://github.com/user/repo

# GitHub 단축형 사용
repomix --remote user/repo
```

## 브랜치 및 커밋 선택

```bash
# 특정 브랜치
repomix --remote user/repo --remote-branch main

# 태그
repomix --remote user/repo --remote-branch v1.0.0

# 커밋 해시
repomix --remote user/repo --remote-branch 935b695
```

## 요구 사항

- Git이 설치되어 있어야 함
- 인터넷 연결
- 저장소에 대한 읽기 권한

## 출력 제어

```bash
# 사용자 지정 출력 위치
repomix --remote user/repo -o custom-output.xml

# XML 형식 사용
repomix --remote user/repo --style xml

# 주석 제거
repomix --remote user/repo --remove-comments
```

## Docker 사용

```bash
# 현재 디렉토리에서 처리 및 출력
docker run -v .:/app -it --rm ghcr.io/yamadashy/repomix \
  --remote user/repo

# 특정 디렉토리에 출력
docker run -v ./output:/app -it --rm ghcr.io/yamadashy/repomix \
  --remote user/repo
```

## 일반적인 문제

### 접근 문제
- 저장소가 공개되어 있는지 확인
- Git 설치 확인
- 인터넷 연결 확인

### 대용량 저장소
- `--include`를 사용하여 특정 경로 선택
- `--remove-comments` 활성화
- 브랜치별로 개별 처리
