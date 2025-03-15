# 개발 환경 설정

## 필수 구성 요소

- Node.js ≥ 18.0.0
- Git
- npm

## 로컬 개발

```bash
# 저장소 복제
git clone https://github.com/yamadashy/repomix.git
cd repomix

# 의존성 설치
npm install

# CLI 실행
npm run repomix
```

## Docker 개발

```bash
# 이미지 빌드
docker build -t repomix .

# 컨테이너 실행
docker run -v ./:/app -it --rm repomix
```

## 프로젝트 구조

```
src/
├── cli/          # CLI 구현
├── config/       # 구성 처리
├── core/         # 핵심 기능
└── shared/       # 공유 유틸리티
```

## 테스트

```bash
# 테스트 실행
npm run test

# 테스트 커버리지
npm run test-coverage

# 린트
npm run lint-biome
npm run lint-ts
npm run lint-secretlint
```

## 릴리스 프로세스

1. 버전 업데이트
```bash
npm version patch  # 또는 minor/major
```

2. 테스트 및 빌드 실행
```bash
npm run test-coverage
npm run build
```

3. 게시
```bash
npm publish
