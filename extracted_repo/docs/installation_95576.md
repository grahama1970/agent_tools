# 설치

## npx 사용 (설치 불필요)

```bash
npx repomix
```

## 전역 설치

### npm
```bash
npm install -g repomix
```

### Yarn
```bash
yarn global add repomix
```

### Homebrew (macOS/Linux)
```bash
brew install repomix
```

## Docker 설치

Docker를 사용하면 환경 설정 문제를 피할 수 있어 가장 편리한 방법 중 하나입니다. 아래와 같이 실행하세요:

```bash
# 현재 디렉토리 처리
docker run -v .:/app -it --rm ghcr.io/yamadashy/repomix

# 특정 디렉토리 처리
docker run -v .:/app -it --rm ghcr.io/yamadashy/repomix path/to/directory

# 원격 저장소 처리
docker run -v ./output:/app -it --rm ghcr.io/yamadashy/repomix --remote yamadashy/repomix
```

## VSCode 확장 프로그램

커뮤니티에서 관리하는 [Repomix Runner](https://marketplace.visualstudio.com/items?itemName=DorianMassoulier.repomix-runner) 확장 프로그램을 통해 VSCode에서 직접 Repomix를 실행할 수 있습니다.

기능:
- 몇 번의 클릭으로 폴더 패키징
- 파일 또는 콘텐츠 모드로 복사
- 출력 파일 자동 정리
- repomix.config.json 지원

[VSCode 마켓플레이스](https://marketplace.visualstudio.com/items?itemName=DorianMassoulier.repomix-runner)에서 설치하세요.

## 시스템 요구 사항

- Node.js: 18.0.0 이상
- Git: 원격 저장소 처리 시 필요

## 설치 확인

설치가 완료된 후, 다음 명령어로 Repomix가 정상적으로 작동하는지 확인하세요:

```bash
repomix --version
repomix --help
```
