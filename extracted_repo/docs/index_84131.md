---
layout: home
title: Repomix
titleTemplate: 코드베이스를 AI 친화적인 형식으로 패키징
aside: false
editLink: false

features:
  - icon: 🤖
    title: AI 최적화
    details: 코드베이스를 AI가 이해하고 처리하기 쉬운 형식으로 변환합니다.

  - icon: ⚙️
    title: Git 인식
    details: .gitignore 파일을 자동으로 인식하고 처리합니다.

  - icon: 🛡️
    title: 보안 중심
    details: Secretlint를 통합하여 민감한 정보를 감지하고 보호합니다.

  - icon: 📊
    title: 토큰 카운팅
    details: LLM 컨텍스트 제한을 위한 파일별 및 전체 토큰 수를 제공합니다.

---

<div class="cli-section">

## 빠른 시작

Repomix를 사용하여 패키지 파일(`repomix-output.txt`)을 생성한 후, 다음과 같은 프롬프트와 함께 AI 어시스턴트에게 전송할 수 있습니다:

```
이 파일은 저장소의 모든 파일을 하나로 통합한 것입니다.
코드를 리팩터링하고 싶으니 먼저 검토해 주세요.
```

AI는 전체 코드베이스를 분석하고 포괄적인 인사이트를 제공할 것입니다:

![Repomix 사용 예시 1](/images/docs/repomix-file-usage-1.png)

구체적인 변경 사항을 논의할 때는 AI가 코드 생성을 도와줍니다. Claude의 Artifacts와 같은 기능을 사용하면 상호 의존적인 여러 파일도 한 번에 받을 수 있습니다:

![Repomix 사용 예시 2](/images/docs/repomix-file-usage-2.png)

즐거운 코딩 되세요! 🚀



## 파워 유저 가이드

고급 사용자를 위해 Repomix는 CLI 인터페이스를 통해 다양한 사용자 정의 옵션을 제공합니다.

### 빠른 시작

프로젝트 디렉토리에서 설치 없이 바로 Repomix를 시작할 수 있습니다:

```bash
npx repomix
```

또는 반복 사용을 위해 전역 설치:

```bash
# npm으로 설치
npm install -g repomix

# 또는 yarn으로 설치
yarn global add repomix

# 또는 Homebrew로 설치 (macOS/Linux)
brew install repomix

# 그런 다음 아무 프로젝트 디렉토리에서 실행
repomix
```

이게 전부입니다! Repomix가 현재 디렉토리에 `repomix-output.txt` 파일을 생성하며, 이 파일에는 AI 친화적인 형식으로 정리된 전체 코드베이스가 포함됩니다.



### 사용법

전체 저장소를 패키징:

```bash
repomix
```

특정 디렉토리를 패키징:

```bash
repomix path/to/directory
```

[glob 패턴](https://github.com/mrmlnc/fast-glob?tab=readme-ov-file#pattern-syntax)을 사용하여 특정 파일이나 디렉토리를 지정:

```bash
repomix --include "src/**/*.ts,**/*.md"
```

특정 파일이나 디렉토리 제외:

```bash
repomix --ignore "**/*.log,tmp/"
```

원격 저장소 처리:
```bash
# 단축형 사용
npx repomix --remote yamadashy/repomix

# 전체 URL 사용 (브랜치 및 특정 경로 지원)
npx repomix --remote https://github.com/yamadashy/repomix
npx repomix --remote https://github.com/yamadashy/repomix/tree/main

# 커밋 URL 사용
npx repomix --remote https://github.com/yamadashy/repomix/commit/836abcd7335137228ad77feb28655d85712680f1
```

새 설정 파일(`repomix.config.json`) 초기화:

```bash
repomix --init
```

생성된 파일은 Claude, ChatGPT, Gemini와 같은 생성형 AI 도구와 함께 사용할 수 있습니다.

#### Docker 사용법

Docker를 사용하여 Repomix를 실행할 수도 있습니다 🐳  
격리된 환경에서 Repomix를 실행하거나 컨테이너를 선호하는 경우에 유용합니다.

기본 사용법(현재 디렉토리):

```bash
docker run -v .:/app -it --rm ghcr.io/yamadashy/repomix
```

특정 디렉토리를 처리:
```bash
docker run -v .:/app -it --rm ghcr.io/yamadashy/repomix path/to/directory
```

원격 저장소를 처리하고 `output` 디렉토리에 출력:

```bash
docker run -v ./output:/app -it --rm ghcr.io/yamadashy/repomix --remote https://github.com/yamadashy/repomix
```

### 출력 형식

선호하는 출력 형식을 선택하세요:

```bash
# XML 형식(기본값)
repomix --style xml

# Markdown 형식
repomix --style markdown

# 일반 텍스트 형식
repomix --style plain
```

### 사용자 정의

`repomix.config.json`을 생성하여 지속적인 설정을 관리할 수 있습니다:

```json
{
  "output": {
    "style": "markdown",
    "filePath": "custom-output.md",
    "removeComments": true,
    "showLineNumbers": true,
    "topFilesLength": 10
  },
  "ignore": {
    "customPatterns": ["*.test.ts", "docs/**"]
  }
}
```

### 더 많은 예제
::: tip
💡 전체 문서와 더 많은 예제는 [GitHub 저장소](https://github.com/yamadashy/repomix)를 참조하세요!
:::

</div>
