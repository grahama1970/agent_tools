---
layout: home
title: Repomix
titleTemplate: 将代码库打包成AI友好的格式
aside: false
editLink: false

features:
  - icon: 🤖
    title: AI 优化
    details: 以 AI 易于理解和处理的方式格式化代码库。

  - icon: ⚙️
    title: Git 感知
    details: 自动识别并尊重您的 .gitignore 文件。

  - icon: 🛡️
    title: 注重安全
    details: 集成 Secretlint 进行强大的安全检查，检测并防止敏感信息的泄露。

  - icon: 📊
    title: 令牌计数
    details: 提供每个文件和整个代码库的令牌计数，便于控制 LLM 上下文限制。

---

<div class="cli-section">

## 快速开始

使用 Repomix 生成打包文件（`repomix-output.txt`）后，您可以将其发送给 AI 助手，并附上这样的提示：

```
此文件包含了仓库中所有文件的合并内容。
我想重构代码，请先帮我审查一下。
```

AI 将分析您的整个代码库并提供全面的见解：

![Repomix 使用示例1](/images/docs/repomix-file-usage-1.png)

在讨论具体修改时，AI 可以帮助生成代码。通过像 Claude 的 Artifacts 这样的功能，您甚至可以一次性接收多个相互依赖的文件：

![Repomix 使用示例2](/images/docs/repomix-file-usage-2.png)

祝您编码愉快！🚀



## 进阶使用指南

对于需要更多控制的高级用户，Repomix 通过其 CLI 界面提供了广泛的自定义选项。

### 快速上手

您可以在项目目录中无需安装即可立即尝试 Repomix：

```bash
npx repomix
```

或者全局安装以便重复使用：

```bash
# 使用 npm 安装
npm install -g repomix

# 或使用 yarn 安装
yarn global add repomix

# 或使用 Homebrew 安装（macOS/Linux）
brew install repomix

# 然后在任意项目目录中运行
repomix
```

就是这么简单！Repomix 将在您的当前目录中生成一个 `repomix-output.txt` 文件，其中包含了以 AI 友好格式整理的整个代码库。



### 基本用法

打包整个代码库：

```bash
repomix
```

打包特定目录：

```bash
repomix path/to/directory
```

使用 [glob 模式](https://github.com/mrmlnc/fast-glob?tab=readme-ov-file#pattern-syntax)打包特定文件：

```bash
repomix --include "src/**/*.ts,**/*.md"
```

排除特定文件：

```bash
repomix --ignore "**/*.log,tmp/"
```

处理远程仓库：
```bash
# 使用简写格式
npx repomix --remote yamadashy/repomix

# 使用完整 URL（支持分支和特定路径）
npx repomix --remote https://github.com/yamadashy/repomix
npx repomix --remote https://github.com/yamadashy/repomix/tree/main

# 使用提交 URL
npx repomix --remote https://github.com/yamadashy/repomix/commit/836abcd7335137228ad77feb28655d85712680f1
```

初始化配置文件（`repomix.config.json`）：

```bash
repomix --init
```

生成打包文件后，您可以将其用于 Claude、ChatGPT、Gemini 等生成式 AI 工具。

#### Docker 使用方法

您也可以使用 Docker 运行 Repomix 🐳  
如果您想在隔离环境中运行 Repomix 或更偏好使用容器，这是一个很好的选择。

基本用法（当前目录）：

```bash
docker run -v .:/app -it --rm ghcr.io/yamadashy/repomix
```

打包特定目录：
```bash
docker run -v .:/app -it --rm ghcr.io/yamadashy/repomix path/to/directory
```

处理远程仓库并输出到 `output` 目录：

```bash
docker run -v ./output:/app -it --rm ghcr.io/yamadashy/repomix --remote https://github.com/yamadashy/repomix
```

### 输出格式

选择您偏好的输出格式：

```bash
# XML 格式（默认）
repomix --style xml

# Markdown 格式
repomix --style markdown

# 纯文本格式
repomix --style plain
```

### 自定义设置

创建 `repomix.config.json` 进行持久化设置：

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

### 更多示例
::: tip
💡 查看我们的 [GitHub 仓库](https://github.com/yamadashy/repomix)获取完整文档和更多示例！
:::

</div>
