---
layout: home
title: Repomix
titleTemplate: コードベースをAIフレンドリーな形式にパッケージング
aside: false
editLink: false

features:
  - icon: 🤖
    title: AI最適化
    details: コードベースをAIが理解・処理しやすい形式にフォーマット

  - icon: ⚙️
    title: Git対応
    details: .gitignoreファイルを自動的に認識し、適切なファイル除外

  - icon: 🛡️
    title: セキュリティ重視
    details: Secretlintを組み込み、機密情報の検出と除外

  - icon: 📊
    title: トークンカウント
    details: ファイルごとおよびコードベース全体のトークン数を計測し、LLMのコンテキスト制限に対応

---

<div class="cli-section">

## クイックスタート

Repomixを使用すると、コードベース全体を1ファイル（`repomix-output.txt`）にできます。

そのまま Claude に次のようなプロンプトと一緒に送ると、

```
このファイルはコードベース内のファイルを1つにまとめたものです。
コードのリファクタリングを行いたいので、まずはコードレビューをお願いします。
```

全体の内容を理解した上で、リファクタリングなどを進めることができます。

![Repomixの使用例1](/images/docs/repomix-file-usage-1.png)

具体的な内容を提案すると、それに従って良い感じのコードを生成してくれます。Claude だと Artifacts 機能で複数のファイルが出力できるため、依存関係にある複数のコードも一緒に生成できます。

![Repomixの使用例2](/images/docs/repomix-file-usage-2.png)

良いコーディング体験を！🚀

## より凝った使い方をしたい方へ

Repomix はCLIでも使えます。

ウェブ上では簡単な設定しかできませんが、より詳細な設定で実行したい場合はCLIを使うことをお勧めします。

### クイックスタート

任意のディレクトリで以下のコマンドを実行すると、 `repomix-output.txt` が生成され、それ以降の使い方は同様です。

```bash
npx repomix
```

または、グローバルにインストールして繰り返し使用することもできます。

```bash
# npmを使用してインストール
npm install -g repomix

# または、yarnを使用
yarn global add repomix

# または、Homebrewを使用（macOS/Linux）
brew install repomix

# その後、任意のプロジェクトディレクトリで実行
repomix
```


### CLIの使用方法

カレントディレクトリ全体をまとめる。

```bash
repomix
```

特定のディレクトリをまとめる。

```bash
repomix path/to/directory
```

[glob パターン](https://github.com/mrmlnc/fast-glob?tab=readme-ov-file#pattern-syntax)を使用して特定のファイルやディレクトリを指定。

```bash
repomix --include "src/**/*.ts,**/*.md"
```

特定のファイルやディレクトリを除外。

```bash
repomix --ignore "**/*.log,tmp/"
```

リモートリポジトリをまとめる。

```bash
# ショートハンド形式を使用
npx repomix --remote yamadashy/repomix

# 完全なURL（ブランチや特定のパスをサポート）
npx repomix --remote https://github.com/yamadashy/repomix
npx repomix --remote https://github.com/yamadashy/repomix/tree/main

# コミットのURLを使用
npx repomix --remote https://github.com/yamadashy/repomix/commit/836abcd7335137228ad77feb28655d85712680f1
```

設定ファイル（`repomix.config.json`）の初期化

```bash
repomix --init
```

生成されたファイルは、Claude、ChatGPT、Geminiなどの生成AIツールで使用できます。

### Docker使用方法 🐳

Dockerを使用してRepomixを実行することも可能で、分離された環境でRepomixを実行したい場合に便利です。

基本的な使用方法

```bash
docker run -v .:/app -it --rm ghcr.io/yamadashy/repomix
```

特定のディレクトリをまとめる
```bash
docker run -v .:/app -it --rm ghcr.io/yamadashy/repomix path/to/directory
```

リモートリポジトリを処理し、`output`ディレクトリに出力

```bash
docker run -v ./output:/app -it --rm ghcr.io/yamadashy/repomix --remote https://github.com/yamadashy/repomix
```

### 出力フォーマット

出力フォーマットは3種類あり、`xml`, `markdown`, `plain` を選択できます。
LLMによっては得意・不得意があるので、適切なフォーマットを選択してください。

```bash
# プレーンテキストフォーマット（デフォルト）
repomix --style plain

# XMLフォーマット
repomix --style xml

# Markdownフォーマット
repomix --style markdown
```

### カスタマイズ

フォルダごとの永続的な設定のために`repomix --init`で`repomix.config.json`を作成できます。

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

### その他の例
::: tip
💡 詳細なドキュメントと更なる例については、[GitHubリポジトリ](https://github.com/yamadashy/repomix)をご覧ください！
:::

</div>
