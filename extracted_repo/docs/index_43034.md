---
layout: home
title: Repomix
titleTemplate: Compacte seu código-fonte em formatos amigáveis para IA
aside: false
editLink: false

features:
  - icon: 🤖
    title: Otimizado para IA
    details: Formata seu código-fonte de uma maneira fácil para a IA entender e processar.

  - icon: ⚙️
    title: Consciente do Git
    details: Respeita automaticamente seus arquivos .gitignore.

  - icon: 🛡️
    title: Focado na Segurança
    details: Incorpora o Secretlint para verificações de segurança robustas para detectar e prevenir a inclusão de informações confidenciais.

  - icon: 📊
    title: Contagem de Tokens
    details: Fornece contagens de tokens para cada arquivo e para todo o repositório, útil para limites de contexto de LLM.

---

<div class="cli-section">

## Início Rápido

Depois de gerar um arquivo compactado (`repomix-output.txt`) usando o Repomix, você pode enviá-lo para um assistente de IA com um prompt como:

```
Este arquivo contém todos os arquivos do repositório combinados em um.
Eu quero refatorar o código, então, por favor, revise-o primeiro.
```

A IA analisará todo o seu código-fonte e fornecerá insights abrangentes:

![Repomix File Usage 1](/images/docs/repomix-file-usage-1.png)

Ao discutir mudanças específicas, a IA pode ajudar a gerar código. Com recursos como o Artifacts do Claude, você pode até receber vários arquivos interdependentes:

![Repomix File Usage 2](/images/docs/repomix-file-usage-2.png)

Feliz codificação! 🚀

## Guia do Usuário Avançado

Para usuários avançados que precisam de mais controle, o Repomix oferece extensas opções de personalização através de sua interface CLI.

### Início Rápido

Você pode experimentar o Repomix instantaneamente no diretório do seu projeto sem instalação:

```bash
npx repomix
```

Ou instale globalmente para uso repetido:

```bash
# Instalar usando npm
npm install -g repomix

# Alternativamente usando yarn
yarn global add repomix

# Alternativamente usando Homebrew (macOS/Linux)
brew install repomix

# Então execute em qualquer diretório de projeto
repomix
```

É isso! O Repomix irá gerar um arquivo `repomix-output.txt` no seu diretório atual, contendo todo o seu repositório em um formato amigável para IA.

### Uso

Para compactar todo o seu repositório:

```bash
repomix
```

Para compactar um diretório específico:

```bash
repomix path/to/directory
```

Para compactar arquivos ou diretórios específicos usando [glob patterns](https://github.com/mrmlnc/fast-glob?tab=readme-ov-file#pattern-syntax):

```bash
repomix --include "src/**/*.ts,**/*.md"
```

Para excluir arquivos ou diretórios específicos:

```bash
repomix --ignore "**/*.log,tmp/"
```

Para compactar um repositório remoto:
```bash
# Usando formato abreviado
npx repomix --remote yamadashy/repomix

# Usando URL completa (suporta branches e caminhos específicos)
npx repomix --remote https://github.com/yamadashy/repomix
npx repomix --remote https://github.com/yamadashy/repomix/tree/main

# Usando URL do commit
npx repomix --remote https://github.com/yamadashy/repomix/commit/836abcd7335137228ad77feb28655d85712680f1
```

Para inicializar um novo arquivo de configuração (`repomix.config.json`):

```bash
repomix --init
```

Depois de gerar o arquivo compactado, você pode usá-lo com ferramentas de IA Generativa como Claude, ChatGPT e Gemini.

#### Uso do Docker

Você também pode executar o Repomix usando o Docker 🐳
Isso é útil se você quiser executar o Repomix em um ambiente isolado ou preferir usar contêineres.

Uso básico (diretório atual):

```bash
docker run -v .:/app -it --rm ghcr.io/yamadashy/repomix
```

Para compactar um diretório específico:
```bash
docker run -v .:/app -it --rm ghcr.io/yamadashy/repomix path/to/directory
```

Processar um repositório remoto e enviar para um diretório `output`:

```bash
docker run -v ./output:/app -it --rm ghcr.io/yamadashy/repomix --remote https://github.com/yamadashy/repomix
```

### Formatos de Saída

Escolha seu formato de saída preferido:

```bash
# Formato XML (padrão)
repomix --style xml

# Formato Markdown
repomix --style markdown

# Formato de texto simples
repomix --style plain
```

### Customização

Crie um `repomix.config.json` para configurações persistentes:

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

### Mais Exemplos
::: tip
💡 Confira nosso [repositório GitHub](https://github.com/yamadashy/repomix) para documentação completa e mais exemplos!
:::

</div>
