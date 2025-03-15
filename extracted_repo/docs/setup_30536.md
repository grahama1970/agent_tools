# Configuração de Desenvolvimento

## Pré-requisitos

- Node.js ≥ 18.0.0
- Git
- npm

## Desenvolvimento Local

```bash
# Clonar repositório
git clone https://github.com/yamadashy/repomix.git
cd repomix

# Instalar dependências
npm install

# Executar CLI
npm run repomix
```

## Desenvolvimento com Docker

```bash
# Construir imagem
docker build -t repomix .

# Executar container
docker run -v ./:/app -it --rm repomix
```

## Estrutura do Projeto

```
src/
├── cli/          # Implementação da CLI
├── config/       # Manipulação de configuração
├── core/         # Funcionalidade principal
└── shared/       # Utilitários compartilhados
```

## Testando

```bash
# Executar testes
npm run test

# Cobertura de teste
npm run test-coverage

# Linting
npm run lint-biome
npm run lint-ts
npm run lint-secretlint
```

## Processo de Release

1. Atualizar versão
```bash
npm version patch  # ou minor/major
```

2. Executar testes e construir
```bash
npm run test-coverage
npm run build
```

3. Publicar
```bash
npm publish
