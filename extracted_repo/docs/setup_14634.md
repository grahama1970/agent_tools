# Configuración del entorno de desarrollo

## Requisitos previos

- Node.js ≥ 18.0.0
- Git
- npm

## Desarrollo local

```bash
# Clonar el repositorio
git clone https://github.com/yamadashy/repomix.git
cd repomix

# Instalar dependencias
npm install

# Ejecutar CLI
npm run repomix
```

## Desarrollo con Docker

```bash
# Construir la imagen
docker build -t repomix .

# Ejecutar el contenedor
docker run -v ./:/app -it --rm repomix
```

## Estructura del proyecto

```
src/
├── cli/          # Implementación de la CLI
├── config/       # Manejo de la configuración
├── core/         # Funcionalidad principal
└── shared/       # Utilidades compartidas
```

## Pruebas

```bash
# Ejecutar pruebas
npm run test

# Cobertura de pruebas
npm run test-coverage

# Linting
npm run lint-biome
npm run lint-ts
npm run lint-secretlint
```

## Proceso de lanzamiento

1. Actualizar la versión
```bash
npm version patch  # o minor/major
```

2. Ejecutar pruebas y construir
```bash
npm run test-coverage
npm run build
```

3. Publicar
```bash
npm publish
