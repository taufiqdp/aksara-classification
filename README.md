# Aksara Classification

This project is a machine learning application for classifying Aksara characters.

## Prerequisites

- Docker
- Docker Compose
- Python 3.8+ (for local development)

## Getting Started

### Clone the Repository

```sh
git clone https://github.com/taufiqdp/aksara-classification.git
cd aksara-classification
```

### Development Setup

1. **Install development dependencies:**

   ```sh
   pip install -r requirements-dev.txt
   ```

2. **Run the application locally:**

   ```sh
   make dev
   ```

   The development server will be available at [http://localhost:8000](http://localhost:8000) with hot-reload enabled.

### Docker Setup

1. **Development environment:**

   ```sh
   make dev-docker
   ```

2. **Production environment:**

   ```sh
   make prod
   ```

The application will be available at [http://localhost:8000](http://localhost:8000).

### Docker Commands

- **Stop containers:**

  ```sh
  make dev-stop    # Development
  make prod-stop   # Production
  ```

- **Rebuild containers:**
  ```sh
  make dev-rebuild    # Development
  make prod-rebuild   # Production
  ```

## Development Tools

This project includes several tools to maintain code quality:

- **Format code:**

  ```sh
  make format    # Runs black and isort
  ```

- **Lint code:**

  ```sh
  make lint      # Runs flake8 and mypy
  ```

- **Run tests:**

  ```sh
  make test      # Runs pytest
  ```

- **Clean project:**
  ```sh
  make clean     # Removes cache and build artifacts
  ```

For a complete list of available commands:

```sh
make help
```

## License

This project is licensed under the MIT License - see the [`LICENSE`](https://github.com/taufiqdp/aksara-classification/blob/main/LICENSE)
