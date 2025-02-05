# Aksara Classification

This project is a machine learning application for classifying Aksara characters.

## Prerequisites

- Docker
- Docker Compose

## Getting Started

### Clone the Repository

```sh
git clone https://github.com/yourusername/aksara-classification.git
cd aksara-classification
```

### Build and Run the Application

1. **Build the Docker images and start the containers:**

   ```sh
   docker compose up --build -d
   ```

2. **Access the application:**

   The application will be available at [http://localhost:8000](http://localhost:8000).

### Stopping the Application

To stop the application, run:

```sh
docker compose down
```

### Rebuilding the Application

If you make changes to the code and need to rebuild the application, run:

```sh
docker compose down && docker compose up --build -d
```

## Development

For development purposes, you can use the following commands defined in the [`Makefile`]:

- **Start the application in development mode:**

  ```sh
  make dev
  ```

- **Format the code:**

  ```sh
  make format
  ```

## License

This project is licensed under the MIT License - see the [`LICENSE`]
