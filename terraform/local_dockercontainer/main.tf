terraform {
  required_providers {
    # We recommend pinning to the specific version of the Docker Provider you're using
    # since new versions are released frequently
    docker = {
      source  = "kreuzwerker/docker"
      version = "2.23.1"
    }
  }
}

# Configure the docker provider
provider "docker" {
}

# Create a Docker image resource for MLflow
resource "docker_image" "mlflow_res" {
  name = "mlflow_server"
  build {
    path = "."
    tag  = ["mlflow_server:latest"]
  }
}

# Create a Docker container resource for MLflow server
resource "docker_container" "mlflow" {
  name  = "mlflow"
  image = docker_image.mlflow_res.image_id

  ports {
    external = 5000
    internal = 5000
  }
}