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

# Create a docker image resource for FastAPI

# resource "docker_image" "my_fastapi_res" {
#   name = "my_fastapi"
#   build {
#     path = "."
#     tag  = ["my_fastapi:develop"]
#     build_arg = {
#       name : "my_fastapi"
#     }
#     label = {
#       author : "vbo"
#     }
#   }
# }

# Create a Docker image resource for MLflow
resource "docker_image" "mlflow_res" {
  name = "mlflow_server"
  build {
    path = "."
    tag  = ["mlflow_server:latest"]
  }
}

# Create a docker container resource

# resource "docker_container" "fastapi" {
#   name    = "fastapi"
#   image   = docker_image.my_fastapi_res.image_id

#   ports {
#     external = 8002
#     internal = 8000
#   }
# }

# Create a Docker container resource for MLflow server
resource "docker_container" "mlflow" {
  name  = "mlflow"
  image = docker_image.mlflow_res.image_id

  ports {
    external = 5000
    internal = 5000
  }
}