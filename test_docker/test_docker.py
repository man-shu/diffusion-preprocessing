import pytest
import docker
import os
from dockerfile_parse import DockerfileParser
import subprocess

@pytest.fixture
def docker_client():
    return docker.from_env()

@pytest.fixture
def test_docker_container(docker_client):
    container = docker_client.containers.run("niflow_diffusion_package", detach = True)
    yield container
    container.stop()
    container.remove()

def test_dockerfile_exists():
    assert os.path.exists("Dockerfile"), "Dockerfile not found"

def test_dockerfile_valid_syntax():
    dockerfile_path = "Dockerfile"  # Specify the path to your Dockerfile

    # Build the Docker image with the `--quiet` option to suppress output
    build_command = ['docker', 'build', '-t', 'test-image', '-f', dockerfile_path, '.']

    try:
        # Execute the build command
        result = subprocess.run(build_command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        output = result.stdout.strip() if result.stdout else result.stderr.strip()
        return result, output
    except subprocess.CalledProcessError as e:
        # If the build command fails, the Dockerfile has invalid syntax
        pytest.fail(f"Dockerfile has invalid syntax:\n{e.stderr.decode()}")

if __name__ == "__main__":

    test_dockerfile_exists()
    result, output = test_dockerfile_valid_syntax()

    print(f"Build Result: {result.returncode}")
    print(f"Build Output:\n{output}")

