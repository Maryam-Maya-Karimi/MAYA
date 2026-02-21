# Use a Python base image (common for LangGraph)
FROM mcr.microsoft.com/devcontainers/python:3.11

# Install system tools he likely uses (git, curl, etc.)
RUN apt-get update && apt-get install -y \
    curl \
    git \
    build-essential
    fluidsynth \
    fluid-soundfont-gm \
    lame \
    libasound2-dev

# Set the working directory inside the container
WORKDIR /workspace