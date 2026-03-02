FROM python:3.11-bookworm

# Avoid prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies (includes common utilities out of the box)
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    curl \
    vim \
    iputils-ping \
    net-tools \
    procps \
    htop \
    wget \
    less \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /workspace

# Copy and install Python dependencies first (for layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install debugpy for VS Code debugging
RUN pip install --no-cache-dir debugpy

# Install the project in editable mode later via postCreateCommand
