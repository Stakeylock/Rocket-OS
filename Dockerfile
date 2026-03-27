FROM ubuntu:22.04

# Prevent interactive prompts during apt installations
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3-pip \
    python3-dev \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy the requirements and install them
# We install exactly what is needed for the aerospace GNC models (including cvxpy)
COPY pyproject.toml requirements-lock.txt* /app/
RUN pip3 install --no-cache-dir \
    numpy \
    scipy \
    pandas \
    seaborn \
    cvxpy \
    tqdm

# Copy the rest of the repository
COPY . /app

# Install the package locally
RUN pip3 install -e .

# Set default command to run all benchmarks
CMD ["python3", "-m", "unittest", "discover"]