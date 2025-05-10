# trunk-ignore-all(checkov/CKV_DOCKER_2)
# trunk-ignore-all(checkov/CKV_DOCKER_3)
FROM python:3.11-slim

# Set the working directory in the container
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    wget \
    build-essential \
    automake \
    libtool \
    pkg-config \
    && rm -rf /var/lib/apt/lists/*

# Install TA-Lib
RUN ARCH=$(dpkg --print-architecture) && \
    wget "https://github.com/ta-lib/ta-lib/releases/download/v0.6.4/ta-lib_0.6.4_${ARCH}.deb" && \
    dpkg -i "ta-lib_0.6.4_${ARCH}.deb" && \
    rm -rf "ta-lib_0.6.4_${ARCH}.deb"

# Copy the 'praice' directory contents into the container at /app/praice
COPY . /app/praice

# Install any needed packages specified in requirements.txt
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

# Command to run Celery (this will be overridden by docker-compose)
CMD ["celery", "-A", "praice.jobs.celery_config:app", "worker", "--loglevel=info"]
