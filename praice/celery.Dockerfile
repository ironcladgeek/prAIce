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
ENV TALIB_VERSION=0.4.0
RUN wget https://github.com/reza-zereh/prAIce/releases/download/v0.0.0/ta-lib-${TALIB_VERSION}-src.tar.gz \
    && tar -xzf ta-lib-${TALIB_VERSION}-src.tar.gz \
    && cd ta-lib \
    && autoreconf -vif \
    && ./configure --prefix=/usr \
    && make \
    && make install \
    && cd .. \
    && rm -rf ta-lib-${TALIB_VERSION}-src.tar.gz ta-lib

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
