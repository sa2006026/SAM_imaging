# SAM Droplet Segmentation - Docker Image
FROM python:3.11-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    wget \
    curl \
    git \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libgtk-3-0 \
    && rm -rf /var/lib/apt/lists/*

# Create app directory
WORKDIR /app

# Copy requirements first for better Docker layer caching
COPY requirements_server.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip setuptools wheel
RUN pip install --no-cache-dir -r requirements_server.txt

# Install mobile_sam specifically
RUN pip install --no-cache-dir git+https://github.com/ChaoningZhang/MobileSAM.git

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p logs model static

# Set proper permissions
RUN chmod +x start_server.py

# Expose port
EXPOSE 9487

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:9487/health || exit 1

# Default command
CMD ["python", "start_server.py", "--mode", "gunicorn"] 