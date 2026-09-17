# Use official lightweight Python image
FROM python:3.11-slim

# Install system dependencies required for OpenCV and GUI windows
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Set working directory inside the container
WORKDIR /app

# Copy requirements and install python packages
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project source code, data, and models
COPY . .

# Default entrypoint to run your main script
ENTRYPOINT ["python", "main.py"]