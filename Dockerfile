FROM python:3.11

# Install system dependencies including ffmpeg
RUN apt-get update && apt-get install -y \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY server.py .

# Create uploads directory
RUN mkdir -p uploads

# Expose port (Railway will override this with PORT env var)
EXPOSE 8000

# Start command
CMD ["python", "server.py"]
