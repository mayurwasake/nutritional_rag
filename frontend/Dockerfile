FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install python dependencies efficiently
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy Streamlit config & application files
COPY . .

# Expose Streamlit default port
EXPOSE 8501

# Add standard healthcheck for Docker-compose
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

# Using standard Streamlit deployment hook
ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0"]