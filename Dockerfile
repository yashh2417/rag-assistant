FROM python:3.10-slim

# Avoid creating .pyc files
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python

# Set working directory inside container
WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY app ./app

# Expose FastAPI on port 8000
EXPOSE 8000

# Launch app with uvicorn (main.py inside app/, so app.main:app)
CMD ["uvicorn", "app.app:app", "--host", "0.0.0.0", "--port", "8000"]




