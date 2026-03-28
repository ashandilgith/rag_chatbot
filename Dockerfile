# Use an official, lightweight Python image
FROM python:3.11-slim

# Set the working directory
WORKDIR /app

# Copy your requirements and install them
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all your project files into the container
COPY . .

# Cloud Run automatically assigns a port via the PORT environment variable.
# We force uvicorn to listen on port 8080, which is Cloud Run's default.
CMD ["uvicorn", "main_zoho:app", "--host", "0.0.0.0", "--port", "8080"]