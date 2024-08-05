FROM python:3.8

# Setting up the working directory
WORKDIR /app

# Installing Python dependencies
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy all files to the working directory
COPY . .

# Running server code by default
CMD ["python"]
