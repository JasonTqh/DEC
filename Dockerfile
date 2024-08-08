FROM python:3.8

# Setting up the working directory
WORKDIR /app

# comcast
RUN apt-get update && \
    apt-get install -y git curl && \
    curl -Lo /usr/local/bin/comcast https://raw.githubusercontent.com/tylertreat/Comcast/master/comcast && \
    chmod +x /usr/local/bin/comcast

# 下载CIFAR-10数据集
ADD https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz /app/data/
RUN tar -xzvf /app/data/cifar-10-python.tar.gz -C /app/data/ && rm /app/data/cifar-10-python.tar.gz
ENV KERAS_DATASETS_DIR=/app/data/cifar-10-batches-py

# Installing Python dependencies
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy all files to the working directory
COPY . .

# Running server code by default
CMD ["python"]
