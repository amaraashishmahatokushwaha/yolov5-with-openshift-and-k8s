FROM python:3.9

WORKDIR /app

RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    make \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgl1-mesa-glx \
    v4l-utils \
    ffmpeg \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# Install torch/vision from PyTorch repo
RUN pip install torch==1.10.0+cpu torchvision==0.11.1+cpu \
    -f https://download.pytorch.org/whl/torch_stable.html

# Install rest of the requirements
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt
    
COPY yolov5s.pt /app/yolov5s.pt
COPY . .


RUN mkdir -p templates

RUN if [ ! -f templates/index.html ]; then \
    echo "<html><head><title>Camera Stream</title></head><body><h1>Camera Stream</h1><img src='/video_feed'></body></html>" > templates/index.html; \
    fi

ENV OPENCV_VIDEOIO_PRIORITY_MSMF=0
ENV OPENCV_VIDEOIO_DEBUG=1
ENV VIDEO_SOURCE=0

EXPOSE 5000

CMD ["python3", "finalrun.py"]
