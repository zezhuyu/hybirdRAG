FROM continuumio/miniconda3:latest

WORKDIR /usr/src/llm
COPY . .

RUN conda install -y cuda -c nvidia && \
    apt update -y && \
    apt install --upgrade python3 python3-pip git -y && \
    python3 -m pip install pip && \
    pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu126 && \
    pip install -r requirements.txt --no-cache-dir --no-build-isolation && \
    pip install flash_attn==2.8.3 --no-build-isolation

EXPOSE 50051
CMD python3 server.py