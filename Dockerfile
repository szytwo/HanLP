FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY api_requirements.txt .

# 阿里云镜像源
ENV PIP_INDEX_URL=https://mirrors.aliyun.com/pypi/simple
# 避免 pip 在安装过程中尝试以 root 用户身份执行某些操作时出现警告
ENV PIP_ROOT_USER_ACTION=ignore

# 升级 pip、setuptools、wheel
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

RUN pip install --no-cache-dir torch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 --index-url https://download.pytorch.org/whl/cu121 \
    && rm -rf /root/.cache/pip /tmp/*

RUN pip install --no-cache-dir hanlp \
    && pip install --no-cache-dir -r api_requirements.txt \
    && rm -rf /root/.cache/pip /tmp/*

COPY . .

EXPOSE 8120

COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh
ENTRYPOINT ["/entrypoint.sh"]
#CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8120"]
