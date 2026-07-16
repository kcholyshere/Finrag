FROM python:3.13-slim

WORKDIR /app

COPY requirements.txt .
# torch is pinned in requirements.txt with no build tag, so a plain PyPI
# install pulls the CUDA/nvidia wheel stack (multi-GB) even though this image
# only ever runs on CPU. Install the matching CPU-only wheels first so the
# later requirements.txt install finds torch already satisfied and skips it.
# torchvision must be pinned alongside it from the same CPU index - a plain
# PyPI torchvision links against the CUDA-enabled torch build's ABI and fails
# at import with "operator torchvision::nms does not exist" against the
# CPU-only torch.
RUN pip install --no-cache-dir torch==2.11.0 torchvision==0.26.0 --extra-index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r requirements.txt

COPY src/ src/

ENV PYTHONPATH=/app

EXPOSE 8501

CMD ["streamlit", "run", "src/ui/app.py", "--server.address=0.0.0.0", "--server.port=8501"]
