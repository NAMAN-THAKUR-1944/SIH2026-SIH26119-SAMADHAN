# SAMADHAN in a container. The C++ core is compiled while the image is built; the GPU engine runs on the CPU
# unless the image is built with the CUDA build of PyTorch.
#
#   docker build -t samadhan .
#   docker run --rm samadhan                                         # self-check of every engine (~1 min)
#   docker run --rm samadhan demo --size M --device cpu              # 110k-variable refinery LP
#   docker run --rm -v "$PWD:/models" samadhan solve /models/model.mps
#
# NVIDIA GPU (Windows: Docker Desktop with the WSL 2 engine; Linux: the NVIDIA Container Toolkit):
#   docker build --build-arg TORCH_INDEX=https://download.pytorch.org/whl/cu128 -t samadhan:gpu .
#   docker run --rm --gpus all samadhan:gpu demo --size XL --tol 1e-6
FROM python:3.12-slim

ARG TORCH_INDEX=https://download.pytorch.org/whl/cpu
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /opt/samadhan
RUN pip install torch --index-url "$TORCH_INDEX" \
 && pip install "numpy>=2.0" "scipy>=1.12" "highspy>=1.7" "pytest>=8"

COPY pyproject.toml README.md LICENSE ./
COPY cpp/ cpp/
COPY samadhan/ samadhan/
# zig is only the compiler: build the core, then remove zig and its cache (~0.5 GB)
RUN pip install "ziglang>=0.13" \
 && python -c "from samadhan.core import build; print('C++ core built:', build())" \
 && pip uninstall -y ziglang && rm -rf /root/.cache
COPY tests/ tests/
COPY benchmarks/ benchmarks/
COPY results/ results/

ENTRYPOINT ["python", "-m", "samadhan"]
CMD ["verify"]
