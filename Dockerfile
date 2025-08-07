FROM r8.im/cog-base:cuda12.1-python3.10-torch2.1.0

RUN pip install --no-cache-dir \
    diffusers==0.24.0 \
    transformers==4.37.2 \
    opencv-python==4.9.0.80 \
    accelerate==0.27.2 \
    xformers==0.0.23 \
    Pillow==10.3.0 \
    controlnet-aux==0.0.7

COPY . /src
WORKDIR /src
