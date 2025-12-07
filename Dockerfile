FROM continuumio/miniconda3

RUN apt update && apt install ffmpeg libsm6 libxext6 -y
RUN apt install gunicorn -y
RUN conda create -n det python=3.7 -y
ENV PATH=/opt/conda/envs/det/bin:$PATH

RUN python -m pip install torch==1.10.0 torchvision==0.11.1

RUN python -m pip install detectron2 -f \
  https://dl.fbaipublicfiles.com/detectron2/wheels/cpu/torch1.10/index.html

RUN pip install opencv-python numpy scipy laspy tqdm matplotlib flask

COPY . .

EXPOSE 8080

CMD [ "gunicorn","--timeout", "12000", "-w", "1", "-b", "0.0.0.0:8080", "app:app" ]