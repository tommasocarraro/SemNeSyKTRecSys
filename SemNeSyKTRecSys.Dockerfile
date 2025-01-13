FROM nvidia/cuda:11.7.1-cudnn8-runtime-ubuntu22.04

# Create a user group and a user
ARG USER=standard
ARG USER_ID=1003
ARG USER_GROUP=standard
ARG USER_GROUP_ID=1003
ARG USER_HOME=/home/${USER}

RUN groupadd --gid $USER_GROUP_ID $USER_GROUP \
    && useradd --uid $USER_ID --gid $USER_GROUP_ID -m $USER \
    && apt-get update && apt-get install -y wget bzip2 curl && \
    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh && \
    bash Miniconda3-latest-Linux-x86_64.sh -b -p /opt/conda && \
    rm Miniconda3-latest-Linux-x86_64.sh && \
    apt-get clean && rm -rf /var/lib/apt/lists/*
ENV PATH="/opt/conda/bin:$PATH"

WORKDIR /app

COPY src /app/src
COPY pretrain_source.py /app/pretrain_source.py
COPY train_target.py /app/train_target.py
COPY environment_gpu.yml /app/environment_gpu.yml
COPY data/kg_paths /app/data/kg_paths
COPY data/ratings /app/data/ratings
COPY source_models /app/source_models
COPY .env /app/.env

RUN conda env create -f /app/environment_gpu.yml && conda clean -a

ENV CONDA_DEFAULT_ENV="amazon_gpu"
ENV PATH=/opt/conda/envs/$CONDA_DEFAULT_ENV/bin:$PATH

# Change ownership of the working directory to the created user
RUN chown -R $USER:$USER_GROUP /app

# Switch to the non-root user
USER $USER

CMD ["python", "train_target.py", "--tune", "music", "movies", "--src_model_path", "./source_models/best_src_music_movies.pth"]
