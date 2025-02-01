FROM python:3.9-slim

# create a user group and a user
ARG USER=standard
ARG USER_ID=1003
ARG USER_GROUP=standard
ARG USER_GROUP_ID=1003
ARG USER_HOME=/home/${USER}
RUN groupadd --gid $USER_GROUP_ID $USER_GROUP && useradd --uid $USER_ID --gid $USER_GROUP_ID -m $USER

# install miniconda
RUN apt-get update && apt-get install -y wget bzip2 curl && \
    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh && \
    bash Miniconda3-latest-Linux-x86_64.sh -b -p /opt/conda && \
    rm Miniconda3-latest-Linux-x86_64.sh && \
    apt-get clean && rm -rf /var/lib/apt/lists/*
ENV PATH="/opt/conda/bin:$PATH"

WORKDIR /app
COPY . /app

# install project dependencies
RUN conda env create -f /app/environment.yml && conda clean -a

# set the default conda environment
ENV CONDA_DEFAULT_ENV="data_processing"
# add conda env to the path
ENV PATH=/opt/conda/envs/$CONDA_DEFAULT_ENV/bin:$PATH

# change ownership of the working directory to the created user
RUN chown -R $USER:$USER_GROUP /app

# switch to the created user
USER $USER

# make the synchronization script executable
RUN chmod +x wait-for-import.sh

# wait for the neo4j import to be done and then execute the main program
ENTRYPOINT ./wait-for-import.sh "/import/import_done" && python3 main_path_finding.py