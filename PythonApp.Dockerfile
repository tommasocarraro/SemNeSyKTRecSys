FROM continuumio/miniconda3

# create the directory for the export
RUN mkdir /export
# ensure that neo4j user will be able to write to it
RUN chmod 777 /export

# create a user group and a user
ARG USER=standard
ARG USER_ID=1003
ARG USER_GROUP=standard
ARG USER_GROUP_ID=1003
ARG USER_HOME=/home/${USER}
RUN groupadd --gid $USER_GROUP_ID $USER_GROUP && useradd --uid $USER_ID --gid $USER_GROUP_ID -m $USER

WORKDIR /app
COPY ./main_path_finding.py /app
COPY ./src/ /app/src
COPY ./data/processed/mappings /app/data/processed/mappings
COPY ./data/ratings /app/data/ratings
COPY ./environment.yml /app
COPY ./config.py /app
COPY .env /app

# install project dependencies
RUN conda env create -f /app/environment.yml && conda clean --all -y

# set the default conda environment
ENV CONDA_DEFAULT_ENV="data_processing"
# add conda env to the path
ENV PATH=/opt/conda/envs/$CONDA_DEFAULT_ENV/bin:$PATH

# change ownership of the working directory to the created user
RUN chown -R $USER:$USER_GROUP /app

# switch to the created user
USER $USER

ENTRYPOINT ["python3", "main_path_finding.py"]