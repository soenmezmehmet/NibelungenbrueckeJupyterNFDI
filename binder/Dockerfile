FROM quay.io/jupyter/scipy-notebook:python-3.10

COPY . ${HOME}
USER root
RUN chown -R ${NB_UID} ${HOME}

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

USER ${NB_UID}

RUN mamba env create -f binder/environment.yml && \
    conda clean --all -f -y && \
    echo "conda activate fenicsx-env" >> ~/.bashrc

ENV CONDA_DEFAULT_ENV=fenicsx-env
ENV PATH=/opt/conda/envs/fenicsx-env/bin:$PATH

RUN conda run -n fenicsx-env python -m ipykernel install --user --name=fenicsx-env --display-name "Python (fenicsx-env)"
