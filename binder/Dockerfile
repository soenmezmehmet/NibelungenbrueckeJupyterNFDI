FROM quay.io/jupyter/base-notebook:notebook-7.4.3

COPY . ${HOME}
USER root
RUN chown -R ${NB_UID} ${HOME}

# Install required system packages using apt (base-notebook is Debian-based)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Switch back to the default user
USER ${NB_UID}

RUN mamba env create -f binder/environment.yml && \
    conda clean --all -f -y && \
    echo "conda activate demonstrator" >> ~/.bashrc

# Set default environment
ENV CONDA_DEFAULT_ENV=demonstrator
ENV PATH=/opt/conda/envs/demonstrator/bin:$PATH

# Ensure Jupyter picks up correct Python/kernel
RUN pip install ipykernel
RUN conda run -n demonstrator python -m ipykernel install --user --name=demonstrator --display-name "Python (demonstrator)"