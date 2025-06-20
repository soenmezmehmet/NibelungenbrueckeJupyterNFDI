FROM quay.io/jupyter/base-notebook:notebook-7.4.3

# Copy code into the notebook user's home directory
COPY . ${HOME}

# Change ownership to the notebook user
USER root
RUN chown -R ${NB_UID}:${NB_GID} ${HOME}

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Switch back to notebook user
USER ${NB_UID}

# Create conda environment from environment.yml
RUN mamba env create -f environment.yml && \
    conda clean --all -f -y && \
    echo "conda activate demonstrator" >> ~/.bashrc

# Set the correct environment variables
ENV CONDA_DEFAULT_ENV=demonstrator
ENV PATH=/opt/conda/envs/demonstrator/bin:$PATH

# Ensure ipykernel is available and Jupyter recognizes the new environment
RUN /opt/conda/envs/demonstrator/bin/python -m ipykernel install --user \
    --name=demonstrator \
    --display-name "Python (demonstrator)"