FROM quay.io/jupyter/base-notebook:notebook-7.4.3

# Copy all files (including environment.yml) into the home directory
COPY . ${HOME}

USER root

# Install system dependencies required by dolfinx/gmsh/pyvista GUI
RUN apt-get update && apt-get install -y --no-install-recommends \
    libxft2 \
    libxrender1 \
    libxext6 \
    libgl1 \
    libglu1-mesa \
    libxi6 \
    libxinerama1 \
    libxcursor1 \
    libx11-6 \
    libxrandr2 \
    build-essential \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Ensure notebook user owns the home directory
RUN chown -R ${NB_UID} ${HOME}
USER ${NB_UID}

# Create conda environment from environment.yml
RRUN mamba env create -f .binder/environment.yml && \
    conda clean --all -f -y && \
    echo "conda activate fenicsx-env" >> ~/.bashrc

# Set the correct PATH and default environment
ENV CONDA_DEFAULT_ENV=fenicsx-env
ENV PATH=/opt/conda/envs/fenicsx-env/bin:$PATH

# Install ipykernel and register environment in Jupyter
RUN pip install ipykernel && \
    conda run -n fenicsx-env python -m ipykernel install --user \
    --name=fenicsx-env --display-name "Python (fenicsx-env)"
