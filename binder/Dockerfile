FROM quay.io/jupyter/base-notebook:notebook-7.4.3

COPY . ${HOME}
USER root
RUN chown -R ${NB_UID} ${HOME}

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
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*


USER ${NB_UID}

RUN mamba env create -f binder/environment.yml && \
    conda clean --all -f -y && \
    echo "conda activate fenicsx-env" >> ~/.bashrc

# Set the correct PATH and default environment
ENV CONDA_DEFAULT_ENV=fenicsx-env
ENV PATH=/opt/conda/envs/fenicsx-env/bin:$PATH

# Ensure Jupyter picks up correct Python/kernel
RUN pip install ipykernel
RUN conda run -n fenicsx-env python -m ipykernel install --user --name=fenicsx-env --display-name "Python (fenicsx-env)"

