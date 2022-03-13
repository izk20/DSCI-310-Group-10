# Group-10 Docker-Image Build
FROM continuumio/anaconda3

RUN conda install --y \
    pandas=1.3.4 \
    scikit-learn=0.24.2 \
    seaborn=0.11.2 \
    matplotlib=3.4.3 \
    pytest

