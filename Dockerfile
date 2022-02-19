FROM continuumio/anaconda3

RUN conda install --y \
    pandas \
    scikit-learn \
    seaborn \
    matplotlib 
RUN pip install mlxtend
