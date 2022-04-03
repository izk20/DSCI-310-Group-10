# Group-10 Docker-Image Build
FROM rocker/rstudio

RUN apt-get update && apt-get install -y --no-install-recommends build-essential r-base python3.9 python3-pip python3-setuptools python3-dev

COPY requirements.txt requirements.txt

RUN pip3 install -r requirements.txt

RUN Rscript -e "install.packages('reticulate')"

RUN Rscript -e "install.packages('knitr', dependencies = TRUE)"

RUN Rscript -e "install.packages('bookdown')"

RUN Rscript -e "tinytex::install_tinytex()"