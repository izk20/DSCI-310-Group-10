# Group-10 Docker Image Build
FROM rocker/rstudio:4.1.3

RUN apt-get update && apt-get install -y --no-install-recommends apt-utils build-essential r-base python3.9 python3-pip python3-setuptools

RUN ln -s /usr/bin/python3.9 /usr/bin/python

RUN python3 -m pip install group10pack

COPY requirements.txt requirements.txt

RUN pip3 install -r requirements.txt


RUN R -e "install.packages('reticulate')"

RUN R -e "install.packages('knitr', dependencies = TRUE)"

RUN R -e "install.packages('bookdown')"

RUN R -e "install.packages('rmarkdown')"
