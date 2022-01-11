# 1) choose base container
# generally use the most recent tag

# base notebook, contains Jupyter and relevant tools

ARG BASE_CONTAINER=ucsdets/datahub-base-notebook:2021.2-stable

# data science notebook
# https://hub.docker.com/repository/docker/ucsdets/datascience-notebook/tags
# ARG BASE_CONTAINER=ucsdets/datascience-notebook:2021.2-stable

# scipy/machine learning (tensorflow, pytorch)
# https://hub.docker.com/repository/docker/ucsdets/scipy-ml-notebook/tags
# ARG BASE_CONTAINER=ucsdets/scipy-ml-notebook:2021.3-42158c8
FROM $BASE_CONTAINER

#LABEL maintainer="UC San Diego ITS/ETS <ets-consult@ucsd.edu>"

# 2) change to root to install packages
USER root


#RUN apt-get -y install traceroute
RUN apt-get update
#RUN apt install python3-pip
#RUN apt-get -y install nmap
#RUN apt-get -y install aria2
RUN echo 'jupyter notebook "$@"' > /run_jupyter.sh && chmod 755 /run_jupyter.sh
# 3) install packages using notebook user
USER jovyan
#RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir features
RUN pip install --no-cache-dir pandas
#RUN pip install --no-cache-dir sys
#RUN pip install --no-cache-dir json
#RUN pip install --no-cache-dir os
RUN pip install --no-cache-dir dimcli
RUN pip install --no-cache-dir etl
RUN pip install --no-cache-dir env_setup
RUN pip install --no-cache-dir requests
#RUN pip install --no-cache-dir csv
RUN pip install --no-cache-dir numpy
#RUN conda install -y re
RUN pip install --no-cache-dir nltk
RUN pip install --no-cache-dir gensim
RUN pip install --no-cache-dir spacy
RUN pip install --no-cache-dir sklearn
#RUN pip install --no-cache-dir sklearn.decomposition
#RUN pip install --no-cache-dir sklearn.feature_extraction.text
#RUN pip install --no-cache-dir sklearn.model_selection
#RUN pip install --no-cache-dir pprint
RUN pip install --no-cache-dir pyLDAvis
#RUN pip install --no-cache-dir pyLDAvis.sklearn
#RUN pip install --no-cache-dir matplotlib.pyplot
RUN pip install --no-cache-dir gensim
#RUN pip install --no-cache-dir gensim.utils
#RUN pip install --no-cache-dir gensim.parsing.preprocessing
#RUN pip install --no-cache-dir nltk.stem
#RUN pip install --no-cache-dir nltk.stem.porter
#RUN pip install --no-cache-dir pickle
RUN pip install --no-cache-dir dash
RUN pip install --no-cache-dir dash_core_components
RUN pip install --no-cache-dir dash_html_components
RUN pip install --no-cache-dir dash_bootstrap_components
#RUN pip install --no-cache-dir plotly.graph_objects
RUN pip install --no-cache-dir plotly
RUN pip install --no-cache-dir IPython
#RUN pip install --no-cache-dir dash.dependencies

#RUN conda install -y scikit-learn
#RUN conda install -y geopandas
#RUN pip install --no-cache-dir babypandas
#RUN pip install --no-cache-dir networkx scipy

COPY src /src
WORKDIR /src

# Override command to disable running jupyter notebook at launch
CMD ["python","run.py"]
