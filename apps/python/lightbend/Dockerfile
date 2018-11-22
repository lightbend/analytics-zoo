from lightbend/analytics-zoo:0.3.0-bigdl_0.6.0-spark_2.3.1

WORKDIR /opt/work

RUN apt-get install -y python-tk

# generated Python script from Jupyter Notebook
ADD ./dnn_anomaly_bigdl.py /opt/work
ADD ./wait-and-start-training.sh /opt/work/analytics-zoo-0.3.0/bin

# data folder : to be mapped as volume during run
RUN mkdir data
