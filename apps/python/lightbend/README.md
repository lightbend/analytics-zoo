# Setting up the environment

## Clone the repo

Remember to use `--recursive` to clone the BigDL part as well.

`$ git clone --recursive https://github.com/lightbend/analytics-zoo.git`

## Build analytics-zoo

```
$ pwd
<analytics zoo installation folder>
$ ./make-dist.sh
```

## Setup environment variables:

```bash
$ export SPARK_HOME=<location of Spark 2.2>
$ export ZOO_HOME=<project home>
$ export MASTER=local[*]
$ export ANALYTICS_ZOO_HOME=<project home>/dist
$ export ZOO_JAR=<project home>/dist/lib/analytics-zoo-0.2.0-SNAPSHOT-jar-with-dependencies.jar
$ export ZOO_CONF=<project home>/dist/conf/spark-analytics-zoo.conf
$ export ZOO_PY_ZIP=<project home>/dist/lib/analytics-zoo-0.2.0-SNAPSHOT-spark-2.1.0-dist.zip

```

## Ensure the following are installed

1. Python3
2. Anaconda
3. Spark 2.2

## Patch spark-env.sh

Enter the following line in `$SPARK_HOME/conf/spark-env.sh`:
`PYSPARK_PYTHON=python3`

## Run Jupyter

```
$ cd analytics-zoo/apps/python/lightbend
$ .../analytics-zoo/scripts/jupyter-with-zoo.sh --master ${MASTER} \
                                                --driver-cores 2  \
                                                --driver-memory 8g  \
                                                --total-executor-cores 2  \
                                                --executor-cores 4  \
                                                --executor-memory 4g
```

## Open notebook

In the browser navigate to `http://localhost:8888` and open the notebook

## Convert notebook to Python script

`$ jupyter nbconvert --to script dnn_anomaly_bigdl.ipynb`

> Ensure `nbconvert` is installed. Or else install using `conda install nbconvert`

Once the notebook is converted to the Python script, we need to remove the `ipython` specific stuff from the generated python script. This is a manual process.

Here's an example ..

Remove / comment out the following line from the generated python script:

```
get_ipython().run_cell_magic('time', '', '# Boot training process\ntrained_model = optimizer.optimize()\nprint("Optimization Done.")')
```

and replace with the following:

```
trained_model = optimizer.optimize()
print("Optimization Done.")
```

Also remove all plot statements from the python script.

## Prepare docker image with the script
First build an image for IntelBigDL using [dockerfile](,/docker/Dockerfile)
````
docker build \
    --build-arg ANALYTICS_ZOO_VERSION=0.3.0 \
    --build-arg BIGDL_VERSION=0.6.0 \
    --build-arg SPARK_VERSION=2.3.1 \
    --rm -t lightbend/analytics-zoo:0.3.0-bigdl_0.6.0-spark_2.3.1 .
````
At the moment, INtel only supports Spark 2.3.1
Now use this image as a starting point for building Anomaly detection image
```
$ pwd
<project home>/apps/python/lightbend
$ docker build --rm -t lightbend/amomaly-detection-intel-bigdl:2.0 .
```

## Run docker image

When running the docker image we need to mount the host folder (make sure to use the full, not relative name) containing `data/CPU_examples.csv` to the container folder `/opt/work/data`.

```
$ docker run -it --rm -v <data folder>:/opt/work/data lightbend/analytics-zoo:0.1.0-spark-2.2.0 bash
root@3d7a2d664c77:/opt/work# pwd
/opt/work
root@3d7a2d664c77:/opt/work# ls
analytics-zoo-0.3.0                                       data                  download-analytics-zoo.sh  spark-2.3.1
analytics-zoo-bigdl_0.6.0-spark_2.3.1-0.3.0-dist-all.zip  dnn_anomaly_bigdl.py  get-pip.py                 start-notebook.sh
root@3d7a2d664c77:/opt/work# analytics-zoo-0.3.0/bin/spark-submit-with-zoo.sh dnn_anomaly_bigdl.py
```

The above command will run the training of the model using the data files in `data` folder and generate a Tensorflow model in `/tmp/model.pb`. This can be used for scoring.







