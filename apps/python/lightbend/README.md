# Setting up the environment

## Clone the repo

Remember to use `--recursive` to clone the BigDL part as well.

`$ git clone --recursive https://github.com/lightbend/analytics-zoo.git`


## Setup environment variables:

```bash
$ export SPARK_HOME=<location of Spark 2.2>
$ export ZOO_HOME=<project home>
$ export MASTER=local[*]
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


