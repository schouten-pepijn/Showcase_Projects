import json
import yaml
from pyspark.sql import SparkSession
import pyspark.sql.functions as F
from pyspark.sql.window import Window
from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.regression import RandomForestRegressor
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
from pyspark.ml.evaluation import RegressionEvaluator


# import config file
with open('config_file.yaml', 'r') as f:
    config = yaml.safe_load(f)

def create_features(**kwargs):   
    # pull the data 
    data = kwargs['ti'].xcom_pull(
        task_ids='api_taskgroup.convert_data_task'
    )
    json_data = json.loads(data)
    # create spark session
    spark = (
        SparkSession
        .builder
        .appName("createMLFeatures")
        .getOrCreate()
    )
    # read the json
    rdd = spark.sparkContext.parallelize([json_data])
    df = spark.read.json(rdd)
    # select relevant columns
    df = df.select(
        F.col("date"),
        F.col("close").alias("price"))
    # calculate the log returns and drop nan
    lag_window = Window.orderBy(F.col("date"))
    df = (
        df
        .withColumn(
            "log_returns",
            F.log(df["price"] / F.lag(df["price"])
                  .over(lag_window))
        )
    )
    df = df.dropna(subset=["log_returns"])
    # calculate the volatility over 10 days
    volatility_window = Window.orderBy(F.col("date")) \
                            .rowsBetween(-10, 0)
    df = (
        df
        .withColumn(
            "volatility",
            F.stddev(df["log_returns"])
                .over(volatility_window)
        )
    )
    df = df.dropna(subset=["volatility"])
    # save features to csv
    df.coalesce(1). \
        write.mode("overwrite") \
            .csv(config['save_path_features'], header=True)


def train_eval_model(**kwargs):
    # create spark session
    spark = (
        SparkSession
        .builder
        .appName("createMLFeatures")
        .getOrCreate()
    )
    # read the features csv
    df = spark.read.csv(
        config['save_path_features'],
        header=True,
        inferSchema=True
    )
    # create vector assembler
    assembler = VectorAssembler(
        inputCols=config['rf_features'],
        outputCol="features"
    )
    df = assembler.transform(df)
    # train test split
    train_data, test_data = df.randomSplit([0.8, 0.2])
    # create scaler
    scaler = StandardScaler(
        inputCol="features",
        outputCol="scaled_features"
    )
    # create random forrest regressor
    rf = RandomForestRegressor(
        featuresCol="features",
        labelCol="volatility"
    )
    # create evaluator
    evaluator = RegressionEvaluator(
        labelCol="volatility",
        predictionCol="prediction",
        metricName="rmse"
    )
    # define the pipeline
    pipeline = Pipeline(stages=[scaler, rf])
    # parameter grid for grid search
    param_grid = (
        ParamGridBuilder()
        .addGrid(rf.numTrees, [10, 20, 50])
        .addGrid(rf.maxDepth, [5, 10, 15])
        .addGrid(rf.maxBins, [16, 32, 64])
        .build()
    )
    # cross validation 
    cv = CrossValidator(
        estimator=pipeline,
        estimatorParamMaps=param_grid,
        evaluator=evaluator,
        numFolds=5,
    )
    cv_model = cv.fit(train_data)
    best_model = cv_model.bestModel
    # make predictions and rmse
    predictions = best_model.transform(test_data)
    rmse = evaluator.evaluate(predictions)
    print(f"\nRMSE on test data: {rmse}\n")
    # convert to json
    predictions_json = (
        predictions 
        .select("date", "volatility", "prediction") 
        .toJSON() 
        .collect()
    )
    # push predictions to xcom
    return predictions_json
