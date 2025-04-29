# pip install pyspark delta-spark datasets pandas
# docker run -it -v "${PWD}/app.py:/app/app.py" -v "${PWD}/data/bronze:/app/data/bronze" -v "${PWD}/data/silver:/app/data/silver" -v "${PWD}/data/gold:/app/data/gold" bitnami/spark:latest bash

import mlflow
import mlflow.spark
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, expr
from delta import configure_spark_with_delta_pip
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.feature import VectorAssembler, StringIndexer
import pandas as pd

builder = (
    SparkSession.builder.appName("DeltaLake")
    .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension")
    .config(
        "spark.sql.catalog.spark_catalog",
        "org.apache.spark.sql.delta.catalog.DeltaCatalog",
    )
    .config("spark.jars.packages", "io.delta:delta-core_2.12:2.0.0")
    .config("spark.sql.debug.maxToStringFields", "10")
    .config("spark.sql.shuffle.partitions", "200")
    .config("spark.databricks.delta.optimizeWrite.enabled", "true")
    .config("spark.databricks.delta.autoCompact.enabled", "true")
)
spark = configure_spark_with_delta_pip(builder).getOrCreate()
mlflow.set_tracking_uri("file:/app/logs/mlruns")

print("download source data")
df = spark.read.option("header", "true").csv("spotify-tracks-dataset.csv")
df = df.drop("Unnamed: 0")

columns_to_cast = {
    "popularity": "int",
    "time_signature": "int",
    "danceability": "float",
    "instrumentalness": "float",
    "tempo": "float",
    "energy": "float",
    "key": "float",
    "loudness": "float",
    "mode": "float",
    "speechiness": "float",
    "acousticness": "float",
    "liveness": "float",
    "valence": "float",
    "duration_ms": "float",
    "explicit": "int",
}

for column_name, new_type in columns_to_cast.items():
    df = df.withColumn(column_name, col(column_name).cast(new_type))

df.show()

#  Загрузить исходные данные в формат Delta Table в папку ./data/bronze/
df.repartition(1).write.format("delta").mode("overwrite").save("./data/bronze/")
spark.sql(
    """
    OPTIMIZE delta.`/app/data/bronze`
    ZORDER BY (track_genre)
"""
)
# Загрузить данные из папки bronze
print("download delta table from ./data/bronze/")
df = spark.read.format("delta").load("./data/bronze/").filter("popularity > 0")
# Очистка данных
df = df.dropDuplicates().dropna(how="all")
numeric_cols = [
    f.name
    for f in df.schema.fields
    if f.dataType.typeName() in ["double", "int", "float", "long"]
]
string_cols = [f.name for f in df.schema.fields if f.dataType.typeName() == "string"]
for col_name in numeric_cols:
    median_value = df.approxQuantile(col_name, [0.5], 0.01)[0]
    df = df.fillna({col_name: median_value})
# df = df.fillna({col_name: "unknown" for col_name in string_cols})

# Записать обработанные данные в папку silver в формате Delta Table
df.write.format("delta").mode("overwrite").save("./data/silver/")

# Выполнить агрегации и анализ, провести оптимизацию

print("Общая информация:")
df.printSchema()
print("Описательная статистика по числовым колонкам:")
df.describe().show()

print("Количество записей по жанрам:")
print(df.groupBy("track_genre").count().orderBy("count", ascending=False).show())

print("Средняя популярность по жанрам:")
print(
    df.groupBy("track_genre")
    .avg("popularity")
    .orderBy("avg(popularity)", ascending=False)
    .show()
)


df = df.drop("track_id", "artists", "album_name", "track_name")

indexer = StringIndexer(inputCol="track_genre", outputCol="track_genre_index")
df = indexer.fit(df).transform(df)

# Записать результат в gold слой.
df.write.format("delta").mode("overwrite").save("./data/gold/")
# df = (
#     spark.read.format("delta")
#     .load("./data/gold/")
#     .filter(col("track_genre_index") == 5)
#     .count()
# )
# Задача - классификация по жанру
numeric_cols = [
    "popularity",
    "duration_ms",
    "danceability",
    "energy",
    "key",
    "loudness",
    "mode",
    "speechiness",
    "acousticness",
    "instrumentalness",
    "liveness",
    "valence",
    "tempo",
    "time_signature",
]
assembler = VectorAssembler(
    inputCols=numeric_cols, outputCol="vectorized_data", handleInvalid="skip"
)
df = assembler.transform(df)

data_train, data_test = df.randomSplit([0.75, 0.25])
# data_train = data_train.cache().repartition(5)
# data_test = data_test.cache().repartition(5)


with mlflow.start_run(run_name="LogisticRegression_with_Spark"):

    # Определяем и тренируем модель
    lr = LogisticRegression(featuresCol="vectorized_data", labelCol="track_genre_index")
    model = lr.fit(data_train)

    # Предсказания
    pred_test = model.transform(data_test)

    # Логирование модели
    mlflow.spark.log_model(model, "logistic_regression_model")

    # Логирование параметров
    mlflow.log_param("max_iter", lr.getMaxIter())
    mlflow.log_param("reg_param", lr.getRegParam())
    mlflow.log_param("elastic_net_param", lr.getElasticNetParam())

    # Логирование метрик
    training_summary = model.summary
    mlflow.log_metric("accuracy", training_summary.accuracy)
    mlflow.log_metric("f1", training_summary.fMeasureByLabel()[1])
