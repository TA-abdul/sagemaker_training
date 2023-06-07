# mods
import os
import argparse
import pyspark
from pyspark.sql import SparkSession
from pyspark.ml import Pipeline
from pyspark.sql.types import DoubleType, StringType, StructField, StructType
from pyspark.ml.feature import StringIndexer, OneHotEncoder, StandardScaler, VectorAssembler

# create app
spark = SparkSession.builder.appName('spark-python-sagemaker-processing').getOrCreate()

# schema
schema = StructType([StructField('sex', StringType(), True), 
                     StructField('length', DoubleType(), True),
                     StructField('diameter', DoubleType(), True),
                     StructField('height', DoubleType(), True),
                     StructField('whole_weight', DoubleType(), True),
                     StructField('shucked_weight', DoubleType(), True),
                     StructField('viscera_weight', DoubleType(), True), 
                     StructField('shell_weight', DoubleType(), True), 
                     StructField('rings', DoubleType(), True)])

def main():
    
    #args
    parser = argparse.ArgumentParser()
    parser.add_argument('--s3-input-bucket', type=str)
    parser.add_argument('--s3-input-prefix', type=str)
    parser.add_argument('--train-split-size', type=float)
    parser.add_argument('--test-split-size', type=float)
    parser.add_argument('--s3-output-bucket', type=str)
    parser.add_argument('--s3-output-prefix', type=str)
    parser.add_argument('--repartition-num', type=int)
    args = parser.parse_args()
    
    # read dataset
    df = spark.read.csv(('s3://' + os.path.join(args.s3_input_bucket, args.s3_input_prefix, 'abalone.csv')),
                        header=False,
                        schema=schema)
    
    # split dataset
    (train_df, test_df) = df.randomSplit([args.train_split_size, args.test_split_size], seed=0)
    
    # pipeline
    inx = StringIndexer(inputCol='sex', outputCol='sex_index')
    ohe = OneHotEncoder(inputCol='sex_index', outputCol='sex_ohe')
    va1 = VectorAssembler(inputCols=['length',
                                     'diameter',
                                     'height',
                                     'whole_weight',
                                     'shucked_weight',
                                     'viscera_weight',
                                     'shell_weight'],
                          outputCol='concats')
    ssr = StandardScaler(inputCol='concats', outputCol='scales')
    va2 = VectorAssembler(inputCols=['sex_ohe', 'scales'], outputCol='features')
    pl = Pipeline(stages=[inx, ohe, va1, ssr, va2])
    
    # fit model
    feature_eng_model = pl.fit(train_df)
    train_features_df = feature_eng_model.transform(train_df).select('features', 'rings')
    test_features_df = feature_eng_model.transform(test_df).select('features', 'rings')
    
    # write
    (train_features_df
    .repartition(args.repartition_num)
    .write
    .mode('overwrite')
    .parquet(('s3://' + os.path.join(args.s3_output_bucket, args.s3_output_prefix, 'train', 'train.parquet'))))
    (test_features_df
    .repartition(args.repartition_num)
    .write
    .mode('overwrite')
    .parquet(('s3://' + os.path.join(args.s3_output_bucket, args.s3_output_prefix, 'test', 'test.parquet'))))
    
    # kill app
    spark.stop()

if __name__ == '__main__':
    main()