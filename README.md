# Spark XKmeans

[![Maven Central](https://img.shields.io/maven-central/v/org.tupol/spark-xkmeans_2.11.svg)](https://mvnrepository.com/artifact/org.tupol/spark-xkmeans) &nbsp;
[![GitHub](https://img.shields.io/github/license/tupol/spark-xkmeans.svg)](https://github.com/tupol/spark-xkmeans/blob/master/LICENSE) &nbsp; 
[![Travis (.org)](https://img.shields.io/travis/tupol/spark-xkmeans.svg)](https://travis-ci.com/tupol/spark-xkmeans) &nbsp; 
[![Codecov](https://img.shields.io/codecov/c/github/tupol/spark-xkmeans.svg)](https://codecov.io/gh/tupol/spark-xkmeans) &nbsp;
[![Javadocs](https://www.javadoc.io/badge/org.tupol/spark-xkmeans_2.11.svg)](https://www.javadoc.io/doc/org.tupol/spark-xkmeans_2.11) &nbsp;
[![Gitter](https://badges.gitter.im/spark-xkmeans/community.svg)](https://gitter.im/spark-xkmeans/community?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge) &nbsp; 
[![Twitter](https://img.shields.io/twitter/url/https/_tupol.svg?color=%2317A2F2)](https://twitter.com/_tupol) &nbsp; 

## Motivation
While working on PoC for fraud detection in the banking sector we experimented with the K-Means 
algorithm for detecting anomalies. Interestingly enough, one of the first questions that came up 
were “why is this an anomaly?”. 

The question makes perfect sense when the amount of features used in the building the scoring
model can easily go to hundreds. 

## Problem
The K-Means implementation in Spark provides a single output column containing the predicted 
cluster. We need to find a way to collect additional information so we can explain why a data point
is an anomaly and explain also each cluster.

## Proposed Solution
To solve the problem, I am using the premise of a gaussian distribution of the data points inside 
the cluster, collect the statistics corresponding to each cluster, overall and by feature. The 
statistical data collected includes count, minimum, maximum, average, standard deviation, skewness 
and kurtosis. With the statistical data, for each data point we can we can use the probability 
density function to compute the probability that a data point belongs to a certain cluster. 

To distinguish anomalies we use a probability based threshold. For anomalies we can examine closer 
the probabilities that the point belongs to the cluster by each feature, and those with the lowest
probabilities are most likely the features that drove the data point to be labeled as an anomaly. 

A similar approach can help explaining each cluster in a human understandable manner, as besides
the cluster centers, which are averages, we have also additional statistical information, like 
variance, skewness and kurtosis.

## Other Remarks
There are some similarities between the proposed XKmeans and the Gaussian Mixture algorithm, 
in the sense that both compute some statistics about the clusters, but XKmeans is not changing the 
KMeans algorithm, but merely collecting ome statistical data on the side in order to produce the 
feature by feature probabilities.

## Usage
The `XKMeans` can be used instead of the traditional `KMeans` algorithm.

```scala
import org.apache.spark.ml.clustering.tupol.XKMeans
import org.apache.spark.ml.evaluation.ClusteringEvaluator
import org.apache.spark.sql.DataFrame

// Loads data.
val dataset: DataFrame = ???

// Trains a k-means model.
val xkmeans = new XKMeans().setK(2).setSeed(1L)
val model = xkmeans.fit(dataset)

// Make predictions
val predictions = model.transform(dataset)

// Evaluate clustering by computing Silhouette score
val evaluator = new ClusteringEvaluator()

val silhouette = evaluator.evaluate(predictions)
println(s"Silhouette with squared euclidean distance = $silhouette")

// Shows the result.
println("Cluster Centers: ")
model.clusterCenters.foreach(println)

```

## Input Parameters

Standard `KMeans` parameters:
- `k` is the number of desired clusters. Note that it is possible for fewer than k clusters to be returned, for example, if there are fewer than k distinct points to cluster.
- `maxIterations` is the maximum number of iterations to run.
- `initializationMode` specifies either random initialization or initialization via `k-means||`.
- `runs` This param has no effect since Spark 2.0.0.
- `initializationSteps` determines the number of steps in the `k-means||` algorithm.
- `epsilon` determines the distance threshold within which we consider k-means to have converged.
- `initialModel` is an optional set of cluster centers used for initialization. If this parameter is supplied, only one run is performed.

Specific `XKMeans` parameters:
- `featuresCol` is an optional list of feature names that can be used to express better the probability by feature.


## Input Columns

| Param name              | Type(s)    | Default                | Description                      |
| ----------------------- | ---------- | ---------------------- | -------------------------------- |
| featuresCol             | Vector     | "features"             | Feature vector                   |

## Output Columns

| Param name              | Type(s)    | Default                | Description                      |
| ----------------------- | ---------- | ---------------------- | -------------------------------- |
| predictionCol           | Int        | "prediction"           | Predicted cluster center         |
| distanceToCentroid      | Double     | "distanceToCentroid"   | Distance to cluster center       |
| probabilityCol          | Double     | "probability"          | Probability of belonging to cluster |
| probabilityByFeatureCol | Vector     | "probabilityByFeature" | Probability by each feature / dimension |


## Conclusion
The approach exemplified through the Spark ML K-Means extension can help understanding the 
clusters and the predictions better. Understanding the clusters can help with all K-Means based 
use-cases, weather they are classification problem or anomaly detection. 
Understanding the prediction results is very valuable for anomaly detection use cases.

The proposed solution has it limitations. 
First, the solution assumes a normal distribution of data inside the cluster so understanding the 
type of data distribution is important. 
Second, understanding each feature is crucial in understanding both the clusters themselves as
well as why some data points were labeled anomalies. 
There are also lessons learned from trying to extend the Spark ML library and what are the
limitations of it.

## Audience
Data scientists that are using Spark ML library and are interested in anomaly detection cases as 
well as developers looking into how to extends the existing Spark ML framework or statistical data 
composition.
