# Spark XKmeans

## Motivation
While working on PoC for fraud detection in the banking sector we experimented with the K-Means algorithm for detecting anomalies. Interestingly enough, one of the first questions that came up were “why is this an anomaly?”. 

The question makes perfect sense when the amount of features used in the building the scoring model can easily go to hundreds. 

## Problem
The K-Means implementation in Spark provides a single output column containing the predicted cluster. We need to find a way to collect additional information so we can explain why a data point is an anomaly and explain also each cluster.

## Proposed Solution
To solve the problem, I am using the premise of a gaussian distribution of the data points inside the cluster, collect the statistics corresponding to each cluster, overall and by feature. The statistical data collected includes count, minimum, maximum, average, standard deviation, skewness and kurtosis. With the statistical data, for each data point we can we can use the probability density function to compute the probability that a data point belongs to a certain cluster. 

To distinguish anomalies we use a probability based threshold. For anomalies we can examine closer the probabilities that the point belongs to the cluster by each feature, and those with the lowest probabilities are most likely the features that drove the data point to be labeled as an anomaly. 

A similar approach can help explaining each cluster in a human understandable manner, as besides the cluster centers, which are averages, we have also additional statistical information, like variance, skewness and kurtosis.

## Conclusion
The approach exemplified through the Spark ML K-Means extension can help understanding the clusters and the predictions better. Understanding the clusters can help with all K-Means based use-cases, weather they are classification problem or anomaly detection. Understanding the prediction results is very valuable for anomaly detection use cases.

The proposed solution has it limitations. 
First, the solution assumes a normal distribution of data inside the cluster so understanding the type of data distribution is important. 
Second, understanding each feature is crucial in understanding both the clusters themselves as well as why some data points were labeled anomalies. 
There are also lessons learned from trying to extend the Spark ML library and what are the limitations of it.

## Audience
Data scientists that are using Spark ML library and are interested in anomaly detection cases as well as developers looking into how to extends the existing Spark ML framework or statistical data composition.
