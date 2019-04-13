package org.apache.spark.ml.clustering.tupol.evaluation

import org.apache.spark.ml.linalg.Vector
import org.apache.spark.rdd.RDD
import org.tupol.stats.VectorStats
import org.tupol.stats.VectorStats._

/**
 * Calculate distance statistics for each feature, by cluster and for the entire model.
 *
 * This provides a deeper insight into the model itself than the {@link ClusteringStats}
 *
 * @param predictions the tuple of cluster id and distance vector used to compute the statistics
 */
case class ClusteringFeaturesSummary(@transient private val predictions: RDD[(Int, Vector)], private val centroids: Array[Vector]) {

  // Cache input if necessary
  private[this] val inputNotCached = predictions.getStorageLevel == None
  if (inputNotCached) predictions.persist()
  /**
   * The distance statistic collected for each cluster as an ordered sequence corresponding to the cluster index
   */
  val summaryByCluster: Seq[VectorStats] = {
    predictions
      // The performance of aggregateByKey is not that great and it can be improved with a mapPartitions approach
      // We keep it like this for now for clarity
      .aggregateByKey(zeroDouble)(
        (stats, record) => stats |+| record.toArray,
        (stats1, stats2) => stats1 |+| stats2)
      .collect.toIndexedSeq.sortBy(_._1).map(_._2)
  }

  val summaryByModel: VectorStats = {
    summaryByCluster.reduce(_ |+| _)
  }

  // Release the cache, if necessary
  if (inputNotCached) predictions.unpersist()

}

/**
 * Basic statistics for each dimension (minimum, mean, maximum and variance)
 *
 * @param min
 * @param avg
 * @param max
 * @param variance
 */
case class DistanceStatsByFeature(count: Long, min: Vector, avg: Vector, max: Vector, variance: Vector)

