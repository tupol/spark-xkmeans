package org.apache.spark.ml.clustering.tupol.evaluation

import org.apache.spark.rdd.RDD
import org.tupol.stats.Stats
import org.tupol.stats.Stats.zeroDouble

/**
 * Calculate distance statistics, by cluster and for the entire model.
 *
 * @param predictions the tuple of cluster id and distance used to compute the statistics
 */
case class ClusteringDistanceSummary(@transient private val predictions: RDD[(Int, Double)]) {

  // Cache input if necessary
  private[this] val inputNotCached = predictions.getStorageLevel == None
  if (inputNotCached) predictions.persist()
  /**
   * The overall distance statistic collected for each cluster as an ordered sequence corresponding to the cluster index
   */
  val summarybyCluster: Seq[Stats] = {
    predictions.aggregateByKey(zeroDouble)(
      (stats, dist) => stats |+| dist,
      (stats1, stats2) => stats1 |+| stats2)
      .collect.toSeq.sortBy(_._1).map(_._2)
  }
  /**
   * The overall distance statistic collected for the entire model
   */
  val summaryByModel: Stats = {
    summarybyCluster.reduce(_ |+| _)
  }

  // Release the cache, if necessary
  if (inputNotCached) predictions.unpersist()

}
