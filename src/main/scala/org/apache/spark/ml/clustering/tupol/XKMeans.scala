package org.apache.spark.ml.clustering.tupol

import org.apache.spark.annotation.Experimental
import org.apache.spark.ml.clustering.tupol.evaluation.{ ClusteringDistanceSummary, ClusteringFeaturesSummary }
import org.apache.spark.ml.clustering.{ KMeans, KMeansModel, KMeansParams }
import org.apache.spark.ml.linalg.{ Vector, Vectors }
import org.apache.spark.ml.param.shared._
import org.apache.spark.ml.param.{ DoubleParam, Param, ParamMap, ParamValidators }
import org.apache.spark.ml.util.Identifiable
import org.apache.spark.mllib.clustering.tupol.adapters._
import org.apache.spark.mllib.clustering.{ DistanceMeasure, KMeans => MLlibKMeans, KMeansModel => MLlibKMeansModel }
import org.apache.spark.mllib.linalg.{ Vectors => OldVectors }
import org.apache.spark.sql.functions._
import org.apache.spark.sql.{ DataFrame, Dataset, Row }

/**
 * Extended KMeans algorithm.
 *
 * Calculates the following:
 * - cluster (prediction); already available in the default KMeans algorithm.
 * - distance to cluster
 * - probability
 * - probability by feature (dimension)
 *
 * Note: The probability by feature algorithm is based on the ideas presented in https://github.com/tupol/naive-ml;
 *       https://github.com/tupol/naive-ml/blob/master/src/main/scala/tupol/ml/clustering/KMeansGaussian.scala.
 *
 * Note: The probability by feature algorithm can be rendered useless if a feature/dimension reduction algorithm
 * is used before applying XKMeans2, as we will be unable to track back the exact feature which contributed to a
 * record being classified as an anomaly.
 *
 * Note: This is by far not a perfect solution yet, as the general assumption is that the data follows a
 * normal distribution, which is not always the case.
 *
 * @param uid
 */
@Experimental
class XKMeans(override val uid: String) extends KMeans
  with XKMeansParams {

  def this() = this(Identifiable.randomUID("xkmeans"))

  setDefault(featureNames -> None, xSigma -> 3.0)

  def setFeatureNames(value: Option[Seq[String]]): this.type = set(featureNames, value)

  override def fit(dataset: Dataset[_]): XKMeansModel = {

    val rdd = dataset.select(col($(featuresCol))).rdd.map { case Row(point: Vector) => OldVectors.fromML(point) }.cache

    val algo = new MLlibKMeans()
      .setK($(k))
      .setInitializationMode($(initMode))
      .setInitializationSteps($(initSteps))
      .setMaxIterations($(maxIter))
      .setSeed($(seed))
      .setEpsilon($(tol))
      .setDistanceMeasure($(distanceMeasure))

    val distanceMeasureInstance: DistanceMeasure = DistanceMeasure.decodeFromString($(distanceMeasure))
    val parentModel = algo.run(rdd)
    val clusterCenters = parentModel.clusterCenters.map(cc => Vectors.dense(cc.toArray))
    val predictions = rdd.map(v => v.asML).zip(parentModel.predict(rdd)).map(_.swap)
    val metrics = new ClusteringDistanceSummary(predictions.map {
      case (k, point) =>
        distanceMeasureInstance.distance(point, clusterCenters(k))
        (k, Vectors.sqdist(point, clusterCenters(k)))
    })
    val metricsByFeature = new ClusteringFeaturesSummary(predictions, parentModel.clusterCenters.map(v => v.asML))
    val model = new XKMeansModel(uid, parentModel, metrics, metricsByFeature)
    copyValues(model)
  }

}

object XKMeansModel {

  def apply(
    sourceModel: XKMeansModel,
    clusterCenters: Array[Vector]): XKMeansModel = {
    sourceModel.copyValues(new XKMeansModel(
      sourceModel.uid,
      new MLlibKMeansModel(clusterCenters.map(cc => OldVectors.fromML(cc))),
      sourceModel.distanceSummary,
      sourceModel.featuresSummary))
  }
}

class XKMeansModel private[ml] (
  override val uid: String,
  override private[clustering] val parentModel: MLlibKMeansModel,
  val distanceSummary: ClusteringDistanceSummary,
  val featuresSummary: ClusteringFeaturesSummary) extends KMeansModel(uid, parentModel) with XKMeansParams {

  override def copy(extra: ParamMap): XKMeansModel = {
    val copied = new XKMeansModel(uid, parentModel, distanceSummary, featuresSummary)
    copyValues(copied, extra)
  }

  override def transform(dataset: Dataset[_]): DataFrame = {

    import org.tupol.stats.vectorops._
    val k = clusterCenters.size
    val xSigma = getXSigma
    val epsilons = (0 until k).map { k =>
      val metK = featuresSummary.summaryByCluster(k)
      val epsilons = (metK.stdev() * xSigma) / 10
      (k, epsilons)
    }.toMap
    val nSigmas = Seq.fill(featuresSummary.summaryByModel.m2.size)(xSigma)
    val degenerateSolution = 1E-9

    val predictUDF = udf((vector: Vector) => predict(vector))

    val distanceUDF = udf((vector: Vector, k: Int) =>
      Vectors.sqdist(vector, Vectors.dense(parentModel.clusterCenters(k).toArray)))

    // Calculate probability of belonging to a cluster for each dimension/feature
    val probabilityByDimUDF = udf((k: Int, record: Vector) => {
      val metK = featuresSummary.summaryByCluster(k)
      val probabilities = metK.probabilityNSigma(record.toArray, epsilons(k), nSigmas, degenerateSolution)
      probabilities
    })

    val distanceMeasureInstance: DistanceMeasure = DistanceMeasure.decodeFromString($(distanceMeasure))
    // Calculate the overall probability of a record to belong to a cluster
    val probabilityUDF = udf((k: Int, record: Vector) => {
      val metK = distanceSummary.summarybyCluster(k)
      val distance = distanceMeasureInstance.distance(record, clusterCenters(k))
      // TODO move the range and or epsilon outside the UDF
      val epsilon = (metK.stdev() * xSigma) / 10
      val probabilities = metK.probabilityNSigma(distance, epsilon, xSigma, degenerateSolution)
      probabilities
    })

    dataset.withColumn($(predictionCol), predictUDF(col($(featuresCol))))
      .withColumn($(distanceToCentroidCol), distanceUDF(col($(featuresCol)), col($(predictionCol))))
      .withColumn($(probabilityByFeatureCol), probabilityByDimUDF(col($(predictionCol)), col($(featuresCol))))
      .withColumn($(probabilityCol), probabilityUDF(col($(predictionCol)), col($(featuresCol))))
  }
}

/**
 * Common params for XKMeans and XKMeansModel
 */
private[clustering] trait XKMeansParams extends KMeansParams
  with HasDistanceToCentroidCol
  with HasProbabilityCol
  with HasProbabilityByFeatureCol {

  /**
   * Set the names of the features corresponding to each dimension of the input vectors column
   *
   * @group param
   */
  final val featureNames = new Param[Option[Seq[String]]](this, "featureNames", "The names of the each feature from the input column of vectors")

  /** @group getParam */
  def getFeatureNames: Option[Seq[String]] = $(featureNames)

  /**
   * The ratio of standard deviation used to compute probability
   *
   * @group param
   */
  final val xSigma = new DoubleParam(this, "xSigma", "The sigma range for probability computation; the ratio of standard deviation used to compute probability", ParamValidators.gt(0))
  /** @group getParam */
  def getXSigma: Double = $(xSigma)

  def setXSigma(value: Double): this.type = set(xSigma, value)

}
