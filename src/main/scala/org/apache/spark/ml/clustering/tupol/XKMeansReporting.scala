package org.apache.spark.ml.clustering.tupol

import org.apache.spark.mllib.clustering.DistanceMeasure
import org.tupol.stats.Stats

/**
 * Defines a set of reports generated from a `PipelineModel` with `XKMeansModel`
 * and proper feature names.
 */
object XKMeansReporting {

  /**
   * Returns a concatenation of all reports from `XKMeansReporting`.
   *
   * @param  pipelineModel `PipelineModel` with `XKMeansModel` and featureNames
   * @return report
   */
  def modelReport(implicit model: XKMeansModel): String =
    s""":## XKMeansModel.UID: ${model.uid}
          :
          :
          :$modelParametersReport
          :
          :
          :$vectorFeatureNamesReport
          :
          :
          :$summaryByModelReport
          :
          :
          :$summaryByClusterReport
          :
          :
          :$summaryByModelByFeaturesReport
          :
          :
          :$summaryByClusterByFeaturesReport
          :
          :
          :$clusterDistancesReport
          :
          :
          :$crossClusterProbabilityReport
          :
          :
          :$legend
          :""".stripMargin(':')

  /**
   * Returns a Markdown table report with
   * [[org.apache.spark.ml.clustering.tupol.evaluation.ClusteringDistanceSummary#summaryByModel]]
   * statistics.
   *
   * @param model XKMeansModel
   * @return report
   */
  def modelParametersReport(implicit model: XKMeansModel): String = {

    f""":### Model Parameters
       :| Parameter               | ${"Explanation"}%-20s | ${"Explanation"}%-90s |
       :| :---------------------- | -------------------: | :----------------------------------------------------------------------------------------- |
       :| k                       | ${model.getK}%20d | ${renderLimitedText(model.k.doc)}%-90s |
       :| xSigma                  | ${model.getXSigma}%20.6f | ${renderLimitedText(model.xSigma.doc)}%-90s |
       :| maxIter                 | ${model.getMaxIter}%20d | ${renderLimitedText(model.maxIter.doc)}%-90s |
       :| tol                     | ${model.getTol}%20.6f | ${renderLimitedText(model.tol.doc)}%-90s |
       :| seed                    | ${model.getSeed}%20d | ${renderLimitedText(model.seed.doc)}%-90s |
       :| initSteps               | ${model.getInitSteps}%20d | ${renderLimitedText(model.initSteps.doc)}%-90s |
       :| initMode                | ${model.getInitMode}%-20s | ${renderLimitedText(model.initMode.doc)}%-90s |
       :| distanceMeasure         | ${model.getDistanceMeasure}%-20s | ${renderLimitedText(model.distanceMeasure.doc)}%-90s |
       :| featuresCol             | ${model.getFeaturesCol}%-20s | ${renderLimitedText(model.featuresCol.doc)}%-90s |
       :| predictionCol           | ${model.getPredictionCol}%-20s | ${renderLimitedText(model.predictionCol.doc)}%-90s |
       :| distanceToCentroidCol   | ${model.getDistanceToCentroidCol}%-20s | ${renderLimitedText(model.distanceToCentroidCol.doc)}%-90s |
       :| probabilityCol          | ${model.getProbabilityCol}%-20s | ${renderLimitedText(model.probabilityCol.doc)}%-90s |
       :| probabilityByFeatureCol | ${model.getProbabilityByFeatureCol}%-20s | ${renderLimitedText(model.probabilityByFeatureCol.doc)}%-90s |"""
      .stripMargin(':')
  }

  private def renderLimitedText(text: String, limit: Int = 90) =
    if (text.length > limit) s"${text.take(87)}..." else text

  /**
   * Returns a Markdown table report with
   * [[org.apache.spark.ml.clustering.tupol.evaluation.ClusteringDistanceSummary#summaryByModel]]
   * statistics.
   *
   * @param model XKMeansModel
   * @return report
   */
  def summaryByModelReport(implicit model: XKMeansModel): String = {
    val byModel = model.distanceSummary.summaryByModel
    f""":### Distance Summary By Model
        :| Parameter        | Value         |
        :| :--------------- | ------------: |
        :| TotalCount       | ${byModel.count.toLong}%13d |
        :| Avg. SSE         | ${byModel.sse / byModel.count}%13.6E |
        :| Min. Distance    | ${byModel.min}%13.6E |
        :| Avg. Distance    | ${byModel.avg}%13.6E |
        :| Max. Distance    | ${byModel.max}%13.6E |
        :| Variance         | ${byModel.variance()}%13.6E |
        :| Std. Deviation   | ${byModel.stdev()}%13.6E |
        :| Skewness         | ${byModel.skewness}%13.6E |
        :| Kurtosis         | ${byModel.kurtosis}%13.6E |"""
      .stripMargin(':')
  }

  /**
   * Returns a Markdown table report with
   * [[org.apache.spark.ml.clustering.tupol.evaluation.ClusteringDistanceSummary#summaryByCluster]]
   * statistics.
   *
   * @param model XKMeansModel
   * @return report
   */
  def summaryByClusterReport(implicit model: XKMeansModel): String = {
    val header =
      f""":### Distance Summary By Cluster
          :| K     | ${"Count"}%-10s | ${"Avg. SSE"}%-13s | ${"Min. Dist."}%-13s | ${"Avg. Dist."}%-13s | ${"Max. Dist."}%-13s | ${"Variance"}%-13s | ${"Std. Dev."}%-13s | ${"Skewness"}%-13s | ${"Kurtosis"}%-13s |
          :| ----: | ---------: | ------------: | ------------: | ------------: | ------------: | ------------: | ------------: | ------------: | ------------: |""".stripMargin(':')

    val rows = model.distanceSummary.summarybyCluster.zipWithIndex.map {
      case (cds, k) =>
        f"| $k%5d | ${cds.count.toLong}%10d | ${cds.sse / cds.count}%13.6E | ${cds.min}%13.6E | ${cds.avg}%13.6E | ${cds.max}%13.6E | ${cds.variance()}%13.6E | ${cds.stdev()}%13.6E | ${cds.skewness}%13.6E | ${cds.kurtosis}%13.6E |"
    }.mkString("\n")
    s""":$header
        :$rows""".stripMargin(':')
  }

  /**
   * Returns a Markdown table report with
   * [[org.apache.spark.ml.clustering.tupol.evaluation.ClusteringDistanceSummary#summaryByCluster]]
   * statistics.
   *
   * @param model XKMeansModel
   * @return report
   */
  def summaryByModelByFeaturesReport(implicit model: XKMeansModel): String = {
    val header =
      f""":### Features Summary By Model
         :| Feat. | ${"Count"}%-10s | ${"Avg. SSE"}%-13s | ${"Min. Val."}%-13s | ${"Avg. Val."}%-13s | ${"Max. Val."}%-13s | ${"Variance"}%-13s | ${"Std. Dev."}%-13s | ${"Skewness"}%-13s | ${"Kurtosis"}%-13s |
         :| ----: | ---------: | ------------: | ------------: | ------------: | ------------: | ------------: | ------------: | ------------: | ------------: |""".stripMargin(':')

    val cds = model.featuresSummary.summaryByModel
    val featNo = cds.min.size
    val rows = (0 until featNo).map { fn =>
      f"| $fn%5d | ${cds.count.toLong}%10d | ${cds.sse(fn) / cds.count}%13.6E | ${cds.min(fn)}%13.6E | ${cds.avg(fn)}%13.6E | ${cds.max(fn)}%13.6E | ${cds.variance()(fn)}%13.6E | ${cds.stdev()(fn)}%13.6E | ${cds.skewness(fn)}%13.6E | ${cds.kurtosis(fn)}%13.6E |"
    }.mkString("\n")
    s""":$header
       :$rows""".stripMargin(':')
  }

  /**
   * Returns a Markdown table report with
   * [[org.apache.spark.ml.clustering.tupol.evaluation.ClusteringDistanceSummary#summaryByCluster]]
   * statistics.
   *
   * @param model XKMeansModel
   * @return report
   */
  def summaryByClusterByFeaturesReport(implicit model: XKMeansModel): String = {
    val header =
      f""":### Features Summary Expanded
          :| K     | Feat. | ${"Count"}%-10s | ${"Avg. SSE"}%-13s | ${"Min. Val."}%-13s | ${"Avg. Val."}%-13s | ${"Max. Val."}%-13s | ${"Variance"}%-13s | ${"Std. Dev."}%-13s | ${"Skewness"}%-13s | ${"Kurtosis"}%-13s |
          :| ----: | ----: | ---------: | ------------: | ------------: | ------------: | ------------: | ------------: | ------------: | ------------: | ------------: |""".stripMargin(':')

    val rows = model.featuresSummary.summaryByCluster.zipWithIndex.map {
      case (cds, k) =>
        val featNo = cds.min.size
        (0 until featNo).map { fn =>
          cds.min(fn)
          f"| $k%5d | $fn%5d | ${cds.count.toLong}%10d | ${cds.sse(fn) / cds.count}%13.6E | ${cds.min(fn)}%13.6E | ${cds.avg(fn)}%13.6E | ${cds.max(fn)}%13.6E | ${cds.variance()(fn)}%13.6E | ${cds.stdev()(fn)}%13.6E | ${cds.skewness(fn)}%13.6E | ${cds.kurtosis(fn)}%13.6E |"
        }.mkString("\n")
    }.mkString("\n")
    s""":$header
       :$rows
       :* The "Avg. Val." column also represents the cluster centers """.stripMargin(':')
  }

  /**
   * Returns a Markdown table report of indexed feature names of the model.
   * Returns an empty table if the `pipelineModel` doesn't have any feature names.
   *
   * @param pipelineModel PipelineModel with XKMeansModel and featureNames
   * @return report
   */
  def vectorFeatureNamesReport(implicit model: XKMeansModel): String = model.getFeatureNames.map { featureNames =>
    val featuresHeader =
      f""":### Feature Names
          :| ${"Idx"}%4s | ${"Feature Name"}%-50s |
          :| ---: | :------------------------------------------------- |""".stripMargin(':')

    val featuresTable = featureNames.zipWithIndex.map {
      case (name, idx) => f"| $idx%4d | $name%-50s |"
    }.mkString("\n")

    s""":$featuresHeader
        :$featuresTable""".stripMargin(':')

  }.getOrElse("")

  /**
   * Returns a Markdown table report with distances between cluster centres.
   *
   * @param model XKMeansModel
   * @return report
   */
  def clusterDistancesReport(implicit model: XKMeansModel): String = {
    import org.apache.spark.mllib.clustering.tupol.adapters._
    val centres = model.clusterCenters

    val header = "| K1   \\   K2 |" + centres.indices.map(idx => f" $idx%11d |").mkString
    val headerDelimiter = "| ----------: |" + centres.map(_ => " ----------: |").mkString

    val distanceMeasureInstance: DistanceMeasure = DistanceMeasure.decodeFromString(model.getDistanceMeasure)
    val dists = centres.map(from => centres.map(to => distanceMeasureInstance.distance(from, to)))

    val rows = dists.map {
      row => row.map(x => f" $x%11.5E |").mkString
    }.zipWithIndex.map {
      case (row, idx) => f"| $idx%11d |$row"
    }.mkString("\n")

    s""":### Cluster Centres Distances
        :$header
        :$headerDelimiter
        :$rows""".stripMargin(':')
  }

  /**
   * Returns a Markdown table report with top list of distances starting from the shortest.
   *
   * @param model XKMeansModel
   * @return report
   */
  def crossClusterProbabilityReport(implicit model: XKMeansModel): String = {
    import org.apache.spark.mllib.clustering.tupol.adapters._

    val distanceMeasureInstance: DistanceMeasure = DistanceMeasure.decodeFromString(model.getDistanceMeasure)
    val centres = model.clusterCenters

    val header = "|     K1 |     K2 |             D | Prob(K2 ∈ K1) | Prob(K1 ∈ K2) |"
    val headerDelimiter = "| -----: | -----: | ------------: | ------------: | ------------: |"

    val centresWithIdxs = centres.zipWithIndex

    val kStats = model.distanceSummary.summarybyCluster

    val pairDists = for {
      (c1, idx1) <- centresWithIdxs
      (c2, idx2) <- centresWithIdxs
      d = distanceMeasureInstance.distance(c1, c2)
      if idx1 < idx2
    } yield (idx1, idx2, d, kStats(idx1).probabilityNSigma(d), kStats(idx2).probabilityNSigma(d))

    val rows = pairDists.sortWith {
      case (t1, t2) => t1._4 + t1._5 > t2._4 + t2._5
    }.map {
      case (k1, k2, d, p12, p21) => f"| $k1%6d | $k2%6d | $d%13.5E | $p12%1.11f | $p21%1.11f |"
    }.mkString("\n")

    s""":### Probability of Clusters to Belong to Other Clusters
       :$header
       :$headerDelimiter
       :$rows
       :
       :""".stripMargin(':')

  }

  /**
   * Returns a Markdown table report with top list of distances starting from the shortest.
   *
   * @param model XKMeansModel
   * @return report
   */
  def clusterDistancesTopListReport(implicit model: XKMeansModel): String = {
    import org.apache.spark.mllib.clustering.tupol.adapters._

    val distanceMeasureInstance: DistanceMeasure = DistanceMeasure.decodeFromString(model.getDistanceMeasure)
    val centres = model.clusterCenters

    val header = "|     K1 |     K2 |           D |     3-σ interval |"
    val headerDelimiter = "| -----: | -----: | ----------: | ---------------: |"

    val centresWithIdxs = centres.zipWithIndex

    val pairDists = for {
      (c1, idx1) <- centresWithIdxs
      (c2, idx2) <- centresWithIdxs
      if idx1 < idx2
    } yield (idx1, idx2, distanceMeasureInstance.distance(c1, c2))

    val dists = pairDists.map {
      case (_, _, d) => d
    }

    val ds = Stats.fromDoubles(dists)

    def sigma(v: Double) = threeSigmaRuler(v, ds.avg, ds.stdev())

    val rows = pairDists.sortBy {
      case (_, _, d) => d
    }.map {
      case (k1, k2, d) => f"| $k1%6d | $k2%6d | $d%11.5E | ${sigma(d)}%16s |"
    }.mkString("\n")

    s""":### Cluster Centres Shortest Distances Top
        :$header
        :$headerDelimiter
        :$rows
        :
        :""".stripMargin(':')

  }

  def threeSigmaRuler(value: Double, mean: Double, sd: Double): String = value match {
    case v if v <= mean - 3 * sd => "(-∞, µ - 3σ]"
    case v if v <= mean - 2 * sd => "(µ - 3σ, µ - 2σ]"
    case v if v <= mean - sd => "(µ - 2σ, µ - σ]"
    case v if v <= mean => "(µ - σ, µ]"
    case v if v <= mean + sd => "(µ, µ + σ]"
    case v if v <= mean + 2 * sd => "(µ + σ, µ + 2σ]"
    case v if v <= mean + 3 * sd => "(µ + 2σ, µ + 3σ]"
    case _ => "(µ + 3σ, +∞)"
  }

  lazy val legend: String =
    """:### Legend
       :
       :| Term             | Explanation                                                     |
       :| :--------------- | :-------------------------------------------------------------- |
       :| Avg, Mean, µ     | Average                                                         |
       :| SD, σ            | Standard Deviation                                              |
       :| SSE              | Sum of Squared Errors                                           |""".stripMargin(':')
}
