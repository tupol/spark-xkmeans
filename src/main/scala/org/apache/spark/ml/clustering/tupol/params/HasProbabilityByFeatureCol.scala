package org.apache.spark.ml.param.shared

import org.apache.spark.ml.param.{ Param, Params }

/**
 * Trait for shared param probabilityByFeatureCol (default: "probabilityByFeature").
 */
private[ml] trait HasProbabilityByFeatureCol extends Params {
  /**
   * Param for probabilityByFeature column name.
   *
   * @group param
   */
  final val probabilityByFeatureCol: Param[String] = new Param[String](this, "probabilityByFeature", "Column name for the probability for each feature (dimension) to belong to a cluster")

  setDefault(probabilityByFeatureCol, "probabilityByFeature")

  /** @group getParam */
  final def getProbabilityByFeatureCol: String = $(probabilityByFeatureCol)

}
