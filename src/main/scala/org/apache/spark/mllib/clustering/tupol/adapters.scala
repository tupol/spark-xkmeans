package org.apache.spark.mllib.clustering.tupol

import org.apache.spark.mllib.clustering.VectorWithNorm
import org.apache.spark.mllib.linalg.{ Vector => MllibVector }
import org.apache.spark.ml.linalg.{ Vector => MlVector }
import org.apache.spark.mllib.linalg.VectorImplicits._

object adapters {

  implicit def vectorToVectorWithNorm(v: MllibVector) = new VectorWithNorm(v)
  implicit def vectorToVectorWithNorm(v: MlVector) = new VectorWithNorm(v)

}
