package org.apache.spark.ml.clustering.tupol

import org.apache.spark.ml.linalg.{ Vector, Vectors }
import org.apache.spark.rdd.RDD

/**
 * Additional operations for linalg.Vector
 */
object vectorops {

  /**
   * Additional operations for Double with Vector
   */
  implicit class DoubleOps(val scalar: Double) extends AnyVal {

    def +(vector: Vector): Vector = {
      opByDim(vector, (x1: Double, x2: Double) => x1 + x2)
    }

    def *(vector: Vector): Vector = {
      opByDim(vector, (x1: Double, x2: Double) => x1 * x2)
    }

    def -(vector: Vector): Vector = {
      opByDim(vector, (x1: Double, x2: Double) => x1 - x2)
    }

    def /(vector: Vector): Vector = {
      opByDim(vector, (x1: Double, x2: Double) => x1 / x2)
    }

    private[ml] def opByDim(vector: Vector, op: (Double, Double) => Double) =
      Vectors.dense(vector.toArray.map { op(scalar, _) })

  }

  /**
   * Added operations by dimension
   * @param vector
   */
  implicit class VectorOps(vector: Vector) {

    /**
     * Squared distances by dimension
     *
     * @param that
     * @return
     */
    def sqdistByDim(that: Vector): Vector = {
      require(vector.size == that.size)
      Vectors.dense(vector.toArray.zip(that.toArray).map { case (x1, x2) => (x1 - x2) * (x1 - x2) })
    }

    /**
     * Add 2 vectors, dimension by dimension
     *
     * @param that
     * @return
     */
    def +(that: Vector): Vector = {
      require(vector.size == that.size)
      op(vector, that, (x1: Double, x2: Double) => x1 + x2)
    }

    /**
     * Add each value in this vector with the provided scalar value
     *
     * @param scalar
     * @return
     */
    def +(scalar: Double): Vector = {
      op(vector, scalar, (x1: Double, x2: Double) => x1 + x2)
    }

    /**
     * Subtract that vector from this vector, dimension by dimension
     *
     * @param that
     * @return
     */
    def -(that: Vector): Vector = {
      require(vector.size == that.size)
      op(vector, that, (x1: Double, x2: Double) => x1 - x2)
    }

    /**
     * Subtract each value in this vector with the provided scalar value
     *
     * @param scalar
     * @return
     */
    def -(scalar: Double): Vector = {
      op(vector, scalar, (x1: Double, x2: Double) => x1 - x2)
    }

    /**
     * Change the sign of each value inside the vector
     *
     * @return
     */
    def unary_- : Vector = Vectors.dense(vector.toArray.map(-_))

    /**
     * Calculate the exponential by dimension
     *
     * @return
     */
    def exp: Vector = map(math.exp(_))

    /**
     * Calculate the square root by dimension
     *
     * @return
     */
    def sqrt: Vector = map(math.sqrt(_))

    /**
     * Calculate the square by dimension
     *
     * @return
     */
    def sqr: Vector = map(x => x * x)

    /**
     * Multiply this vector with that vector, dimension by dimension
     *
     * @param that
     * @return
     */
    def *(that: Vector): Vector = {
      require(vector.size == that.size)
      op(vector, that, (x1: Double, x2: Double) => x1 * x2)
    }

    /**
     * Multiply each value in this vector with the provided scalar value
     *
     * @param scalar
     * @return
     */
    def *(scalar: Double): Vector = {
      op(vector, scalar, (x1: Double, x2: Double) => x1 * x2)
    }

    /**
     * Divide each value in this vector with the provided scalar value
     *
     * @param scalar
     * @return
     */
    def /(scalar: Double): Vector = {
      op(vector, scalar, (x1: Double, x2: Double) => x1 / x2)
    }

    /**
     * Divide this vector with that vector, dimension by dimension
     *
     * @param that
     * @return
     */
    def /(that: Vector): Vector = {
      require(vector.size == that.size)
      op(vector, that, (x1: Double, x2: Double) => x1 / x2)
    }

    def map(op: (Double) => Double): Vector =
      Vectors.dense(vector.toArray.map(op))

    private[ml] def op(v1: Vector, v2: Vector, op: (Double, Double) => Double): Vector =
      Vectors.dense(v1.toArray.zip(v2.toArray).map { case (x1, x2) => op(x1, x2) })

    private[ml] def op(v1: Vector, scalar: Double, op: (Double, Double) => Double): Vector =
      Vectors.dense(v1.toArray.map { op(_, scalar) })

    private[ml] def op(v1: Vector, op: (Double) => Double): Vector =
      Vectors.dense(v1.toArray.map { op(_) })

  }

  /**
   * Added operations by Vector dimension
   * @param vectors
   */
  implicit class IterableVectorsOps(vectors: Iterable[Vector]) {

    lazy val size = vectors.size

    lazy val sumByDim: Vector = reduceByDim((a: Double, b: Double) => a + b)

    lazy val meanByDim: Vector = sumByDim / size

    lazy val maxByDim: Vector = reduceByDim(math.max)

    lazy val minByDim: Vector = reduceByDim(math.min)

    lazy val varianceByDim: Vector = varianceByDim(meanByDim)

    def varianceByDim(mean: Vector): Vector =
      vectors.map(_.sqdistByDim(mean)).sumByDim / (size - 1)

    private def reduceByDim(op: (Double, Double) => Double): Vector =
      vectors.reduce((v1, v2) =>
        Vectors.dense(v1.toArray.zip(v2.toArray).
          map(x => op(x._1, x._2))))

  }

  /**
   * Added operations by Vector dimension
   *
   * @param vectors
   */
  implicit class RDDVectorsOps(vectors: RDD[Vector]) {

    lazy val size = vectors.count

    lazy val meanByDim: Vector = sumByDim / size

    lazy val sumByDim: Vector = reduceByDim((a: Double, b: Double) => a + b)

    lazy val maxByDim: Vector = reduceByDim(math.max)

    lazy val minByDim: Vector = reduceByDim(math.min)

    def varianceByDim(mean: Vector): Vector =
      vectors.map(_.sqdistByDim(mean)).sumByDim / (size - 1)

    lazy val varianceByDim: Vector = varianceByDim(meanByDim)

    private def reduceByDim(op: (Double, Double) => Double): Vector =
      vectors.reduce((v1, v2) =>
        Vectors.dense(v1.toArray.zip(v2.toArray).
          map(x => op(x._1, x._2))))

    private def stats(op: (Double, Double) => Double) = {
      import org.tupol.stats.VectorStats
      vectors.mapPartitions { pdata =>
        Iterator(VectorStats.fromDVectors(pdata.map(_.toArray.toSeq).toIterable))
      }
    }.reduce(_ |+| _)

  }

}
