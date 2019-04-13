package org.apache.spark.ml.clustering.tupol

import org.apache.spark.ml.clustering.GaussianMixture
import org.apache.spark.ml.clustering.tupol.ClusterGen2D._
import org.apache.spark.ml.linalg.{ VectorUDT, Vectors }
import org.apache.spark.sql.Row
import org.apache.spark.sql.types.{ StructField, StructType }
import org.scalatest.{ FunSuite, Matchers }

/**
 * Simple tests for XKMeansSharedSparkContext
 */
class GaussianTest extends FunSuite with SparkContextSpec with Matchers {

  val centers_2 = Seq(Vectors.dense(Array(0.0, 0.0)), Vectors.dense(Array(4.0, 0.0)))

  test("Probability for 2 simple clusters") {

    // Create the data points in clusters:
    // Two circles with radius 1, one centered in (0, 0) and the other one in (4, 0) )
    val dataPoints = centers_2.map(p => disc(1000, p)).reduce(_ ++ _)

    // Define a schema for the features as Vectors
    val schema = new StructType(Array(StructField("features", new VectorUDT, false)))

    // Create a DataFrame with the clusters
    val df = sqlContext.createDataFrame(sc.parallelize(dataPoints).map(Row(_)), schema)

    // Train the XKMeans2 model
    val gmix = new GaussianMixture().setK(2).setMaxIter(10).setTol(1E-6).fit(df)

    gmix.gaussians.map(_.mean).foreach(println)

    // Test probability by dimension for a record close to the centroid
    val testData_Center = sc.parallelize(Seq(Vectors.dense(Array(0.0, 0.0)), Vectors.dense(Array(4.0, 0.0))))
    val testDF_Center = sqlContext.createDataFrame(testData_Center.map(Row(_)), schema)
    val resultDF_Center = gmix.transform(testDF_Center)

  }

}
