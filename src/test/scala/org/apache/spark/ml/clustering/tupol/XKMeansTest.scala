package org.apache.spark.ml.clustering.tupol

import org.apache.spark.ml.clustering.tupol.ClusterGen2D._
import org.apache.spark.ml.linalg.{ VectorUDT, Vectors }
import org.apache.spark.sql.Row
import org.apache.spark.sql.types.{ StructField, StructType }
import org.scalatest.{ FunSuite, Matchers }

/**
 * Simple tests for XKMeansSharedSparkContext
 */
class XKMeansTest extends FunSuite with SparkContextSpec with Matchers {

  val centers_2 = Seq(Vectors.dense(Array(0.0, 0.0)), Vectors.dense(Array(4.0, 0.0)))

  test("Probability by feature/dimension with no explicit description for an over-sized cluster number") {

    // Create the data points in clusters:
    // Two circles with radius 1, one centered in (0, 0) and the other one in (4, 0) )
    val dataPoints = centers_2.map(p => disc(1000, p)).reduce(_ ++ _)

    // Define a schema for the features as Vectors
    val schema = new StructType(Array(StructField("features", new VectorUDT, false)))

    // Create a DataFrame with the clusters
    val df = sqlContext.createDataFrame(sc.parallelize(dataPoints).map(Row(_)), schema)

    // Train the XKMeans model
    val xkmeans = new XKMeans().setK(2).setMaxIter(50).setTol(1E-6).fit(df)

    // Test probability by dimension for a record close to the centroid
    val testData_Center = sc.parallelize(Seq(Vectors.dense(Array(0.0, 0.0))))
    val testDF_Center = sqlContext.createDataFrame(testData_Center.map(Row(_)), schema)
    val resultDF_Center = xkmeans.transform(testDF_Center)
    val probByDim_Center = resultDF_Center.first().getAs[Seq[Double]]("probabilityByFeature")
    probByDim_Center(0) should be > 0.900
    probByDim_Center(1) should be > 0.900

    println(XKMeansReporting.modelReport(xkmeans))
  }

  test("Probability by feature/dimension for 2 simple clusters") {

    // Create the data points in clusters:
    // Two circles with radius 1, one centered in (0, 0) and the other one in (4, 0) )
    val dataPoints = centers_2.map(p => disc(1000, p)).reduce(_ ++ _)

    // Define a schema for the features as Vectors
    val schema = new StructType(Array(StructField("features", new VectorUDT, false)))

    // Create a DataFrame with the clusters
    val df = sqlContext.createDataFrame(sc.parallelize(dataPoints).map(Row(_)), schema)

    // Train the XKMeans model
    val xkmeans = new XKMeans().setSeed(1).setK(2).setMaxIter(10).setTol(1E-6)
      .setFeatureNames(Some(Seq("feature0", "feature1"))).fit(df)

    // Test probability by dimension for a record close to the centroid
    val testData_Center = sc.parallelize(Seq(Vectors.dense(Array(0.0, 0.0))))
    val testDF_Center = sqlContext.createDataFrame(testData_Center.map(Row(_)), schema)
    val resultDF_Center = xkmeans.transform(testDF_Center)
    val probByDim_Center = resultDF_Center.first().getAs[Seq[Double]]("probabilityByFeature")
    probByDim_Center(0) should be > 0.900
    probByDim_Center(1) should be > 0.900

    // Test probability by dimension for a record far from the centroid on the first dimension
    val testData_Dim1 = sc.parallelize(Seq(Vectors.dense(Array(1.5, 0.0))))
    val testDF_Dim1 = sqlContext.createDataFrame(testData_Dim1.map(Row(_)), schema)
    val resultDF_Dim1 = xkmeans.transform(testDF_Dim1)
    val probByDim_Dim1 = resultDF_Dim1.first().getAs[Seq[Double]]("probabilityByFeature").toArray
    probByDim_Dim1(0) should be < 0.4
    probByDim_Dim1(1) should be > 0.9

    // Test probability by dimension for a record far from the centroid on the second dimension
    val testData_Dim2 = sc.parallelize(Seq(Vectors.dense(Array(0.0, 1.5))))
    val testDF_Dim2 = sqlContext.createDataFrame(testData_Dim2.map(Row(_)), schema)
    val resultDF_Dim2 = xkmeans.transform(testDF_Dim2)
    val probByDim_Dim2 = resultDF_Dim2.first().getAs[Seq[Double]]("probabilityByFeature").toArray
    probByDim_Dim2(0) should be > 0.900
    probByDim_Dim2(1) should be < 0.4
  }

  test("Probability by feature/dimension for 2 clusters, one of which has variance 0 on one dimension") {

    // Create the data points in clusters :
    // A horizontal line starting from (0, 0) to (1, 0) and a circle with a radius 1 centered in (8, 0) )
    val dataPoints = line(1000, Vectors.dense(Array(0.0, 0.0))) ++ disc(1000, Vectors.dense(Array(8.0, 0.0)))

    // Define a schema for the features as Vectors
    val schema = new StructType(Array(StructField("features", new VectorUDT, false)))

    // Create a DataFrame with the clusters
    val df = sqlContext.createDataFrame(sc.parallelize(dataPoints).map(Row(_)), schema)

    // Train the XKMeans model
    val xkmeans = new XKMeans().setK(2).setMaxIter(10).setTol(1E-6).fit(df)

    // Test probability by dimension for a record close to the centroid of the line cluster
    val testData_Center = sc.parallelize(Seq(Vectors.dense(Array(0.5, 0.0))))
    val testDF_Center = sqlContext.createDataFrame(testData_Center.map(Row(_)), schema)
    val resultDF_Center = xkmeans.transform(testDF_Center)
    val probByDim_Center = resultDF_Center.first().getAs[Seq[Double]]("probabilityByFeature").toArray
    probByDim_Center(0) should be > 0.9
    probByDim_Center(1) should be > 0.9

    // Test probability by dimension for a record close to the middle of the line cluster on the first dimension,
    // but slightly off the second dimension
    val testData_CenterOff = sc.parallelize(Seq(Vectors.dense(Array(0.5, 0.001))))
    val testDF_CenterOff = sqlContext.createDataFrame(testData_CenterOff.map(Row(_)), schema)
    val resultDF_CenterOff = xkmeans.transform(testDF_CenterOff)
    val probByDim_CenterOff = resultDF_CenterOff.first().getAs[Seq[Double]]("probabilityByFeature").toArray
    probByDim_CenterOff(0) should be > 0.9
    probByDim_CenterOff(1) should be > 0.9

    // Test probability by dimension for a record close to the middle of the line cluster on the first dimension,
    // but very very little off the second dimension
    val testData_CenterOffSmall = sc.parallelize(Seq(Vectors.dense(Array(0.5, 0.000000001))))
    val testDF_CenterOffSmall = sqlContext.createDataFrame(testData_CenterOffSmall.map(Row(_)), schema)
    val resultDF_CenterOffSmall = xkmeans.transform(testDF_CenterOffSmall)
    val probByDim_CenterOffSmall = resultDF_CenterOffSmall.first().getAs[Seq[Double]]("probabilityByFeature").toArray
    probByDim_CenterOffSmall(0) should be > 0.9
    probByDim_CenterOffSmall(1) should be > 0.9

    // Test probability by dimension for a record far from the centroid on the first dimension and right on the line in the second dimension
    val testData_Dim1 = sc.parallelize(Seq(Vectors.dense(Array(1.6, 0.00000))))
    val testDF_Dim1 = sqlContext.createDataFrame(testData_Dim1.map(Row(_)), schema)
    val resultDF_Dim1 = xkmeans.transform(testDF_Dim1)
    val probByDim_Dim1 = resultDF_Dim1.first().getAs[Seq[Double]]("probabilityByFeature").toArray
    probByDim_Dim1(0) should be < 0.4
    probByDim_Dim1(1) should be > 0.900
  }

}
