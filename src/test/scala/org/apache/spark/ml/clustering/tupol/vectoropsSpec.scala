package org.apache.spark.ml.clustering.tupol

import org.apache.spark.ml.clustering.tupol.vectorops._
import org.apache.spark.ml.linalg.Vectors
import org.scalatest.{ FunSuite, Matchers }

/**
 * Simple tests for vectorops
 */
class vectoropsSpec extends FunSuite with Matchers {

  test("Add a scalar and a vector") {

    val vector = Vectors.dense(Array(-2.0, -1.0, 0.0, 1.0, 2.0))

    (0.0 + vector) shouldBe vector

    (1.0 + vector) shouldBe Vectors.dense(Array(-1.0, 0.0, 1.0, 2.0, 3.0))

    (-1.0 + vector) shouldBe Vectors.dense(Array(-3.0, -2.0, -1.0, 0.0, 1.0))

    (0.5 + vector) shouldBe Vectors.dense(Array(-1.5, -0.5, 0.5, 1.5, 2.5))

  }

  test("Add a vector and a scalar") {

    val vector = Vectors.dense(Array(-2.0, -1.0, 0.0, 1.0, 2.0))

    (vector + 0.0) shouldBe vector

    (vector + 1.0) shouldBe Vectors.dense(Array(-1.0, 0.0, 1.0, 2.0, 3.0))

    (vector + -1.0) shouldBe Vectors.dense(Array(-3.0, -2.0, -1.0, 0.0, 1.0))

    (vector + 0.5) shouldBe Vectors.dense(Array(-1.5, -0.5, 0.5, 1.5, 2.5))

  }

  test("Subtract a vector from a scalar") {

    val vector = Vectors.dense(Array(-2.0, -1.0, 0.0, 1.0, 2.0))

    (0.0 - vector) shouldBe Vectors.dense(Array(2.0, 1.0, 0.0, -1.0, -2.0))

    (1.0 - vector) shouldBe Vectors.dense(Array(3.0, 2.0, 1.0, 0.0, -1.0))

    (-1.0 - vector) shouldBe Vectors.dense(Array(1.0, 0.0, -1.0, -2.0, -3.0))

    (0.5 - vector) shouldBe Vectors.dense(Array(2.5, 1.5, 0.5, -0.5, -1.5))

  }

  test("Subtract a scalar from a vector") {

    val vector = Vectors.dense(Array(-2.0, -1.0, 0.0, 1.0, 2.0))

    (vector - 0.0) shouldBe vector

    (vector - 1.0) shouldBe Vectors.dense(Array(-3.0, -2.0, -1.0, 0.0, 1.0))

    (vector - -1.0) shouldBe Vectors.dense(Array(-1.0, 0.0, 1.0, 2.0, 3.0))

    (vector - 0.5) shouldBe Vectors.dense(Array(-2.5, -1.5, -0.5, 0.5, 1.5))

  }

  test("Multiply a scalar and a vector") {

    val vector = Vectors.dense(Array(-2.0, -1.0, 0.0, 1.0, 2.0))

    // Hmm... negative zero...
    (0.0 * vector) shouldBe Vectors.dense(Array(-0.0, -0.0, 0.0, 0.0, 0.0))

    (1.0 * vector) shouldBe vector

    (-1.0 * vector) shouldBe Vectors.dense(Array(2.0, 1.0, -0.0, -1.0, -2.0))

    (0.5 * vector) shouldBe Vectors.dense(Array(-1.0, -0.5, 0.0, 0.5, 1.0))

  }

  test("Multiply a vector and a scalar") {

    val vector = Vectors.dense(Array(-2.0, -1.0, 0.0, 1.0, 2.0))

    // Hmm... negative zero...
    (vector * 0.0) shouldBe Vectors.dense(Array(-0.0, -0.0, 0.0, 0.0, 0.0))

    (vector * 1.0) shouldBe vector

    (vector * -1.0) shouldBe Vectors.dense(Array(2.0, 1.0, -0.0, -1.0, -2.0))

    (vector * 0.5) shouldBe Vectors.dense(Array(-1.0, -0.5, 0.0, 0.5, 1.0))

  }

  test("Divide a scalar by a vector") {

    val vector = Vectors.dense(Array(-2.0, -1.0, 0.0, 1.0, 2.0))

    (0.0 / vector) shouldBe Vectors.dense(Array(-0.0, -0.0, Double.NaN, 0.0, 0.0))

    (1.0 / vector) shouldBe Vectors.dense(Array(-0.5, -1.0, Double.PositiveInfinity, 1.0, 0.5))

    (-1.0 / vector) shouldBe Vectors.dense(Array(0.5, 1.0, Double.NegativeInfinity, -1.0, -0.5))

    (0.5 / vector) shouldBe Vectors.dense(Array(-0.25, -0.5, Double.PositiveInfinity, 0.5, 0.25))

  }

  test("Divide a vector by a scalar") {

    val vector = Vectors.dense(Array(-2.0, -1.0, 0.0, 1.0, 2.0))

    // Hmm... negative zero...
    (vector / 0.0) shouldBe Vectors.dense(Array(Double.NegativeInfinity, Double.NegativeInfinity, Double.NaN, Double.PositiveInfinity, Double.PositiveInfinity))

    (vector / 1.0) shouldBe vector

    (vector / -1.0) shouldBe Vectors.dense(Array(2.0, 1.0, -0.0, -1.0, -2.0))

    (vector / 0.5) shouldBe Vectors.dense(Array(-4.0, -2.0, 0.0, 2.0, 4.0))

  }

  test("Negative vector") {

    val vector = Vectors.dense(Array(-2.0, -1.0, 0.0, 1.0, 2.0))

    // Hmm... negative zero...
    -vector shouldBe Vectors.dense(Array(2.0, 1.0, -0.0, -1.0, -2.0))

    -(-vector) shouldBe vector

  }

}
