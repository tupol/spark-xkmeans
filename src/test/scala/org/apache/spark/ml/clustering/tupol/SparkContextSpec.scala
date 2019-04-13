package org.apache.spark.ml.clustering.tupol

import org.apache.spark.sql.SQLContext
import org.apache.spark.{ SparkConf, SparkContext }
import org.scalatest.{ BeforeAndAfterAll, Suite }

/**
 * Shares a SparkContext for all the tests in the Suite.
 */
trait SparkContextSpec extends BeforeAndAfterAll {
  this: Suite =>

  private val master = "local[*]"
  private val appName = this.getClass.getSimpleName

  @transient private var _sc: SparkContext = _
  @transient private var _sqlContext: SQLContext = _

  def sc: SparkContext = _sc

  def sqlContext: SQLContext = _sqlContext

  val conf: SparkConf = new SparkConf()
    .setMaster(master)
    .setAppName(appName)

  override def beforeAll(): Unit = {
    System.clearProperty("spark.driver.port")
    System.clearProperty("spark.hostPort")
    super.beforeAll()
    _sc = new SparkContext(conf)
    _sqlContext = new SQLContext(_sc)
  }

  override def afterAll(): Unit = {
    try {
      if (_sqlContext != null) {
        _sqlContext = null
      }
      if (_sc != null) {
        _sc.stop()
        _sc = null
      }
    } finally {
      super.afterAll()
      System.clearProperty("spark.driver.port")
      System.clearProperty("spark.hostPort")
    }
  }

}
