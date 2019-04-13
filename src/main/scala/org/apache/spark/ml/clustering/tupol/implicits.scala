package org.apache.spark.ml.clustering.tupol

import org.apache.spark.ml.util.Identifiable
import org.apache.spark.ml.{ PipelineModel, Transformer }

/**
 *
 */
package object implicits {

  /**
   * PipelineModel implicit decorator
   *
   * @param pipelineModel
   */
  implicit class PipelineModelOps(val pipelineModel: PipelineModel) extends AnyVal {

    /**
     * Rename the PipelineModel (essentially changing the uid)
     *
     * @param uid
     * @return
     */
    def withUid(uid: String): PipelineModel =
      new PipelineModel(uid, pipelineModel.stages)

    /**
     * Rename the PipelineModel (essentially changing the uid prefix)
     *
     * @param prefix
     * @return
     */
    def withUidPrefix(prefix: String): PipelineModel =
      new PipelineModel(Identifiable.randomUID(prefix), pipelineModel.stages)

    /**
     * Append one more transformer to stages.
     *
     * @param transformer
     * @return
     */
    def appendStage(transformer: Transformer): PipelineModel =
      new PipelineModel(pipelineModel.uid, pipelineModel.stages :+ transformer)

    /**
     * Append more transformers to stages.
     *
     * @param transformers
     * @return
     */
    def appendStages(transformers: Transformer*): PipelineModel =
      new PipelineModel(pipelineModel.uid, pipelineModel.stages ++ transformers)

    /**
     * Append the stages of the `otherPipelineModel`.
     *
     * @param otherPipelineModel
     * @return
     */
    def appendPipeline(otherPipelineModel: PipelineModel): PipelineModel =
      new PipelineModel(pipelineModel.uid, pipelineModel.stages ++ otherPipelineModel.stages)

    /**
     * Append more pipelines into this pipeline by appending the stages of each one.
     *
     * @param pipelineModels
     * @return
     */
    def appendPipelines(pipelineModels: PipelineModel*): PipelineModel =
      new PipelineModel(
        pipelineModel.uid,
        pipelineModel.stages ++ pipelineModels.foldLeft(Array[Transformer]())((acc, model) => acc ++ model.stages))

  }

}
