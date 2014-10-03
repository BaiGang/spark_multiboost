

package org.apache.spark.mllib.classification.multilabel.baselearners

import org.apache.spark.annotation.Experimental
import org.apache.spark.mllib.classification._
import org.apache.spark.mllib.linalg.{ Vectors, Vector }
import org.apache.spark.mllib.regression.{ GeneralizedLinearAlgorithm, LabeledPoint }
import org.apache.spark.mllib.util.WeightedMultiLabeledPoint
import org.apache.spark.rdd.RDD

@Experimental
trait BinaryClassificationModel {
  def predictPoint(testData: Vector): Double
}

@Experimental
trait BinaryClassificationAlgorithm[M <: BinaryClassificationModel] {
  def run(dataSet: RDD[LabeledPoint]): M
}

@Experimental
class GeneralizedBinaryBaseLearnerModel[BCM <: BinaryClassificationModel](
    alpha: Double,
    votes: Vector,
    binaryClassifier: BCM) extends BaseLearnerModel {

  def predict(testData: RDD[Vector]): RDD[Vector] = {
    testData.map(predict)
  }

  def predict(testData: Vector): Vector = {
    val binPredict = binaryClassifier.predictPoint(testData) * 2.0 - 1.0
    Vectors.fromBreeze(votes.toBreeze * alpha * binPredict)
  }
}

class GeneralizedBinaryBaseLearnerAlgorithm[BCM <: BinaryClassificationModel, BCA <: BinaryClassificationAlgorithm[BCM]](
    _numFeatureDimensions: Int,
    _numClasses: Int,
    binaryClassificationAlgo: BCA) extends BaseLearnerAlgorithm[GeneralizedBinaryBaseLearnerModel[BCM]] {

  def numFeatureDimensions = _numFeatureDimensions
  def numClasses = _numClasses

  def run(dataSet: RDD[WeightedMultiLabeledPoint]): GeneralizedBinaryBaseLearnerModel[BCM] = {
    val convertedBinaryDataset = dataSet map { wmlp =>
      val weightedSumLabel = (wmlp.weights.toArray zip wmlp.data.labels.toArray)
        .map(pair => pair._1 * pair._2)
        .reduce(_ + _)
      val labelSign = if (weightedSumLabel > 0.0) 1.0 else 0.0
      LabeledPoint(labelSign, wmlp.data.features)
    }
    val bc = binaryClassificationAlgo.run(convertedBinaryDataset)

    val edges = dataSet.map { wmlp =>
      (wmlp.weights, wmlp.data.labels, 2.0 * bc.predictPoint(wmlp.data.features) - 1.0)
    }.map {
      case (weights, labels, binPredict) =>
        for (i <- 0 until numClasses) yield binPredict * weights(i) * labels(i)
    }.reduce { (a, b) =>
      for (i <- 0 until numClasses) yield a(i) + b(i)
    }
    val votes = edges.map(s => if (s > 0.0) 1.0 else -1.0).toArray

    val fullEdge = edges.reduce((a, b) => math.abs(a) + math.abs(b))
    val alpha = 0.5 * math.log((1.0 + fullEdge) / (1.0 - fullEdge))

    new GeneralizedBinaryBaseLearnerModel[BCM](alpha, Vectors.dense(votes), bc)
  }

  def run(dataSet: RDD[Array[WeightedMultiLabeledPoint]], seed: Long): GeneralizedBinaryBaseLearnerModel[BCM] = {
    throw new NotImplementedError(s"Aggregated ``run'' is not implemented for ${this.getClass}.")
  }
}

class LRClassificationModel(lrModel: LogisticRegressionModel)
    extends BinaryClassificationModel {

  def predictPoint(testData: Vector): Double = {
    lrModel.predict(testData)
  }
}

class LRClassificationAlgorithm(lrSGD: LogisticRegressionWithSGD)
    extends BinaryClassificationAlgorithm[LRClassificationModel] {

  def run(dataSet: RDD[LabeledPoint]): LRClassificationModel = {
    new LRClassificationModel(lrSGD.run(dataSet))
  }
}

class SVMClassificationModel(svmModel: SVMModel)
    extends BinaryClassificationModel {

  def predictPoint(testData: Vector): Double = {
    svmModel.predict(testData)
  }
}

class SVMClassificationAlgorithm(svmSGD: SVMWithSGD)
    extends BinaryClassificationAlgorithm[SVMClassificationModel] {

  def run(dataSet: RDD[LabeledPoint]): SVMClassificationModel = {
    new SVMClassificationModel(svmSGD.run(dataSet))
  }
}