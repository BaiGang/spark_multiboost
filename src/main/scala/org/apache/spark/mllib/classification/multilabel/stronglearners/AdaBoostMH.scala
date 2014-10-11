/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package org.apache.spark.mllib.classification.multilabel.stronglearners

import org.apache.spark.annotation.Experimental
import org.apache.spark.mllib.classification.multilabel.baselearners.{ BaseLearnerAlgorithm, BaseLearnerModel, DecisionStumpAlgorithm, DecisionStumpModel }
import org.apache.spark.mllib.classification.multilabel.{ GeneralizedAdditiveModel, MultiLabelClassificationModel }
import org.apache.spark.mllib.linalg.{ Vector, Vectors }
import org.apache.spark.mllib.util.{ MultiLabeledPoint, WeightedMultiLabeledPoint }
import org.apache.spark.rdd.RDD

import scala.language.higherKinds

/**
 *
 * @param numClasses
 * @param numFeatureDimensions
 * @param baseLearnersList
 * @tparam BM
 */
@Experimental
class AdaBoostMHModel[BM <: MultiLabelClassificationModel](
  numClasses: Int,
  numFeatureDimensions: Int,
  baseLearnersList: List[BM])
    extends StrongLearnerModel with GeneralizedAdditiveModel[BM] {

  var debugString: String = ""

  def this() = this(0, 0, List())

  override def models = baseLearnersList

  override def predict(testData: RDD[Vector]): RDD[Vector] = {
    testData map predict
  }

  override def predict(testData: Vector): Vector = {

    val rawPredicts: Vector = models.foldLeft(
      Vectors.dense(Array.fill[Double](numClasses)(0.0))) { (sum, item) =>
        val predicts = item predict testData
        Vectors.fromBreeze(sum.toBreeze + predicts.toBreeze)
      }

    val predictArray: Array[Double] = rawPredicts.toArray.map {
      case p: Double => if (p >= 0.0) 1.0 else -1.0
    }
    Vectors.dense(predictArray)
  }

  override def toString = models mkString ";\n"
}

@Experimental
object AdaBoostMHModel {
  def apply[BM <: MultiLabelClassificationModel](numClasses: Int,
    numFeatureDimensions: Int,
    baseLearnersList: List[BM]) = {
    new AdaBoostMHModel[BM](numClasses, numFeatureDimensions, baseLearnersList)
  }
}

@Experimental
class AdaBoostMHAlgorithm[BM <: BaseLearnerModel, BA <: BaseLearnerAlgorithm[BM]](
    baseLearnerAlgo: BA,
    _numClasses: Int,
    _numFeatureDimensions: Int,
    numIterations: Int) extends StrongLearnerAlgorithm[BM, BA, AdaBoostMHModel[BM]] {

  override def numClasses = _numClasses
  override def numFeatureDimensions = _numFeatureDimensions

  def run(dataSet: RDD[MultiLabeledPoint]): AdaBoostMHModel[BM] = {

    val weightedDataSet = AdaBoostMHAlgorithm.initWeights(numClasses, dataSet)

    /**
     * The encapsulation of the iteration data which consists of:
     * @param model the resulted model, a strong learner, of previous iterations.
     * @param dataSet the re-weighted multilabeled data points.
     */
    case class IterationData(model: AdaBoostMHModel[BM], dataSet: RDD[WeightedMultiLabeledPoint])

    val finalIterationData = (1 to numIterations).foldLeft(IterationData(
      AdaBoostMHModel.apply[BM](numClasses, numFeatureDimensions, List()),
      weightedDataSet)) { (iterData: IterationData, iter: Int) =>

      logInfo(s"Start $iter-th iteration.")

      // 1. train a new base learner
      val baseLearner = baseLearnerAlgo.run(iterData.dataSet)

      // 1.1 update strong learner
      val updatedStrongLearner = AdaBoostMHModel.apply[BM](
        numClasses, numFeatureDimensions, iterData.model.models :+ baseLearner)

      // 2. get the weak hypothesis
      val predictsAndPoints = iterData.dataSet map { wmlPoint =>
        (baseLearner.predict(wmlPoint.data.features), wmlPoint)
      }

      // 3. sum up the normalize factor
      val summedZ = predictsAndPoints.aggregate(0.0)({
        // seqOp
        case (sum: Double, (predict: Vector, wmlp: WeightedMultiLabeledPoint)) =>
          (predict.toArray zip wmlp.data.labels.toArray zip wmlp.weights.toArray)
            .map {
              case ((p, l), w) =>
                w * math.exp(-p * l)
            }.sum + sum
      }, { _ + _ })

      logInfo(s"Weights normalization factor (Z) value: $summedZ")
      updatedStrongLearner.debugString = iterData.model.debugString + s"\nZ=$summedZ"

      // XXX: should be using multi-label metrics in mllib.
      // 3.1 hamming loss
      val strongPredictsAndLabels = iterData.dataSet.map { wmlp =>
        (updatedStrongLearner.predict(wmlp.data.features), wmlp.data.labels)
      }
      val hammingLoss = strongPredictsAndLabels.flatMap {
        case (predict, label) =>
          predict.toArray zip label.toArray
      }.filter {
        case (p, l) =>
          p * l < 0.0
      }.count.toDouble / (predictsAndPoints.count * numClasses).toDouble

      logInfo(s"Iter $iter. Hamming loss: $hammingLoss")
      updatedStrongLearner.debugString = iterData.model.debugString + s"\nIter $iter. Hamming loss: $hammingLoss"

      // 4. re-weight the data set
      val reweightedDataSet = predictsAndPoints map {
        case (predict: Vector, wmlp: WeightedMultiLabeledPoint) =>
          val updatedWeights = for (i <- 0 until numClasses)
            yield wmlp.weights(i) * math.exp(-predict(i) * wmlp.data.labels(i)) / summedZ
          WeightedMultiLabeledPoint(
            Vectors.dense(updatedWeights.toArray), wmlp.data)
      }

      // 5. next recursion
      IterationData(updatedStrongLearner, reweightedDataSet)
    }
    finalIterationData.model
  }
}

object AdaBoostMHAlgorithm {

  /**
   * Calculate the initial weight of the dataset.
   *
   * @param numClasses Num of class labels.
   * @param dataSet The dataset RDD;
   * @return A RDD of WeightedMultiLabeledPoint with initial weights in it.
   */
  def initWeights(
    numClasses: Int,
    dataSet: RDD[MultiLabeledPoint]): RDD[WeightedMultiLabeledPoint] = {
    val w = 1.0 / (dataSet.count().toDouble * numClasses)
    val initialWeight = Vectors.dense(
      Array.fill[Double](numClasses)(w))
    dataSet map {
      case mlPoint: MultiLabeledPoint =>
        WeightedMultiLabeledPoint(initialWeight, mlPoint)
    }
  }
}

