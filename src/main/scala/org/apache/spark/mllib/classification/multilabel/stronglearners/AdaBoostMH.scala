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

import scala.language.higherKinds
import org.apache.spark.SparkContext._
import org.apache.spark.mllib.classification.multilabel.{ MultiLabelClassificationAlgorithm, MultiLabelClassificationModel, MultiLabeledPoint, WeightedMultiLabeledPoint }
import org.apache.spark.rdd.RDD
import org.apache.spark.mllib.linalg.{ Vectors, Vector }
import org.apache.spark.annotation.Experimental
import org.apache.spark.mllib.classification.multilabel.baselearners.{ DecisionStumpAlgorithm, DecisionStumpModel, BaseLearnerModel, BaseLearnerAlgorithm }

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
    extends StrongLearnerModel[BM] {

  def this() = this(0, 0, List())

  override def baseLearners = baseLearnersList

  override def predict(testData: RDD[Vector]): RDD[Vector] = {
    testData map predict
  }

  override def predict(testData: Vector): Vector = {
    baseLearners.map {
      case baseLearner =>
        baseLearner.predict(testData)
    }

    val rawPredicts: Vector = baseLearners.foldLeft(
      Vectors.dense(Array.fill[Double](numClasses)(0.0))) {
        (sum, item) =>
          val predicts = item predict testData
          Vectors.fromBreeze(sum.toBreeze + predicts.toBreeze)
      }

    val predictArray: Array[Double] = rawPredicts.toArray.map {
      case p: Double =>
        if (p >= 0.0) 1.0 else -1.0
    }
    Vectors.dense(predictArray)
  }

  override def toString = baseLearners mkString ";\n"
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

    def accumBoosting(
      accumBaseLearners: List[BM],
      dataSet: RDD[WeightedMultiLabeledPoint],
      itersRemained: Int): List[BM] = {
      if (itersRemained == 0) {
        accumBaseLearners
      } else {
        // 1. train a new base learner
        val baseLearner = baseLearnerAlgo.run(dataSet)

        // 2. get the hypothesis
        val predictsAndPoints = dataSet map { wmlPoint =>
          (baseLearner.predict(wmlPoint.data.features),
            wmlPoint)
        }

        // 3. sum up the normalize factor
        val summedZ = predictsAndPoints.flatMap {
          case (predicts, wmlPoint) =>
            predicts.toArray zip wmlPoint.data.labels.toArray zip wmlPoint.weights.toArray
        }.map {
          case ((p, l), w) =>
            w * math.exp(-p * l)
        }.sum

        // 4. re-weight the data set
        val reweightedDataSet = predictsAndPoints map {
          case (predicts, wmlPoint) =>
            val updatedWeights = for (i <- 0 until numClasses)
              yield wmlPoint.weights(i) * math.exp(-predicts(i) * wmlPoint.data.labels(i)) / summedZ
            WeightedMultiLabeledPoint(
              Vectors.dense(updatedWeights.toArray),
              wmlPoint.data)
        }

        // 5. next recursion
        accumBoosting(baseLearner :: accumBaseLearners, reweightedDataSet, itersRemained - 1)
      }
    }

    AdaBoostMHModel.apply[BM](numClasses, numFeatureDimensions,
      accumBoosting(List(), weightedDataSet, numIterations))
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
    val initialWeight = Vectors.dense(
      Array.fill[Double](numClasses)(
        1.0 / (numClasses * dataSet.count())))
    dataSet map {
      case mlPoint: MultiLabeledPoint =>
        WeightedMultiLabeledPoint(initialWeight, mlPoint)
    }
  }
}

object AdaBoostMH {
  /**
   * Train an AdaBoost.MH model with DecisionStump as base learner.
   *
   */
  def train(
    dataSet: RDD[MultiLabeledPoint],
    numIters: Int): AdaBoostMHModel[DecisionStumpModel] = {
    val numClasses = dataSet.take(1)(0).labels.size
    val numFeatureDimensions = dataSet.take(1)(0).features.size

    val decisionStumpAlgo = new DecisionStumpAlgorithm(numClasses, numFeatureDimensions)
    val adaboostMHAlgo = new AdaBoostMHAlgorithm[DecisionStumpModel, DecisionStumpAlgorithm](
      decisionStumpAlgo,
      numClasses,
      numFeatureDimensions,
      numIters)

    adaboostMHAlgo.run(dataSet)
  }
}
