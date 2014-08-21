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

package org.apache.spark.mllib.classification.multilabel.baselearners

import org.apache.spark.mllib.classification.multilabel.{ WeightedMultiLabeledPoint, MultiLabeledPoint }
import org.apache.spark.mllib.linalg.{ Vectors, Vector }
import org.apache.spark.SparkContext._
import org.apache.spark.annotation.Experimental
import org.apache.spark.rdd.RDD
import org.apache.spark.Logging
import org.apache.spark.util.random.BernoulliSampler

@Experimental
case class FeatureCut(
    featureIndex: Int,
    decisionThreshold: Double) extends Serializable {
  def cut(point: MultiLabeledPoint): Double = cut(point.features)

  def cut(weightedPoint: WeightedMultiLabeledPoint): Double = {
    cut(weightedPoint.data.features)
  }

  def cut(features: Vector): Double = {
    if (features(featureIndex) > decisionThreshold) 1.0 else -1.0
  }

  override def toString = "FeatureCut(" + featureIndex + "," + decisionThreshold + ")"
}

@Experimental
class DecisionStumpModel(
    alpha: Double,
    votes: Vector,
    cut: FeatureCut) extends BaseLearnerModel {
  override def predict(features: Vector): Vector = {
    Vectors.fromBreeze(votes.toBreeze * alpha * cut.cut(features))
  }
  override def predict(features: RDD[Vector]): RDD[Vector] = {
    features map predict
  }
  override def toString = {
    "DecisionStumpModel, alpha = " + alpha + ", " + " votes = " + votes + ", feature cut: " + cut
  }
}

@Experimental
class DecisionStumpAlgorithm(
    _numClasses: Int,
    _numFeatureDimensions: Int,
    sampleRate: Double = 0.3,
    featureRate: Double = 1.0)
  extends BaseLearnerAlgorithm[DecisionStumpModel]
  with Serializable with Logging {

  require(sampleRate > 0.0 && sampleRate <= 1.0,
    s"sampleRate $sampleRate is out of range.")
  require(featureRate > 0.0 && featureRate <= 1.0,
    s"feature downSample Rate $featureRate is out of range.")

  override def numClasses = _numClasses
  override def numFeatureDimensions = _numFeatureDimensions

  override def run(dataSet: RDD[Iterable[WeightedMultiLabeledPoint]], seed: Long = 0): DecisionStumpModel = {

    val allSplitMetrics = dataSet flatMap DecisionStumpAlgorithm.getLocalSplitMetrics

    DecisionStumpAlgorithm.findBestSplitMetrics(allSplitMetrics)
  }

}

object DecisionStumpAlgorithm {

  def apply(numClasses: Int, numFeatureDimensions: Int) = {
    new DecisionStumpAlgorithm(numClasses, numFeatureDimensions)
  }

  /**
   * The data abstraction of feature split metrics.
   * @param featureCut the feature and the split value of the stump
   * @param votes the label-dependent mapping of the stump
   * @param edge the full multo-class edge of the stump
   */
  private[DecisionStumpAlgorithm] case class SplitMetric(featureCut: FeatureCut, votes: Vector, edge: Double)

  def getLocalSplitMetrics(dataSet: Iterable[WeightedMultiLabeledPoint]): Iterable[SplitMetric] = {

    // TODO: implementation
    Iterable(new SplitMetric(new FeatureCut(1, 0.5), Vectors.dense(1.0), 1.0))
  }

  /**
   * Find the best stump split to minimize the loss given all split candidates.
   * @param splitMetrics the collection of split candidates and corresponding edges
   * @return the best feature cut
   */
  def findBestSplitMetrics(splitMetrics: RDD[SplitMetric]): DecisionStumpModel = {

    val (model, loss) = splitMetrics.aggregate(
      (new DecisionStumpModel(1.0, Vectors.dense(1.0), new FeatureCut(1, 0.5)), 1e20))({ (result, item) =>
        val alpha = getAlpha(item.edge)
        val loss = getExpLoss(alpha, item.edge)
        if (loss < result._2)
          (new DecisionStumpModel(alpha, item.votes, item.featureCut), loss)
        else
          result
      }, { (result1, result2) =>
        // comOp, choose the one with smaller loss
        if (result1._2 < result2._2)
          result1
        else
          result2
      }
      )
    model
  }

  /**
   * Choose alpha value.
   * @param gamma the edge value of the base learner
   * @return the base learner factor
   */
  def getAlpha(gamma: Double) = 0.5 * math.log((1.0 + gamma) / (1.0 - gamma))

  /**
   * Get the exponential loss of the base learner.
   * Mathematically according to [Kegl2013] Appendix A.
   * @param alpha the base learner factor
   * @param edge the edge value of the base learner
   * @return the loss Z(h,W)
   */
  def getExpLoss(alpha: Double, edge: Double) = {
    val expPlus = math exp alpha
    val expMinus = 1.0 / expPlus
    0.5 * (expPlus + expMinus - edge * (expPlus - expMinus))
  }

  /**
   * DEPRECATED
   * Given a feature index, find the best split threshold on this feature.
   * N.B. Currently we need the data set sorted on the feature. This will be
   * prohibitively inefficient. Keep it for now. And I will optimize and
   * fix this after I set up the learning framework.
   *
   * @param sortedFeatLabelSet The data set of (feat, label, weight), sorted on feature.
   * @param classWiseEdges The initial edge for each class label.
   * @return (votes:Vector, thresh:Double, edge:Vector)
   */
  def findBestStumpOnFeature(
    sortedFeatLabelSet: RDD[(Double, Vector, Vector)],
    classWiseEdges: Vector): (Vector, Double, Vector) = {

    /**
     * The accumulated best stump data.
     *
     * @param bestEdge The best edge vector.
     * @param accumEdge The updated edge vector of stump with v=1
     * @param bestThresh The best threshold of the feature cut.
     * @param preFeatureVal Feature value of the previous point.
     */
    case class AccBestStumpData(
        bestEdge: Vector,
        accumEdge: Vector,
        bestThresh: Double,
        preFeatureVal: Double) {
      override def toString = bestEdge + ", " + accumEdge + ", " + bestThresh + ", " + preFeatureVal
    }

    val bestStump = sortedFeatLabelSet.aggregate(
      AccBestStumpData(
        classWiseEdges,
        classWiseEdges,
        -1e20,
        -1e19))({
        // the seqOp
        case (acc: AccBestStumpData, featLabelWeightTriplet: (Double, Vector, Vector)) =>

          val updatedEdge = Vectors.dense({
            for (i <- 0 until acc.accumEdge.size)
              yield acc.accumEdge(i) -
              2.0 * featLabelWeightTriplet._2(i) * featLabelWeightTriplet._3(i)
          }.toArray)

          if (acc.preFeatureVal == -1e19) {
            // initial
            AccBestStumpData(
              acc.bestEdge,
              updatedEdge,
              -1e20,
              featLabelWeightTriplet._1)
          } else {
            // update the threshold if the new edge on featureIndex is better
            if (acc.preFeatureVal != featLabelWeightTriplet._1
              && acc.accumEdge.toArray.reduce(math.abs(_) + math.abs(_))
              > acc.bestEdge.toArray.reduce(math.abs(_) + math.abs(_))) {
              AccBestStumpData(
                acc.accumEdge,
                updatedEdge,
                0.5 * (acc.preFeatureVal + featLabelWeightTriplet._1),
                featLabelWeightTriplet._1)
            } else {
              AccBestStumpData(
                acc.bestEdge,
                updatedEdge,
                acc.bestThresh,
                featLabelWeightTriplet._1)
            }
          }
      }, {
        // the combOp
        case (acc1: AccBestStumpData, acc2: AccBestStumpData) =>
          val edgeSum1 = acc1.bestEdge.toArray.reduce(math.abs(_) + math.abs(_))
          val edgeSum2 = acc2.bestEdge.toArray.reduce(math.abs(_) + math.abs(_))
          if (edgeSum1 > edgeSum2) acc1 else acc2
      })

    val votesArray = (bestStump.bestEdge.toArray map {
      case edge: Double =>
        if (edge > 0.0) 1.0 else -1.0
    }
    ).toArray

    (Vectors.dense(votesArray),
      bestStump.bestThresh,
      bestStump.bestEdge)
  }
}
