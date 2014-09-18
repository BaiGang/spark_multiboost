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

import org.apache.spark.mllib.linalg.{ Vectors, Vector }
import org.apache.spark.SparkContext._
import org.apache.spark.annotation.Experimental
import org.apache.spark.mllib.util.{ MultiLabeledPoint, WeightedMultiLabeledPoint }
import org.apache.spark.rdd.RDD

import org.apache.spark.Logging
import org.apache.spark.util.random.BernoulliSampler
import scala.collection.mutable.ArrayBuffer

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
case class DecisionStumpModel(
    alpha: Double,
    votes: Vector,
    cut: Option[FeatureCut]) extends BaseLearnerModel {
  override def predict(features: Vector): Vector = {
    Vectors.fromBreeze(votes.toBreeze * alpha * {
      cut match {
        case Some(featureCut) => featureCut.cut(features)
        case None => 1.0
      }
    })
  }
  override def predict(features: RDD[Vector]): RDD[Vector] = {
    features map predict
  }
  override def toString = {
    s"($alpha,$votes,$cut)"
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

  override def run(dataSet: RDD[Array[WeightedMultiLabeledPoint]], seed: Long = 0): DecisionStumpModel = {

    val bernoulliSampler = new BernoulliSampler[Int](featureRate)
    bernoulliSampler setSeed seed

    val allSplitMetrics = dataSet flatMap (DecisionStumpAlgorithm.getLocalSplitMetrics(
      bernoulliSampler sample Iterator.range(0, numFeatureDimensions))(_))

    DecisionStumpAlgorithm findBestSplitMetrics (
      DecisionStumpAlgorithm aggregateSplitMetrics allSplitMetrics)
  }

}

object DecisionStumpAlgorithm {

  def apply(numClasses: Int, numFeatureDimensions: Int) = {
    new DecisionStumpAlgorithm(numClasses, numFeatureDimensions)
  }

  /**
   * The data abstraction of feature split metrics.
   * @param featureCut the feature and the split value of the stump
   * @param edges the vector of class-wise edges of the stump
   */
  case class SplitMetric(featureCut: Option[FeatureCut], edges: Vector)

  def getLocalSplitMetrics(featureSet: Iterator[Int])(
    dataSet: Array[WeightedMultiLabeledPoint]): Iterator[SplitMetric] = {

    val numClasses = dataSet(1).data.labels.size
    // 1. initial edge
    val initialEdge = dataSet.foldLeft(Array.fill(numClasses)(0.0)) { (edge, wmlp) =>
      (for (l <- 0 until numClasses) yield edge(l) + wmlp.weights(l) * wmlp.data.labels(l)).toArray
    }

    featureSet.flatMap { featureIndex =>
      // 2. fold to calculate split metrics on each split value
      dataSet.sortBy(_.data.features(featureIndex)).foldLeft(
        new ArrayBuffer[SplitMetric]() += SplitMetric(None, Vectors.dense(initialEdge))
      ) { (metrics, wmlp) =>
          val lastMetric = metrics.last
          val updatedEdge = Vectors.dense((for (i <- 0 until wmlp.weights.size)
            yield lastMetric.edges(i) - 2.0 * wmlp.weights(i) * wmlp.data.labels(i)).toArray)

          lastMetric.featureCut match {
            case Some(cut) if math.abs(cut.decisionThreshold - wmlp.data.features(featureIndex)) < 1e-6 =>
              // update the edge, do not insert new split but update the last one
              val updatedMetric = SplitMetric(lastMetric.featureCut, updatedEdge)
              metrics.dropRight(1) += updatedMetric
            case _ =>
              // insert a new split candidate
              metrics += SplitMetric(Some(FeatureCut(featureIndex, wmlp.data.features(featureIndex))), updatedEdge)
          }
        }
    }
  }

  /**
   * For each split, aggregate the edges from different data partitions.
   * @param allSplitMetrics RDD of all split candidates and the corresponding metrics
   * @return the summed metrics for each split candidate
   */
  def aggregateSplitMetrics(allSplitMetrics: RDD[SplitMetric]): RDD[SplitMetric] = {

    // votes are combined as sigma(v * edge)/sigma(edge)
    allSplitMetrics.map {
      case metric: SplitMetric =>
        (metric.featureCut.hashCode, metric)
    }.reduceByKey { (metric1, metric2) =>
      SplitMetric(metric1.featureCut, Vectors.fromBreeze(metric1.edges.toBreeze + metric2.edges.toBreeze))
    }.map { case (hash: Int, metric: SplitMetric) => metric }
  }

  /**
   * Find the best stump split to minimize the loss given all split candidates.
   * @param splitMetrics the collection of split candidates and corresponding edges
   * @return the best feature cut
   */
  def findBestSplitMetrics(splitMetrics: RDD[SplitMetric]): DecisionStumpModel = {

    val (model, _) = splitMetrics.aggregate(
      (new DecisionStumpModel(1.0, Vectors.dense(1.0), None), 1e20))({ (result, item) =>
        val fullEdge = item.edges.toArray.reduce(math.abs(_) + math.abs(_))
        val alpha = getAlpha(fullEdge)
        val votes = Vectors.dense((for (e <- item.edges.toArray) yield if (e > 0.0) 1.0 else -1.0).toArray)
        val loss = getExpLoss(alpha, fullEdge)
        if (loss < result._2) (DecisionStumpModel(alpha, votes, item.featureCut), loss) else result
      }, { (result1, result2) =>
        // comOp, choose the one with smaller loss
        if (result1._2 < result2._2) result1 else result2
      })
    model
  }

  /**
   * Choose alpha value, which is the base learner factor.
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

}
