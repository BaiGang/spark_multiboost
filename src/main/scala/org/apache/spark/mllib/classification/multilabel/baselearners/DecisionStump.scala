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
    featureRate: Double = 1.0) extends BaseLearnerAlgorithm[DecisionStumpModel]
with Serializable with Logging {
  require(sampleRate > 0.0 && sampleRate <= 1.0, s"sampleRate $sampleRate is out of range.")
  require(featureRate > 0.0 && featureRate <= 1.0, s"feature downSample Rate $featureRate is out of range.")
  override def numClasses = _numClasses
  override def numFeatureDimensions = _numFeatureDimensions

  /**
   * Train the DecisionStumpModel.
   * @param dataSet The data set.
   * @param seed The seed for the random sampler.
   * @return
   */
  override def run(dataSet: RDD[WeightedMultiLabeledPoint], seed: Long = 0): DecisionStumpModel = {
    // 0. do sub-sampling
    val sampledDataSet = dataSet.sample(false, sampleRate, seed).cache()
    // 1. class-wise edge
    val classWiseEdges = sampledDataSet.aggregate(
      Vectors.dense(Array.fill[Double](numClasses)(0.0)))({
        // the seqOp
        (edge, point) =>
          Vectors.dense({
            for (l <- 0 until numClasses)
              yield edge(l) + point.weights(l) * point.data.labels(l)
          }.toArray)
      }, {
        // the combOp
        (edge1, edge2) =>
          Vectors.fromBreeze(edge1.toBreeze + edge2.toBreeze)
      })

    // 2. for each feature, select the best split
    // 2.1 select a subset of feature indices
    val bernoulliSampler = new BernoulliSampler[Int](featureRate)
    bernoulliSampler.setSeed(seed)
    val selectedFeatureIndices = bernoulliSampler.sample(Iterator.range(0, numFeatureDimensions))

    val (totalEnergy, bestFeature, bestThreshold, alpha: Double, votesVec) = {
      for {
        featureIndex <- selectedFeatureIndices

        // TODO: use sortBy[K] instead of a series of operations
        sortedFeatDataSet = sampledDataSet
          .keyBy[Double](wmlp => wmlp.data.features(featureIndex))
          .sortByKey(ascending = true, numPartitions = sampledDataSet.partitions.size)
          .values
        // sortedFeatDataSet = sampledDataSet.sortBy(wmlp => wmlp.data.features(featureIndex))

        (votes, threshold, edge) = DecisionStumpAlgorithm.findBestStumpOnFeature(
          sortedFeatDataSet,
          featureIndex,
          classWiseEdges)

        edgeNorm = edge.toArray.reduce(math.abs(_) + math.abs(_))
        alpha = 0.5 * math.log((1.0 + edgeNorm) / (1.0 - edgeNorm))
        energy = math.sqrt(1.0 - edge.toArray.foldLeft(0.0) { (s, e) => s + e * e })
      } yield (energy, featureIndex, threshold, alpha, votes)
    }.foldLeft((1e20, -1, -1e20, -1e20, Vectors.dense(1.0))) {
      case (accumulativeTuple: (Double, Int, Double, Double, Vector),
        candidateTuple: (Double, Int, Double, Double, Vector)) =>
        if (candidateTuple._1 < accumulativeTuple._1) {
          candidateTuple
        } else {
          accumulativeTuple
        }
    }

    logInfo("Best feature [" + bestFeature + "], split value [" + bestThreshold
      + "], with energy [" + totalEnergy + "] ")

    // 3. generate the model
    new DecisionStumpModel(
      alpha,
      votesVec,
      new FeatureCut(bestFeature, bestThreshold))
  }
}

object DecisionStumpAlgorithm {

  def apply(numClasses: Int, numFeatureDimensions: Int) = {
    new DecisionStumpAlgorithm(numClasses, numFeatureDimensions)
  }

  /**
   * Given a feature index, find the best split threshold on this feature.
   * N.B. Currently we need the data set sorted on the feature. This will be
   * prohibitively inefficient. Keep it for now. And I will optimize and
   * fix this after I set up the learning framework.
   *
   * @param sortedFeatDataSet The sorted data set on this feature.
   * @param featureIndex The feature index.
   * @param classWiseEdges The initial edge for each class label.
   * @return (votes:Vector, thresh:Double, edge:Vector)
   */
  def findBestStumpOnFeature(
    sortedFeatDataSet: RDD[WeightedMultiLabeledPoint],
    featureIndex: Int,
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

    val bestStump = sortedFeatDataSet.aggregate(
      AccBestStumpData(
        classWiseEdges,
        classWiseEdges,
        -1e20,
        -1e19))({
        // the seqOp
        case (acc: AccBestStumpData, wmlPoint: WeightedMultiLabeledPoint) =>

          val updatedEdge = Vectors.dense({
            for (i <- 0 until acc.accumEdge.size)
              yield acc.accumEdge(i) - 2.0 * wmlPoint.weights(i) * wmlPoint.data.labels(i)
          }.toArray)

          if (acc.preFeatureVal == -1e19) {
            // initial
            AccBestStumpData(
              acc.bestEdge,
              updatedEdge,
              -1e20,
              wmlPoint.data.features(featureIndex))
          } else {
            // update the threshold if the new edge on featureIndex is better
            if (acc.preFeatureVal != wmlPoint.data.features(featureIndex)
              && acc.accumEdge.toArray.reduce(math.abs(_) + math.abs(_))
              > acc.bestEdge.toArray.reduce(math.abs(_) + math.abs(_))) {
              AccBestStumpData(
                acc.accumEdge,
                updatedEdge,
                0.5 * (acc.preFeatureVal + wmlPoint.data.features(featureIndex)),
                wmlPoint.data.features(featureIndex))
            } else {
              AccBestStumpData(
                acc.bestEdge,
                updatedEdge,
                acc.bestThresh,
                wmlPoint.data.features(featureIndex))
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
