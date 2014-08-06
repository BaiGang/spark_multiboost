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

import org.scalatest.FunSuite
import org.apache.spark.mllib.util.LocalSparkContext
import org.apache.spark.mllib.classification.multilabel.MultiLabeledPointParser
import org.apache.spark.mllib.classification.multilabel.WeightedMultiLabeledPoint
import org.apache.spark.mllib.classification.multilabel.MultiLabeledPoint
import org.apache.spark.mllib.linalg.Vectors

class DecisionStumpSuite extends FunSuite with LocalSparkContext {

  lazy val dataSet = {
    sc.parallelize(Array(
      "([0.0625,0.0625],([1.0,-1.0],[0.7,0.3,0.9]))",
      "([0.0625,0.0625],([1.0,-1.0],[0.4,0.6,1.3]))",
      "([0.0625,0.0625],([-1.0,-1.0],[1.3,0.6,0.4]))",
      "([0.0625,0.0625],([1.0,-1.0],[0.3,0.6,0.5]))",
      "([0.0625,0.0625],([1.0,-1.0],[0.9,0.6,1.3]))",
      "([0.0625,0.0625],([-1.0,1.0],[1.4,0.8,1.3]))",
      "([0.0625,0.0625],([-1.0,1.0],[1.2,0.8,2.2]))",
      "([0.0625,0.0625],([-1.0,1.0],[1.3,1.2,0.5]))"
    )).map(MultiLabeledPointParser.parseWeighted).cache()
  }

  test("Test FeatureCut response.") {
    val featureCut = new FeatureCut(1, 0.5)
    val multiLabeledPoint = MultiLabeledPoint(
      Vectors.dense(1.0, -1.0),
      Vectors.dense(0.5, 1.0, 0.8))
    assert(featureCut.cut(multiLabeledPoint) === 1.0)

    val weightedMultiLabelPoint = WeightedMultiLabeledPoint(
      Vectors.dense(0.3, 0.7),
      MultiLabeledPoint(
        Vectors.dense(1.0, -1.0),
        Vectors.dense(4.1, 0.2, 2.0)))
    assert(featureCut.cut(weightedMultiLabelPoint) === -1.0)
  }

  test("Find best split threshold on a given feature.") {

    val (votes, threshold, edge) = DecisionStumpAlgorithm.findBestStumpOnFeature(
      dataSet, 1, Vectors.dense(0.0, -0.125))

    // In dataSet, there is an obvious gap in feature 1. So threshold is optimally 0.7.
    // The weighted per-class error rate is:
    //  mu_1_minus = 0.0625 * 7
    //  mu_2_minus = 0.0625 * 0
    // The weighted per-class correct classification rate is:
    //  mu_1_plus = 0.0625 * 1
    //  mu_2_plus = 0.0625 * 8
    // So vote_1 is -1 because mu_1_minus > mu_1_plus,
    // and vote_2 is 1 because mu_2_minus < mu_2_plus.
    // As for the edge:
    //  1. The initial edge is:
    //    edge_1^0 = 0.0625 * (4 - 4) = 0.0
    //    edge_2^0 = 0.0625 * (3 - 5) = -0.125
    //  2. class-wise edges for each sample is:
    //    0. (0.0, -0.125)
    //    1. (-0.125, 0.0)
    //    2. (-0.25, 0.125)
    //    3. (-0.125, 0.25)
    //    4. (-0.25, 0.375)
    //    5. (-0.375, 0.5)  * best stump
    //    6. (-0.25, 0.375)
    //    7. (-0.125, 0.25)
    //    8. (0.0, 0.125)
    //  3. best edge for each sample is:
    //    (-0.375, 0.5)
    //
    assert(votes === Vectors.dense(-1.0, 1.0))
    assert(threshold === 0.7)
    assert(edge === Vectors.dense(-0.375, 0.5))
  }

  test("Test training a very basic decision stump model.") {
    val decisionStumpAlgo = new DecisionStumpAlgorithm(2, 3)
    val decisionStumpModel: DecisionStumpModel = decisionStumpAlgo.run(dataSet)

    // TODO: use multi-label metrics in PR https://github.com/apache/spark/pull/1270
    // Here we just use a very simple miss-prediction count for metrics.

    val predicts = dataSet.flatMap {
      case wmlPoint: WeightedMultiLabeledPoint =>
        (decisionStumpModel predict wmlPoint.data.features).toArray
    }.collect
    val labels = dataSet.flatMap {
      case wmlPoint: WeightedMultiLabeledPoint =>
        wmlPoint.data.labels.toArray
    }.collect

    val missPredictCount = (predicts zip labels).map(
      zip => if (zip._1 * zip._2 < 0.0) 1.0 else 0.0)
      .reduce(_ + _)

    assert(missPredictCount / labels.length < 0.125)
  }
}

