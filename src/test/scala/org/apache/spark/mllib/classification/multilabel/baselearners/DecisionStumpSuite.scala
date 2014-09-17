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
import org.apache.spark.SparkContext._
import org.apache.spark.mllib.util.LocalSparkContext
import org.apache.spark.mllib.classification.multilabel.MultiLabeledPointParser
import org.apache.spark.mllib.classification.multilabel.WeightedMultiLabeledPoint
import org.apache.spark.mllib.classification.multilabel.MultiLabeledPoint
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.rdd.RDD
import org.apache.spark.rdd.PairRDDFunctions

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

  test("Calulate split metrics and generate the model.") {

    // To get split metrics on features 0, 1 and 2.
    val getSplitFunc = DecisionStumpAlgorithm.getLocalSplitMetrics(Iterator(0, 1, 2)) _

    val allSplitMetrics = dataSet.map((0, _))
      .groupByKey()
      .map(_._2.toArray)
      .flatMap(getSplitFunc)

    assert(allSplitMetrics.count() === 16 + 3)

    val aggregatedSplitMetrics = DecisionStumpAlgorithm aggregateSplitMetrics allSplitMetrics

    assert(aggregatedSplitMetrics.count === 16 + 1)

    val model: DecisionStumpModel = DecisionStumpAlgorithm findBestSplitMetrics aggregatedSplitMetrics

    // From the dataset, we see that the best DecisionStumpModel should be FeatureCut(1,0.6) with
    // votes [-1, 1], which has only one classification error on the dataset.
    assert(model.votes === Vectors.dense(-1.0, 1.0))
    assert(model.cut === Some(FeatureCut(1, 0.6)))
  }

}

