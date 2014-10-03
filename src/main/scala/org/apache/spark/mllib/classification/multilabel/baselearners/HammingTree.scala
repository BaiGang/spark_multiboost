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

import org.apache.spark.mllib.linalg.{ Vector, Vectors }
import org.apache.spark.mllib.util.WeightedMultiLabeledPoint
import org.apache.spark.rdd.RDD

/**
 * The Hamming tree model.
 * A basic tree learner adapted to the multi-class setup of AdaBoost.MH.
 */
class HammingTreeModel extends BaseLearnerModel {

  /**
   * Given the feature vector, predict the labels.
   *
   * @param features feature of the data point
   * @return Vector of the labels.
   */
  override def predict(features: Vector): Vector = {
    // TODO: predictions
    Vectors.dense(1.0, 2, 0)
  }

  /**
   * Given a RDD of the feature vectors, predict the labels.
   *
   * @param features RDD of the feature vectors.
   * @return RDD of the predicted label assignments.
   */
  override def predict(features: RDD[Vector]): RDD[Vector] = {
    features map predict
  }

}

/**
 * The algorithm class for training HammingTreeModel.
 *
 * @param numNodes Num of nodes in a tree.
 */
class HammingTreeAlgorithm(
    _numClasses: Int,
    _numFeatureDimensions: Int,
    numNodes: Int) extends BaseLearnerAlgorithm[HammingTreeModel] {

  override def numClasses = _numClasses
  override def numFeatureDimensions = _numFeatureDimensions

  /**
   * Procedure for training a HammingTreeModel.
   * @param dataSet The weighted training data set.
   * @return A HammingTreeModel trained with the dataset.
   */
  override def run(dataSet: RDD[WeightedMultiLabeledPoint]): HammingTreeModel = {
    new HammingTreeModel()
  }
}
