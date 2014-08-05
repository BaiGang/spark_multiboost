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

package com.sina.adalgo.multiboost

import org.apache.spark.annotation.Experimental
import org.apache.spark.mllib.linalg.Vector
import org.apache.spark.rdd.RDD

@Experimental
trait MultiLabelClassificationModel extends Serializable {
  def predict(testData: RDD[Vector]): RDD[Vector]
  def predict(testData: Vector): Vector
}

@Experimental
trait MultiLabelClassificationAlgorithm[
    M <: MultiLabelClassificationModel] extends Serializable {
  // XXX: no abstract interface for now
  // def run(dataSet: RDD[WeightedMultiLabeledPoint]): M
  // def run(dataSet: RDD[MultiLabeledPoint]): M
  def numClasses: Int
  def numFeatureDimensions: Int
}
