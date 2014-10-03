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

package org.apache.spark.mllib.util

import org.apache.spark.SparkException
import org.apache.spark.mllib.linalg.{ Vector, Vectors }

/**
 * Class that represents the features and labels of a data point.
 *
 * @param labels Vector of labels for this data point, typically in the form [1,0,1,0,0].
 * @param features Vector of features for this data point.
 */
case class MultiLabeledPoint(labels: Vector, features: Vector) {
  override def toString = s"($labels,$features)"
}

case class WeightedMultiLabeledPoint(
    weights: Vector,
    data: MultiLabeledPoint) {
  require(weights.size == data.labels.size)
  override def toString = s"($weights,$data)"
}

/**
 * Parser for [[org.apache.spark.mllib.util.MultiLabeledPoint]]
 * and [[org.apache.spark.mllib.util.WeightedMultiLabeledPoint]].
 */
object MultiLabeledPointParser {

  private def parseAnyToVector(any: Any): Vector = {
    any match {
      case values: Array[Double] =>
        Vectors.dense(values)
      case Seq(size: Double, indices: Array[Double], values: Array[Double]) =>
        Vectors.sparse(size.toInt, indices.map(_.toInt), values)
      case other =>
        throw new SparkException(s"Cannot parse $other!")
    }
  }

  /**
   * Parses a string resulted from `MultiLabeledPoint#toString` into
   * an [[org.apache.spark.mllib.util.MultiLabeledPoint]]
   * @param s A string in the form of `([1,0,1],[0.5,4.0,3.0])`, as resulted
   *          from `MultiLabeledPoint#toString`.
   * @return The parsed MultiLabeledPoint.
   */
  def parse(s: String): MultiLabeledPoint = {
    NumericParser.parse(s) match {
      case Seq(labels: Any, features: Any) =>
        MultiLabeledPoint(
          parseAnyToVector(labels),
          parseAnyToVector(features))
      case other =>
        throw new SparkException(s"Cannot parse $other.")
    }
  }

  def parseWeighted(s: String): WeightedMultiLabeledPoint = {
    NumericParser.parse(s) match {
      case Seq(weights: Any, Seq(labels: Any, features: Any)) =>
        WeightedMultiLabeledPoint(
          parseAnyToVector(weights),
          MultiLabeledPoint(
            parseAnyToVector(labels),
            parseAnyToVector(features)))
      case other =>
        throw new SparkException(s"Cannot parse $other")
    }
  }

}

