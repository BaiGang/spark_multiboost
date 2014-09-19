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

package org.apache.spark.mllib.classification.multilabel

import org.scalatest.FunSuite
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.util.{ MultiLabeledPoint, MultiLabeledPointParser }

class MultiLabeledPointSuite extends FunSuite {

  test("parse multi-labeled points") {
    Seq(
      MultiLabeledPoint(
        Vectors.dense(1.0, 0.0, 1.0),
        Vectors.dense(0.5, 1.3, 2.2)),
      MultiLabeledPoint(
        Vectors.sparse(2, Array(1), Array(-1.0)),
        Vectors.sparse(2, Array(1), Array(1.0))),
      MultiLabeledPoint(
        Vectors.dense(1.0, 0.0, 1.0),
        Vectors.sparse(2, Array(1), Array(1.0))),
      MultiLabeledPoint(
        Vectors.sparse(2, Array(1), Array(-1.0)),
        Vectors.dense(1.0, 0.0, 1.0))
    ).foreach(p => assert(p === MultiLabeledPointParser.parse(p.toString)))
  }
}
