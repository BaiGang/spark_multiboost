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

package org.apache.spark.examples.mllib

import java.io._
import scopt.OptionParser
import org.apache.spark.mllib.classification.multilabel.stronglearners.AdaBoostMHAlgorithm
import org.apache.spark.mllib.classification.multilabel.stronglearners.AdaBoostMHModel
import org.apache.spark.mllib.classification.multilabel.baselearners.DecisionStumpAlgorithm
import org.apache.spark.mllib.classification.multilabel.baselearners.DecisionStumpModel
import org.apache.spark.{ SparkContext, SparkConf }
import org.apache.spark.rdd.RDD
import org.apache.spark.mllib.classification.multilabel.MultiLabeledPoint
import org.apache.spark.mllib.classification.multilabel.MultiLabeledPointParser

object MultiBoost {

  object StrongLearnerType extends Enumeration {
    type StrongLearnerType = Value
    val AdaBoostMH = Value
  }

  object BaseLearnerType extends Enumeration {
    type BaseLearnerType = Value
    val DecisionStump = Value
    val HammingTree = Value
  }

  import StrongLearnerType._
  import BaseLearnerType._

  implicit val StrongLearnerTypeRead: scopt.Read[StrongLearnerType.Value] =
    scopt.Read.reads(StrongLearnerType withName _)

  implicit val BaseLearnerTypeRead: scopt.Read[BaseLearnerType.Value] =
    scopt.Read.reads(BaseLearnerType withName _)

  case class Params(
    trainingData: String = null,
    testingData: String = null,
    model: String = null,
    numIters: Int = 20,
    sampleRate: Double = 0.4,
    baseLearner: BaseLearnerType = DecisionStump,
    strongLearner: StrongLearnerType = AdaBoostMH)

  def main(args: Array[String]) {
    val parser = new OptionParser[Params]("MultiBoost") {
      head("MultiBoost: A multi-class, multi-label classifier.")
      opt[String]("training_data")
        .text(s"input paths to training data of multi-labeled examples in MultiLabeledPoint format.")
        .required()
        .action((x, c) => c.copy(trainingData = x))
      opt[String]("testing_data")
        .text(s"paths to testing data in MultiLabeledPoint format.")
        .required()
        .action((x, c) => c.copy(testingData = x))
      opt[String]("model")
        .text(s"output path to persisted model.")
        .required()
        .action((x, c) => c.copy(model = x))
      opt[Double]("sample_rate")
        .text(s"rate of down-sampling the training dataset, trading accuracy for efficiency.")
        .action((x, c) => c.copy(sampleRate = x))
      opt[Int]("num_iterations")
        .text(s"num of iterations for the strong learner. default=20")
        .action((x, c) => c.copy(numIters = x))
      opt[StrongLearnerType]("strongLearner")
        .text(s"the strong learner algorithm (${StrongLearnerType.values.mkString(",")}}),"
          + s" default: AdaBoostMH.")
        .action((x, c) => c.copy(strongLearner = x))
      opt[BaseLearnerType]("baseLearner")
        .text(s"the base learner algorithm (${BaseLearnerType.values.mkString(",")}),"
          + s" default: DecisionStump.")
        .action((x, c) => c.copy(baseLearner = x))
    }
    parser.parse(args, Params()) map {
      case params: Params =>

        // Currently we only support AdaBoost.MH with Decision Stump as base learner.
        if (params.baseLearner != DecisionStump
          && params.strongLearner != AdaBoostMH) {
          println(s"Currently we only support Adaboost.MH with DecisionStump.")
          sys.exit(1)
        }

        println(s"params: $params")

        // execute the training
        run(params)
    } getOrElse {
      sys.exit(1)
    }
  }

  def run(params: Params) {
    val conf = new SparkConf().setAppName("MultiBoost")
    val sc = new SparkContext(conf)
    val trainingData = sc.textFile(params.trainingData, sc.defaultMinPartitions)
      .map(MultiLabeledPointParser.parse)

    val testingData = sc.textFile(params.testingData, sc.defaultMinPartitions)
      .map(MultiLabeledPointParser.parse)

    println(s"Num of training samples: ${trainingData.count()}\n" +
      s"Num of testing samples: ${testingData.count()}")

    val sample1 = testingData.take(1)(0)
    println(s"Sample1: $sample1")
    val numClasses = sample1.labels.size
    val numFeatureDimensions = sample1.features.size

    val baseLearnerAlgo = new DecisionStumpAlgorithm(numClasses, numFeatureDimensions, params.sampleRate)
    val strongLearnerAlgo = new AdaBoostMHAlgorithm[DecisionStumpModel, DecisionStumpAlgorithm](
      baseLearnerAlgo,
      numClasses,
      numFeatureDimensions,
      params.numIters)

    println("Start training...")
    val model = strongLearnerAlgo.run(trainingData)
    println("Training done.")

    println(s"Writing the model to ${params.model}...")
    val writer = new PrintWriter(new File(params.model))
    writer.write(model.toString)
    writer.close()
    println("Writing done.")

    println("Start testing...")
    val predicts = testingData.map {
      case s: MultiLabeledPoint =>
        model predict s.features
    }.cache()
    println("Testing done.")

    val predictsAndLabels = (predicts zip testingData map {
      case (p, d) => (p.toArray, d.labels.toArray)
    }).cache()

    val hammingLoss = computeHammingLoss(predictsAndLabels)
    val accuracy = computeAccuracy(predictsAndLabels)
    val strictAccuracy = computeStrictAccuracy(predictsAndLabels)
    val precision = computePrecision(predictsAndLabels)
    val recall = computeRecall(predictsAndLabels)
    val f1Score = computeF1Score(precision, recall)

    println(s"Num of training samples: ${trainingData.count}\n"
      + s"Num of testing samples: ${testingData.count}\n"
      + s"Testing hamming loss is: $hammingLoss\n"
      + s"Testing accuracy is: $accuracy\n"
      + s"Testing strict accuracy is: $strictAccuracy\n"
      + s"Testing precision is: $precision\n"
      + s"Testing recall is: $recall\n"
      + s"F1 score is: $f1Score")
    sc.stop()
  }

  def computeHammingLoss(predictsAndLabels: RDD[(Array[Double], Array[Double])]): Double = {
    predictsAndLabels.flatMap {
      case (ps, ls) =>
        (ps zip ls) filter {
          case (p, l) =>
            p * l < 0.0
        }
    }.count().toDouble /
      (predictsAndLabels.count() * predictsAndLabels.take(1)(0)._2.size)
  }

  def computePrecision(predictsAndLabels: RDD[(Array[Double], Array[Double])]): Double = {
    val positiveSet = predictsAndLabels.flatMap {
      case (predicts, labels) =>
        (predicts zip labels) filter (_._1 > 0.0)
    }

    positiveSet.filter {
      case (predict, label) =>
        predict * label > 0.0
    }.count().toDouble / positiveSet.count()
  }

  def computeRecall(predictsAndLabels: RDD[(Array[Double], Array[Double])]): Double = {
    val trueSet = predictsAndLabels.flatMap {
      case (predicts, labels) =>
        (predicts zip labels) filter (_._2 > 0.0)
    }

    trueSet.filter {
      case (predict, label) =>
        predict * label > 0.0
    }.count().toDouble / trueSet.count()
  }

  def computeAccuracy(predictsAndLabels: RDD[(Array[Double], Array[Double])]): Double = {
    val total = predictsAndLabels.flatMap {
      case (predicts, labels) =>
        predicts zip labels
    }

    total.filter { case (p, l) => p * l > 0.0 }.count().toDouble / total.count()
  }

  def computeStrictAccuracy(predictsAndLabels: RDD[(Array[Double], Array[Double])]): Double = {
    predictsAndLabels.filter {
      case (predicts, labels) =>
        (predicts zip labels).filter {
          case (p, l) =>
            p * l > 0.0
        }.size == predicts.size
    }.count().toDouble / predictsAndLabels.count().toDouble
  }

  def computeF1Score(precision: Double, recall: Double): Double = {
    if (precision + recall > 0.0) 2.0 * precision * recall / (precision + recall) else 0.0
  }
}
