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

package com.sina.adtech.multiboost

import org.apache.spark.mllib.classification.{ SVMWithSGD, LogisticRegressionWithSGD }
import org.apache.spark.mllib.util.{ MultiLabeledPoint, MultiLabeledPointParser }
import scopt.OptionParser
import java.net.URI
import org.apache.hadoop.conf.Configuration
import org.apache.hadoop.fs.{ FileSystem, Path }
import org.apache.spark.mllib.classification.multilabel.stronglearners.AdaBoostMHAlgorithm
import org.apache.spark.mllib.classification.multilabel.baselearners._
import org.apache.spark.{ SparkContext, SparkConf }
import org.apache.spark.rdd.RDD
import org.apache.spark.Logging

object MultiBoost extends Logging {

  object StrongLearnerType extends Enumeration {
    type StrongLearnerType = Value
    val AdaBoostMH = Value
  }

  object BaseLearnerType extends Enumeration {
    type BaseLearnerType = Value
    val DecisionStump = Value
    val HammingTree = Value
    val LRBase = Value
    val SVMBase = Value
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
    modelPath: String = null,
    numIters: Int = 20,
    numPartitions: Int = 4,
    sampleRate: Double = 1.0,
    featureRate: Double = 1.0,
    baseLearner: BaseLearnerType = DecisionStump,
    strongLearner: StrongLearnerType = AdaBoostMH,
    jobDescription: String = "Spark MultiBoost Classification.")

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
      opt[String]("model_path")
        .text(s"output path to persisted model.")
        .required()
        .action((x, c) => c.copy(modelPath = x))
      opt[Double]("sample_rate")
        .text(s"rate of down-sampling the training dataset, trading accuracy for efficiency.")
        .action((x, c) => c.copy(sampleRate = x))
      opt[Double]("feature_rate")
        .text(s"rate of down-sampling the feature set.")
        .action((x, c) => c.copy(featureRate = x))
      opt[Int]("num_iterations")
        .text(s"num of iterations for the strong learner. default=20")
        .action((x, c) => c.copy(numIters = x))
      opt[Int]("num_partitions")
        .text(s"num of partitions for sc.textFile")
        .action((x, c) => c.copy(numPartitions = x))
      opt[StrongLearnerType]("strong_learner")
        .text(s"the strong learner algorithm (${StrongLearnerType.values.mkString(",")}}),"
          + s" default: AdaBoostMH.")
        .action((x, c) => c.copy(strongLearner = x))
      opt[BaseLearnerType]("base_learner")
        .text(s"the base learner algorithm (${BaseLearnerType.values.mkString(",")}),"
          + s" default: DecisionStump.")
        .action((x, c) => c.copy(baseLearner = x))
      opt[String]("job_description")
        .text(s"the description info in the Name field of spark/hadoop cluster ui.")
        .action((x, c) => c.copy(jobDescription = x))
    }
    parser.parse(args, Params()) map {
      case params: Params =>

        logInfo(s"params: $params")

        // execute the training
        run(params)
    } getOrElse {
      sys.exit(1)
    }
  }

  def run(params: Params) {
    val conf = new SparkConf()
      .setAppName(params.jobDescription)
      .setExecutorEnv("SPARK_JAVA_OPTS", "-Dspark.akka.frameSize=128")
      .set("spark.akka.frameSize", "128")
      .set("spark.storage.memoryFraction", "0.5")
      .set("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
      .set("spark.kryo.registrator", "com.sina.adtech.multiboost.KryoRegistrator")

    val sc = new SparkContext(conf)
    val trainingData = sc.textFile(params.trainingData, params.numPartitions)
      .map(MultiLabeledPointParser.parse)

    val testingData = sc.textFile(params.testingData, params.numPartitions)
      .map(MultiLabeledPointParser.parse)

    logInfo(s"Num of training samples: ${trainingData.count()}\n" +
      s"Num of testing samples: ${testingData.count()}")

    val sample1 = testingData.take(1)(0)
    logInfo(s"Sample1: $sample1")
    val numClasses = sample1.labels.size
    val numFeatureDimensions = sample1.features.size

    val model = params.baseLearner match {
      case BaseLearnerType.DecisionStump =>
        val baseLearnerAlgo = new DecisionStumpAlgorithm(numClasses, numFeatureDimensions,
          params.sampleRate, params.featureRate)
        val strongLearnerAlgo = new AdaBoostMHAlgorithm[DecisionStumpModel, DecisionStumpAlgorithm](
          baseLearnerAlgo, numClasses, numFeatureDimensions, params.numIters)
        strongLearnerAlgo.run(trainingData)

      case BaseLearnerType.LRBase =>
        val binaryAlgo = new LRClassificationAlgorithm
        val baseLearnerAlgo = new GeneralizedBinaryBaseLearnerAlgorithm[LRClassificationModel, LRClassificationAlgorithm](numClasses, numFeatureDimensions, binaryAlgo)
        val strongLearnerAlgo = new AdaBoostMHAlgorithm[GeneralizedBinaryBaseLearnerModel[LRClassificationModel], GeneralizedBinaryBaseLearnerAlgorithm[LRClassificationModel, LRClassificationAlgorithm]](baseLearnerAlgo, numClasses, numFeatureDimensions, params.numIters)
        strongLearnerAlgo.run(trainingData)

      case BaseLearnerType.SVMBase =>
        val binaryAlgo = new SVMClassificationAlgorithm
        val baseLearnerAlgo = new GeneralizedBinaryBaseLearnerAlgorithm[SVMClassificationModel, SVMClassificationAlgorithm](numClasses, numFeatureDimensions, binaryAlgo)
        val strongLearnerAlgo = new AdaBoostMHAlgorithm[GeneralizedBinaryBaseLearnerModel[SVMClassificationModel], GeneralizedBinaryBaseLearnerAlgorithm[SVMClassificationModel, SVMClassificationAlgorithm]](baseLearnerAlgo, numClasses, numFeatureDimensions, params.numIters)
        strongLearnerAlgo.run(trainingData)
    }

    val predictsAndLabels = testingData.map { mlp =>
      ((model predict mlp.features).toArray, mlp.labels.toArray)
    }.cache()

    val hammingLoss = computeHammingLoss(predictsAndLabels)
    val accuracy = computeAccuracy(predictsAndLabels)
    val strictAccuracy = computeStrictAccuracy(predictsAndLabels)
    val precision = computePrecision(predictsAndLabels)
    val recall = computeRecall(predictsAndLabels)
    val f1Score = computeF1Score(precision, recall)

    var resultStr = s"$model\n\n"
    resultStr += s"${model.debugString}\n\n"
    resultStr += s"Num of training samples: ${trainingData.count()}\n"
    resultStr += s"Num of testing samples: ${testingData.count()}\n"
    resultStr += s"Testing hamming loss is: $hammingLoss\n"
    resultStr += s"Testing accuracy is: $accuracy\n"
    resultStr += s"Testing strict accuracy is: $strictAccuracy\n"
    resultStr += s"Testing precision is: $precision\n"
    resultStr += s"Testing recall is: $recall\n"
    resultStr += s"F1 score is: $f1Score\n"

    val hadoopConf = new Configuration()
    val fs = FileSystem.get(URI.create(params.modelPath), hadoopConf)
    var dst = new Path(params.modelPath + "/model.txt")
    val out = fs.create(dst)
    out.write(resultStr.getBytes("UTF-8"))

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
        (predicts zip labels).count {
          case (p, l) =>
            p * l > 0.0
        } == predicts.size
    }.count().toDouble / predictsAndLabels.count().toDouble
  }

  def computeF1Score(precision: Double, recall: Double): Double = {
    if (precision + recall > 0.0) 2.0 * precision * recall / (precision + recall) else 0.0
  }
}
