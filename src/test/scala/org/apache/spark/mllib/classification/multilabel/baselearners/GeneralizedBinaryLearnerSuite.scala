
package org.apache.spark.mllib.classification.multilabel.baselearners

import org.apache.spark.mllib.classification.SVMModel
import org.apache.spark.mllib.classification.multilabel.stronglearners.AdaBoostMHModel
import org.scalatest.FunSuite
import org.apache.spark.SparkContext._
import org.apache.spark.mllib.util.LocalSparkContext
import org.apache.spark.mllib.linalg.Vectors

class GeneralizedBinaryLearnerSuite extends FunSuite with LocalSparkContext {

  test("GeneralizedBinaryBaseLearner with SVM") {
    val svmModel1 = new SVMClassificationModel(new SVMModel(Vectors.dense(1.0, -1.0, 1.0), 0.0))
    val svmModel2 = new SVMClassificationModel(new SVMModel(Vectors.dense(-1.0, -1.0, 0.0), 0.0))
    val featureVec1 = Vectors.dense(2.0, 3.0, 4.0)
    val featureVec2 = Vectors.dense(-5.0, 4.0, 3.0)

    assert(svmModel1.predictPoint(featureVec1) === 1.0)
    assert(svmModel1.predictPoint(featureVec2) === 0.0)
    assert(svmModel2.predictPoint(featureVec1) === 0.0)
    assert(svmModel2.predictPoint(featureVec2) === 1.0)

    val gbm1 = new GeneralizedBinaryBaseLearnerModel[SVMClassificationModel](
      1.0, Vectors.dense(1.0, -1.0), svmModel1)
    val gbm2 = new GeneralizedBinaryBaseLearnerModel[SVMClassificationModel](
      0.5, Vectors.dense(1.0, 1.0), svmModel2)

    assert(gbm1.predict(featureVec1) === Vectors.dense(1.0, -1.0))
    assert(gbm1.predict(featureVec2) === Vectors.dense(-1.0, 1.0))
    assert(gbm2.predict(featureVec1) === Vectors.dense(-0.5, -0.5))
    assert(gbm2.predict(featureVec2) === Vectors.dense(0.5, 0.5))

    val adaboostMHModel = new AdaBoostMHModel[GeneralizedBinaryBaseLearnerModel[SVMClassificationModel]](
      2, 3, List(gbm1, gbm2))

    assert(adaboostMHModel.predict(featureVec1) === Vectors.dense(1.0, -1.0))
    assert(adaboostMHModel.predict(featureVec2) === Vectors.dense(-1.0, 1.0))
  }

  test("GeneralizedBinaryBaseLearner with LR") {

  }
}
