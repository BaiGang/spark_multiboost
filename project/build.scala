import sbt._
import Keys._
import scala._
import com.github.retronym.SbtOneJar
import com.typesafe.sbt.SbtScalariform.scalariformSettings
import org.scalastyle.sbt.ScalastylePlugin
import sbtassembly.Plugin._
import AssemblyKeys._

object spark_multiboost extends Build {
  lazy val sharedLibraryDependencies = Seq(
    "org.scalatest" %% "scalatest" % "2.1.5" % "test",
    "org.apache.spark" %% "spark-core" % "1.1.0" % "provided",
    "org.apache.spark" %% "spark-mllib" % "1.1.0" % "provided",
    "org.apache.hadoop" % "hadoop-client" % "2.4.0" % "provided",
    "org.scalanlp" %% "breeze" % "0.8" % "provided",
    "com.github.scopt" %% "scopt" % "3.2.0"
  )

  def extraAssemblySettings() = Seq(
    test in assembly := {},
    mergeStrategy in assembly := {
      case PathList("org", "datanucleus", xs @ _*)             => MergeStrategy.discard
      case m if m.toLowerCase.endsWith("manifest.mf")          => MergeStrategy.discard
      case m if m.toLowerCase.matches("meta-inf.*\\.sf$")      => MergeStrategy.discard
      case "log4j.properties"                                  => MergeStrategy.discard
      case m if m.toLowerCase.startsWith("meta-inf/services/") => MergeStrategy.filterDistinctLines
      case "reference.conf"                                    => MergeStrategy.concat
      case _                                                   => MergeStrategy.first
    }
  )

  def scalaSettings = Seq(
    scalaVersion := "2.10.4",
    scalacOptions ++= Seq(
      "-optimize",
      "-unchecked",
      "-deprecation"
    ),

    libraryDependencies ++= sharedLibraryDependencies
  ) ++ scalariformSettings ++ ScalastylePlugin.Settings ++ assemblySettings ++ extraAssemblySettings

  def buildSettings =
    Project.defaultSettings ++
    scalaSettings

  lazy val root = {
    val settings = buildSettings ++ Seq(name := "spark_multiboost", version := "0.4.1") ++ SbtOneJar.oneJarSettings
    Project(id = "spark_multiboost", base = file("."), settings = settings)
  }
}

