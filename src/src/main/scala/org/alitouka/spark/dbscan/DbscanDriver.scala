package org.alitouka.spark.dbscan

import org.apache.spark.{SparkConf, SparkContext}
import org.alitouka.spark.dbscan.util.io.IOHelper
import org.alitouka.spark.dbscan.util.commandLine._
import org.alitouka.spark.dbscan.spatial.rdd.PartitioningSettings
import org.alitouka.spark.dbscan.util.debug.{DebugHelper, Clock}

/** A driver program which runs DBSCAN clustering algorithm
 *
 */
object DbscanDriver {

  private [dbscan] class Args (var minPts: Int = DbscanSettings.getDefaultNumberOfPoints,
      var borderPointsAsNoise: Boolean = DbscanSettings.getDefaultTreatmentOfBorderPoints)
      extends CommonArgs with EpsArg with NumberOfPointsInPartitionArg

  private [dbscan] class ArgsParser
    extends CommonArgsParser (new Args (), "DBSCAN clustering algorithm")
    with EpsArgParsing [Args]
    with NumberOfPointsInPartitionParsing [Args] {

    opt[Int] ("numPts")
      .required()
      .foreach { args.minPts = _ }
      .valueName("<minPts>")
      .text("TODO: add description")

    opt[Boolean] ("borderPointsAsNoise")
      .foreach { args.borderPointsAsNoise = _ }
      .text ("A flag which indicates whether border points should be treated as noise")
  }


  def main (args: Array[String]): Unit = {
    val t1 = System.currentTimeMillis()

    val argsParser = new ArgsParser ()

    if (argsParser.parse (args)) {

      val clock = new Clock ()


      val conf = new SparkConf()
        .setMaster(argsParser.args.masterUrl)
        .setAppName("alitouka DBSCAN")
        .setJars(Array(argsParser.args.jar))

      if (argsParser.args.debugOutputPath.isDefined) {
        conf.set (DebugHelper.DebugOutputPath, argsParser.args.debugOutputPath.get)
      }


      val sc = new SparkContext(conf)

      println("\n--------start--------\n")
      println("* start:             " + t1)
      val data = IOHelper.readDataset(sc, argsParser.args.inputPath).cache()
      println("trainData: " + data.count())
      val t2 = System.currentTimeMillis()
      println("* after preprocess:  " + t2)

      val settings = new DbscanSettings ()
        .withEpsilon(argsParser.args.eps)
        .withNumberOfPoints(argsParser.args.minPts)
        .withTreatBorderPointsAsNoise(argsParser.args.borderPointsAsNoise)
        .withDistanceMeasure(argsParser.args.distanceMeasure)

      val partitioningSettings = new PartitioningSettings (numberOfPointsInBox = argsParser.args.numberOfPoints)

      val clusteringResult = Dbscan.train(data, settings, partitioningSettings)
      IOHelper.saveClusteringResult(clusteringResult, argsParser.args.outputPath)
      val t3 = System.currentTimeMillis()
      println("* after train:       " + t3)

      clock.logTimeSinceStart("Clustering")

      println("\n--------success--------\n")

      val trainingProcess = (t3 - t1).toDouble / 1000
      val trainingStep = (t3 - t2).toDouble / 1000
      val dataProcess = (t2 - t1).toDouble / 1000
      println("[s]train total:     " + trainingProcess)
      println("[s]data preprocess: " + dataProcess)
      println("[s]train:           " + trainingStep)
    }
  }
}
