ThisBuild / scalaVersion     := "2.12.14"
ThisBuild / version          := "0.1.0-SNAPSHOT"
ThisBuild / organization     := "com.keithalcock"
ThisBuild / organizationName := "Keith Alcock"

name := "torchscript"
 
lazy val root = (project in file("."))

libraryDependencies ++= {
  Seq(
    "org.scalatest" %% "scalatest" % "3.2.8" % Test
  )
}

