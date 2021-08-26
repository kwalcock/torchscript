ThisBuild / scalaVersion     := "2.12.14"
ThisBuild / version          := "0.1.0-SNAPSHOT"
ThisBuild / organization     := "com.keithalcock"
ThisBuild / organizationName := "Keith Alcock"

name := "torchscript"
 
lazy val root = (project in file("."))

lazy val demo = project

libraryDependencies ++= {
  Seq(
    // This one requires the next.
    // "org.pytorch"           % "pytorch_java_only" % "1.9.0",
    // The next one can't be found.  Use jars in lib directory.
    // "com.facebook.fbjni"    % "fbjni-java-only"   % "0.0.3",
    // And this is a transitive dependency
    // "com.facebook.soloader" % "nativeloader"      % "0.8.0",
    "org.scalatest"     %% "scalatest"         % "3.2.8" % Test
  )
}


