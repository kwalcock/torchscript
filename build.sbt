
ThisBuild / scalaVersion     := "2.12.14"
ThisBuild / version          := "0.1.0-SNAPSHOT"
ThisBuild / organization     := "com.keithalcock"
ThisBuild / organizationName := "Keith Alcock"

ThisBuild / Test / fork := true // This doesn't seem to work.

name := "torchscript"
 
lazy val library = project

lazy val root = (project in file("."))
  .dependsOn(library)
  .enablePlugins(JavaAppPackaging)

lazy val demo = project
  .dependsOn(library)

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


