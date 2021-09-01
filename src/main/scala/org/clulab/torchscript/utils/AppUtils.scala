package org.clulab.torchscript.utils

object AppUtils {

  def showEnv(name: String): Unit = {
    val value = Option(System.getenv(name)).getOrElse("")
    println(s"$name = $value")
  }

  def showProp(name: String): Unit = {
    val value = Option(System.getProperty(name)).getOrElse("")
    println(s"$name = $value")
  }

  def showStatus(): Unit = {
    showEnv("PATH")
    showEnv("LD_LIBRARY_PATH")
    showEnv("DYLD_LIBRARY_PATH")
    showProp("java.library.path")
  }
}
