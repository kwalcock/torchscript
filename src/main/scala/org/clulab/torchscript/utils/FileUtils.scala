package org.clulab.torchscript.utils

import java.io._
import java.nio.charset.StandardCharsets

object FileUtils {
  val utf8: String = StandardCharsets.UTF_8.toString

  def newPrintWriterFromFile(file: File): PrintWriter = {
    new PrintWriter(new OutputStreamWriter(new BufferedOutputStream(new FileOutputStream(file)), utf8))
  }
}
