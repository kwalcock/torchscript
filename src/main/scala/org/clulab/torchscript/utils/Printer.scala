package org.clulab.torchscript.utils

import java.io.PrintWriter

abstract class Printer {
  def println(text: => String): Unit
  def close(): Unit
}

class RealPrinter(printWriter: PrintWriter) extends Printer {

  def println(text: => String): Unit = printWriter.println(text)

  def close(): Unit = printWriter.close()
}

class FakePrinter() extends Printer {

  def println(text: => String): Unit = ()

  def close(): Unit = ()
}
