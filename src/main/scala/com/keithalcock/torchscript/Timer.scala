package com.keithalcock.torchscript

class Timer(val description: String) {
  protected var elapsedTime: Option[Long] = None
  protected val divisor = 1000000000 // for nanoseconds

  def time[R](block: => R): R = {
    val t0 = System.nanoTime()
    val result: R = block    // call-by-name
    val t1 = System.nanoTime()

    elapsedTime = Some(t1 - t0)
    result
  }

  def getElapsed(): Double = elapsedTime.get.toDouble / divisor

  def mean(times: IndexedSeq[Double]): Double = {
    times.sum / times.length
  }

  def stddev(times: IndexedSeq[Double]): Double = {
    val mu = mean(times)
    val deviation = times
      .map { x => math.pow(x - mu, 2.0) }
      .sum
    val result = math.sqrt(deviation / times.length)

    result
  }
}
