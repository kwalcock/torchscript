package com.keithalcock.torchscript

import com.keithalcock.torchscript.utils.Closer.AutoCloser
import com.keithalcock.torchscript.utils.StringUtils

import scala.io.Source

object NerVocab {
  val mask = "<MASK>"

  def apply(embeddingPath: String): Map[String, Int] = {
    val words = Source.fromFile(embeddingPath).autoClose { source =>
      source.getLines.drop(1).map { line =>
        StringUtils.beforeFirst(line, ' ')
      }.toVector // must be realized before close
    }
    // This first one isn't technically sorted.
    val sortedWords = mask +: words.sorted
    // Add one so that we can preserve index 0 for our padding
    val vocab = sortedWords.zipWithIndex.map { case (word, index) =>
      // println(s"${index}\t$word")
      word -> index
    }.toMap

    vocab
  }
}
