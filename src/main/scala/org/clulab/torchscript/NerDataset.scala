package org.clulab.torchscript

import org.clulab.torchscript.utils.Closer.AutoCloser
import org.clulab.torchscript.utils.StringUtils
import org.pytorch.Tensor

import scala.collection.mutable
import scala.collection.mutable.ArrayBuffer
import scala.io.Source

class NerDataset(datasetPath: String, val vocab: Map[String, Int]) extends IndexedSeq[(Tensor, Tensor)] {

  case class Sample(tokens: Array[String], labels: Array[String])

  object Sample {
    def apply(tokensAndLabels: Array[(String, String)]): Sample = {
      val (tokens, labels) = tokensAndLabels.unzip
      Sample(tokens, labels)
    }
  }

  val samples = Source.fromFile(datasetPath).autoClose { source =>
    val samples = new ArrayBuffer[Sample]()
    val lines = source.getLines

    while (lines.nonEmpty) {
      val sampleLines = lines.takeWhile(_ != " ").toArray
      val tokensAndLabels = sampleLines.map { line =>
        (
          StringUtils.beforeFirst(line, ' ', true),
          StringUtils.afterLast(line, ' ', false)
        )
      }
      samples += Sample(tokensAndLabels)
    }
    samples.toArray
  }
  val labels = {
    val labels = mutable.Map.empty[String, Int]

    samples.foreach { sample =>
      sample.labels.foreach { label =>
        labels.getOrElseUpdate(label, labels.size)
      }
    }
    labels
  }

  override def length: Int = samples.length

  override def apply(index: Int): (Tensor, Tensor) = {
    require(0 <= index && index < length)
    val sample = samples(index)
    val tokenIndexes = sample.tokens.map { token =>
      vocab.getOrElse(token, vocab(NerVocab.mask))
    }
    val labelIndexes = sample.labels.map { label =>
      labels(label)
    }

    (
      Tensor.fromBlob(tokenIndexes, Array(1L, tokenIndexes.length.toLong)),
      Tensor.fromBlob(labelIndexes, Array(1L, labelIndexes.length.toLong))
    )
  }
}
