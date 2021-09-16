package org.clulab.torchscript

import org.clulab.torchscript.utils.Closer.AutoCloser
import org.clulab.torchscript.utils.StringUtils
import org.pytorch.Tensor

import scala.annotation.tailrec
import scala.collection.mutable
import scala.io.Source

class NerDataset(datasetPath: String, val vocab: Map[String, Int]) extends IndexedSeq[(Tensor, Tensor)] {

  protected def getSamples(lines: Iterator[String]): Array[NerDataset.Sample] = {

    @tailrec
    def recGetSamples(list: List[NerDataset.Sample]): List[NerDataset.Sample] = {
      if (lines.isEmpty) list.reverse
      else {
        val tokensAndLabels = lines.takeWhile(_ != " ").toArray.map { line =>
          (
            StringUtils.beforeFirst(line, ' ', all = true),
            StringUtils.afterLast(line, ' ', all = false)
          )
        }

        recGetSamples(NerDataset.Sample(tokensAndLabels) :: list)
      }
    }

    recGetSamples(Nil).toArray
  }

  protected val samples: Array[NerDataset.Sample] = Source.fromFile(datasetPath).autoClose { source =>
    getSamples(source.getLines)
  }
  protected val labels: mutable.Map[String, Int] = {
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

object NerDataset {

  case class Sample(tokens: Array[String], labels: Array[String])

  object Sample {
    def apply(tokensAndLabels: Array[(String, String)]): Sample = {
      val (tokens, labels) = tokensAndLabels.unzip
      Sample(tokens, labels)
    }
  }
}
