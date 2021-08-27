package com.keithalcock.torchscript

import org.pytorch.Tensor

import scala.collection.mutable
import scala.collection.mutable.ArrayBuffer
import scala.io.Source

class NerDataset(datasetPath: String, val vocab: Map[String, Int]) extends IndexedSeq[(Tensor, Tensor)] {

  case class Sample(tokens: Array[String], labels: Array[String])

  val samples = new ArrayBuffer[Sample]()
  val labels = mutable.Map.empty[String, Int]

  {
    // Return the samples and labels from this?
    val tokens = new ArrayBuffer[String]()
    val labels = new ArrayBuffer[String]()
    val source = Source.fromFile(datasetPath)

    source.getLines.foreach { line =>
      val cur = line.trim

      if (cur.isEmpty) {
        samples.append(Sample(tokens.toArray, labels.toArray))
        tokens.clear()
        labels.clear()
      }
      else {
        val Array(token, label) = cur.split(' ')
        tokens.append(token)
        labels.append(label)

        // Make this one a set and just add
        if (!this.labels.contains(label))
          this.labels(label) = -1
      }
    }
    source.close

    this.labels.keys.toSeq.sorted.zipWithIndex.foreach { case (key, index) =>
      this.labels(key) = index
    }
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
