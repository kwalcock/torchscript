package com.keithalcock.torchscript

import org.pytorch.Tensor

import scala.collection.mutable
import scala.collection.mutable.ArrayBuffer
import scala.io.Source

class NerDataset(datasetPath: String, embeddingPath: String) extends IndexedSeq[(Tensor, Tensor)] {

  case class Sample(tokens: Array[String], labels: Array[String])

  val samples = new ArrayBuffer[Sample]()
  val labels = mutable.Map.empty[String, Int]
  val vocab = mutable.Map.empty[String, Int]

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
        val Array(token, label) = cur.split('\t')
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

  {
    val source = Source.fromFile(embeddingPath)

    source.getLines.foreach { line =>
      val cur = line.trim.split('\t').head

      this.vocab(cur) = -1
    }
    source.close

    // Add one so that we can preserve index 0 for our padding
    this.vocab.keys.toSeq.sorted.zipWithIndex.foreach { case (key, index) =>
      this.vocab(key) = index + 1
    }
    this.vocab(NerDataset.mask) = this.vocab.size
  }

  override def length: Int = samples.length

  override def apply(index: Int): (Tensor, Tensor) = {
    require(0 <= index && index < length)
    val sample = samples(index)
    val tokenIndexes = sample.tokens.map { token =>
      vocab.getOrElse(token, vocab(NerDataset.mask))
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
  val mask = "<MASK>"
}