package com.keithalcock.torchscript

import org.pytorch.IValue
import org.pytorch.Module
import org.pytorch.Tensor

import java.io.File

object RunTorchScript extends App {
  val datapath = "../data/ner-conll/"
  val embedpath = "../data/glove.840B.300d.10f.txt"
  val modelpath = "../data/model.pt"

  val datamodule = new NerDatamodule(datapath, embedpath)
  println(s"Size of dataset: ${datamodule.len}")
  println(s"Size of vocab: ${datamodule.vocabSize}")

  def makeModel(): Module = Module.load(modelpath)

  val model: Module = makeModel()

  def cropInput(tensorPair: (Tensor, Tensor)): Tensor = tensorPair._1

  val timer = new Timer("The timer")

  {
    // Train
    val times = datamodule.trainDataset.map { tensorPair =>
      val cropped = cropInput(tensorPair)
      timer.time {
        model.forward(IValue.from(cropped))
      }
      timer.getElapsed
    }
    println(s"Mean train sample time: ${timer.mean(times)}")
    println(s"                 stdev: ${timer.stddev(times)}")
  }

  {
    // Val
    val times = datamodule.valDataset.map { tensorPair =>
      val cropped = cropInput(tensorPair)
      timer.time {
        model.forward(IValue.from(cropped))
      }
      timer.getElapsed
    }
    println(s"Mean val sample time: ${timer.mean(times)}")
    println(s"               stdev: ${timer.stddev(times)}")
  }

  {
    // Test
    val times = datamodule.testDataset.map { tensorPair =>
      val cropped = cropInput(tensorPair)
      timer.time {
        model.forward(IValue.from(cropped))
      }
      timer.getElapsed
    }
    println(s"Mean test sample time: ${timer.mean(times)}")
    println(s"                stdev: ${timer.stddev(times)}")
  }
}
