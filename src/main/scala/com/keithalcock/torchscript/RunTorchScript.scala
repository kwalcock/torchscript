package com.keithalcock.torchscript

import org.pytorch.IValue
import org.pytorch.Module
import org.pytorch.Tensor

class RunTorchScript extends App {
  val datapath = "../../../../data/ner-conll/"
  val embedpath = "../../../../data/glove.840B.300d.10f.txt"
  val modelpath = "./model.pt"

  def makeModel(): Module = Module.load(modelpath)

  val model: Module = makeModel()

  val datamodule = new NerDatamodule(datapath, embedpath) // move this up
  print(s"Size of dataset: ${datamodule.len}")
  print(s"Size of vocab: ${datamodule.vocabSize}")

  def cropInput(tensorPair: (Tensor, Tensor)): Tensor = tensorPair._1

  val timer = new Timer("The timer")

  {
    // Train
    val times = datamodule.trainDataset.map { tensorPair =>
      timer.time {
        val cropped = cropInput(tensorPair)
        model.forward(IValue.from(cropped))
      }
      timer.getElapsed
    }
    print(s"Mean train sample time: ${timer.mean(times)}")
    print(s"                 stdev: ${timer.stddev(times)}")
  }

  {
    // Val
    val times = datamodule.valDataset.map { tensorPair =>
      timer.time {
        val cropped = cropInput(tensorPair)
        model.forward(IValue.from(cropped))
      }
      timer.getElapsed
    }
    print(s"Mean val sample time: ${timer.mean(times)}")
    print(s"               stdev: ${timer.stddev(times)}")
  }

  {
    // Test
    val times = datamodule.testDataset.map { tensorPair =>
      timer.time {
        val cropped = cropInput(tensorPair)
        model.forward(IValue.from(cropped))
      }
      timer.getElapsed
    }
    print(s"Mean test sample time: ${timer.mean(times)}")
    print(s"                stdev: ${timer.stddev(times)}")
  }
}
