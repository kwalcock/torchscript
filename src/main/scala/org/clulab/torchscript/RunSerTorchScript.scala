package org.clulab.torchscript

import org.clulab.torchscript.utils.{AppUtils, Timer}
import org.pytorch.IValue
import org.pytorch.Module
import org.pytorch.Tensor

object RunSerTorchScript extends App {
  val datapath = "../data/ner-conll/"
  val embedpath = "../data/glove.840B.300d.10f.txt"
  val modelpath = "../data/model.pt"

  AppUtils.showStatus()

  val model: Module = Module.load(modelpath)

  val datamodule = new NerDatamodule(datapath, embedpath)
  println(s"Size of dataset: ${datamodule.len}")
  println(s"Size of vocab: ${datamodule.vocabSize}")

  def cropInput(tensorPair: (Tensor, Tensor)): Tensor = tensorPair._1

  val timer = new Timer("The timer")

  def runDataset(name: String, dataset: NerDataset): Unit = {
    val times = dataset.map { tensorPair =>
      val cropped = cropInput(tensorPair)
      val result = timer.time {
        model.forward(IValue.from(cropped))
      }
      // println(result.toTensor.shape.mkString(", "))
      // println(result.toTensor.getDataAsFloatArray.map(value => f"$value%1.8f").mkString(", ")) // can't be double
      timer.getElapsed
    }
    println(f"  Mean $name sample time: ${timer.mean(times)}%.8f")
    println(f"Stddev $name sample time: ${timer.stddev(times)}%.8f")
  }

  runDataset("train", datamodule.trainDataset)
  runDataset("  val", datamodule.valDataset)
  runDataset(" test", datamodule.testDataset)

  System.exit(0);
}
