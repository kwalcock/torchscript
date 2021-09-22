package org.clulab.torchscript

import org.clulab.torchscript.utils.Closer.AutoCloser
import org.clulab.torchscript.utils._
import org.pytorch.{IValue, Module, Tensor}

import java.io.File

object RunParOnnx extends App {
  val debug = true
  val threads = 1
  val loops = 64

  val datapath = "../data/ner-conll/"
  val embedpath = "../data/glove.840B.300d.10f.txt"
  val modelpath = "../data/model.pt"

  AppUtils.showStatus()

  val model: Module = Module.load(modelpath)

  val datamodule = new NerDatamodule(datapath, embedpath)
  println(s"Size of dataset: ${datamodule.len}")
  println(s"Size of vocab: ${datamodule.vocabSize}")

  def cropInput(tensorPair: (Tensor, Tensor)): Tensor = tensorPair._1

  def runDataset(name: String, dataset: NerDataset, printer: Printer, index: Int): Unit = {
    val timer = new Timer("The timer")
    val times = dataset.map { tensorPair =>
      val cropped = cropInput(tensorPair)
      val result = timer.time {
        model.forward(IValue.from(cropped))
      }
      printer.println(result.toTensor.shape.mkString(", "))
      printer.println(result.toTensor.getDataAsFloatArray.map(value => f"$value%1.8f").mkString(", ")) // can't be double
      timer.getElapsed
    }
    println(f"  Mean $name sample time $index: ${timer.mean(times)}%.8f")
    println(f"Stddev $name sample time $index: ${timer.stddev(times)}%.8f")
  }

  def newPrinter(index: Int): Printer = {
    if (debug)
      new FakePrinter()
    else
      new RealPrinter(FileUtils.newPrintWriterFromFile(new File(s"RunTorchScript-$index.txt")))
  }


  val timer = new Timer("elapsed time")
  timer.time {
    ThreadUtils.parallelize(1.to(loops), threads).foreach { index =>
      newPrinter(index).autoClose { printWriter =>
        runDataset("train", datamodule.trainDataset, printWriter, index)
        runDataset("  val", datamodule.valDataset, printWriter, index)
        runDataset(" test", datamodule.testDataset, printWriter, index)
      }
    }
  }
  val time = timer.getElapsed()
  println(s"The elapsed time for\t$loops\tloops on\t$threads\tthreads was\t$time\tseconds.")

  System.exit(0);
}
