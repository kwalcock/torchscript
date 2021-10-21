package org.clulab.torchscript

import ai.onnxruntime.{OnnxTensor, OrtEnvironment, OrtSession}
import org.clulab.torchscript.utils.{AppUtils, FakePrinter, FileUtils, Printer, RealPrinter, ThreadUtils, Timer}
import org.clulab.torchscript.utils.Closer.AutoCloser
import org.pytorch.Tensor

import java.io.File
import java.nio.LongBuffer
import java.util.{HashMap => JHashMap}

object RunParOnnx extends App {
  val debug = true
  val threads = 128
  val loops = 64

  val datapath = "../data/ner-conll/"
  val embedpath = "../data/glove.840B.300d.10f.txt"
  val modelpath = "../data/model.onnx"

  AppUtils.showStatus()

  val ortEnvironment = OrtEnvironment.getEnvironment
  val session = ortEnvironment.createSession(modelpath, new OrtSession.SessionOptions)

  val datamodule = new NerDatamodule(datapath, embedpath)
  println(s"Size of dataset: ${datamodule.len}")
  println(s"Size of vocab: ${datamodule.vocabSize}")

  def cropInput(tensorPair: (Tensor, Tensor)): Tensor = tensorPair._1

  def runDataset(name: String, dataset: NerDataset, printer: Printer, index: Int): Unit = {
    val inputs = new JHashMap[String, OnnxTensor]() // May need several of these
    val timer = new Timer("The timer")
    val times = dataset.indices.map { index =>
      val tensorPair = dataset.applyLong(index)
      val cropped = cropInput(tensorPair)
      val longArray = cropped.getDataAsLongArray()
      val longBuffer = LongBuffer.wrap(longArray)
      val shape = cropped.shape
      val onnxTensor = OnnxTensor.createTensor(ortEnvironment, longBuffer, shape)

      inputs.put("input", onnxTensor)

      val result = timer.time {
        session.run(inputs).autoClose { outputs =>
          val output = outputs
              .get("output").get
              .getValue
              .asInstanceOf[Array[Array[Array[Float]]]]
              .head
              .flatten

          output
        }
      }
      printer.println(result.map(value => f"$value%1.8f").mkString(", ")) // can't be double
      timer.getElapsed
    }
    println(f"  Mean $name sample time $index: ${timer.mean(times)}%.8f")
    println(f"Stddev $name sample time $index: ${timer.stddev(times)}%.8f")
  }

  def newPrinter(index: Int): Printer = {
    if (debug)
      new FakePrinter()
    else
      new RealPrinter(FileUtils.newPrintWriterFromFile(new File(s"RunParOnnx-$index.out")))
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
