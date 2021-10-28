package org.clulab.torchscript

import ai.onnxruntime.{OnnxTensor, OrtEnvironment, OrtSession}
import org.clulab.torchscript.utils.{AppUtils, Timer}
import org.clulab.torchscript.utils.Closer.AutoCloser
import org.pytorch.Tensor

import java.nio.LongBuffer
import java.util.{HashMap => JHashMap}

object RunSerOnnx extends App {
  val datapath = "../data/ner-conll/"
  val embedpath = "../data/glove.840B.300d.10f.txt"
  val modelpath = "../data/model.onnx"

  AppUtils.showStatus()

  val ortEnvironment = OrtEnvironment.getEnvironment
  val session = ortEnvironment.createSession(modelpath, new OrtSession.SessionOptions)
  val inputs = new JHashMap[String, OnnxTensor]()

  val datamodule = new NerDatamodule(datapath, embedpath)
  println(s"Size of dataset: ${datamodule.len}")
  println(s"Size of vocab: ${datamodule.vocabSize}")

  def cropInput(tensorPair: (Tensor, Tensor)): Tensor = tensorPair._1

  val timer = new Timer("The timer")

  def runDataset(name: String, dataset: NerDataset): Unit = {
    val times = dataset.indices.map { index =>
      // TODO: This can be done more quickly, but it isn't timed anyway.
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
      // println(result.map(value => f"$value%1.8f").mkString(", "))
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
