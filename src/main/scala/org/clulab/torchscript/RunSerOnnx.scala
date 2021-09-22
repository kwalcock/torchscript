package org.clulab.torchscript

import ai.onnxruntime.{OnnxTensor, OrtEnvironment, OrtSession}
import org.clulab.torchscript.utils.{AppUtils, Timer}
import org.clulab.torchscript.utils.Closer.AutoCloser
import org.pytorch.Tensor

import java.nio.IntBuffer
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
    val times = dataset.map { tensorPair =>
      val cropped = cropInput(tensorPair)
      val intArray = cropped.getDataAsIntArray()
      val intBuffer = IntBuffer.wrap(intArray)
      val shape = cropped.shape

      val result = timer.time {
        val onnxTensor = OnnxTensor.createTensor(ortEnvironment, intBuffer, shape)
        inputs.put("input", onnxTensor)

        session.run(inputs).autoClose { outputs =>
           // manipulate the results
          outputs.forEach { output =>
            val (name, value) = (output.getKey, output.getValue)
            println(name)
            println(value)
          }
        }
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
