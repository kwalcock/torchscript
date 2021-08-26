package demo

import org.pytorch.IValue
import org.pytorch.Module
import org.pytorch.Tensor

object ScalaApp extends App {
  val mod: Module = Module.load("./demo/data/demo-model.pt1");
  val data: Tensor =
    Tensor.fromBlob(
      Array[Int](1, 2, 3, 4, 5, 6), // data
      Array[Long](2, 3) // shape
    )
  val result: IValue = mod.forward(IValue.from(data), IValue.from(3.0));
  val output: Tensor = result.toTensor();
  println("shape: " + output.shape().mkString(", "));
  println("data: " + output.getDataAsFloatArray().mkString(", "));

  // Workaround for https://github.com/facebookincubator/fbjni/issues/25
  System.exit(0);
}
