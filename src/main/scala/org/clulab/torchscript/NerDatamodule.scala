package org.clulab.torchscript

class NerDatamodule(datasetPath: String, embeddingPath: String) {
  val vocab: Map[String, Int] = NerVocab(embeddingPath)

  val trainDataset: NerDataset = new NerDataset(datasetPath + "eng-2col.train", vocab)
  val   valDataset: NerDataset = new NerDataset(datasetPath + "eng-2col.testa", vocab)
  val  testDataset: NerDataset = new NerDataset(datasetPath + "eng-2col.testb", vocab)

  val size: Int = trainDataset.length + valDataset.length + testDataset.length
  val numClasses: Int = trainDataset.labelLength
  val vocabSize: Int = trainDataset.vocab.size

  def len: Int = size
}
