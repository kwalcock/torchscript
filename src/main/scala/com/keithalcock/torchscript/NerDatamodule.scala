package com.keithalcock.torchscript

class NerDatamodule(datasetPath: String, embeddingPath: String) {
  // TODO: Don't load the embeddings three times!
  val trainDataset = new NerDataset(datasetPath + "eng-2col.train", embeddingPath)
  val   valDataset = new NerDataset(datasetPath + "eng-2col.testa", embeddingPath)
  val  testDataset = new NerDataset(datasetPath + "eng-2col.testb", embeddingPath)

  val size = trainDataset.length + valDataset.length + testDataset.length
  val numClasses = trainDataset.labels.size
  val vocabSize = trainDataset.vocab.size

  def len: Int = size
}
