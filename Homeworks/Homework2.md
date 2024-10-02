# Homework 2
## The hands-on project for CS441 is divided into three homeworks to build a [Large Language Model (LLM)](https://en.wikipedia.org/wiki/Large_language_model) from scratch. In the first homework students implemented an LLM encoder and computed the embedding vectors of the input text using massively parallel distributed computations in the cloud. The goal of the second homework is to train the decoder using a neural network library as part of a cloud-based computational platform called Spark, so that this trained model can be used to generate text. Later, in the third, final homework students will create an LLM-based generative system using [Amazon Bedrock](https://aws.amazon.com/bedrock) or their own trained LLM to respond to clients' requests using cloud-deployed lambda functions. Much of the background information is based on the book [Build a Large Language Model (From Scratch)](https://github.com/rasbt/LLMs-from-scratch) that provides an example of the LLM implementation in Python and it is available from [Safari Books Online](https://learning.oreilly.com/videos/build-a-large/) that you can access with your academic subscription. All images in this homework description are used from this book.
### The goal of this homework is for students to gain experience with solving a distributed computational problem using cloud computing technologies, specifically, Spark on the AWS EMR.
### Grade: 15%
#### This Git repo contains the homework description that uses an open-source implementation of [deeplearning4j for Spark](https://deeplearning4j.konduit.ai/spark/tutorials/dl4j-on-spark-quickstart) that is a suite of tools for training deep learning models using Spark. 

## Preliminaries
As part of the first homework assignment students learned how to create and manage Git project repository, create an application in Scala, create tests using widely popular Scalatest framework, and expand on the provided SBT build and run script for their applications and they learned a Map/Reduce framework. First things first, if you haven't done so, you must create your account at [Github](https://github.com/), a Git repo management system. Please make sure that you write your name in your README.md in your repo as it is specified on the class roster. Since it is a large class, please use your UIC email address for communications and for signing your projects and you should avoid using emails from other accounts like funnybunny2003@gmail.com. As always, the homeworks class' Teams channel is the preferred way to exchange information and ask questions. If you don't receive a response within a few hours, please contact your TA or the professor by tagging our names. If you use emails it may be a case that your direct emails went to the spam folder.

Next, if you haven't done so, you will install [IntelliJ](https://www.jetbrains.com/student/) with your academic license, the JDK, the Scala runtime and the IntelliJ Scala plugin and the [Simple Build Toolkit (SBT)](https://www.scala-sbt.org/1.x/docs/index.html) and make sure that you can create, compile, and run Java and Scala programs. Please make sure that you can run [various Java tools from your chosen JDK between versions 8 and 22](https://docs.oracle.com/en/java/javase/index.html).

In all homeworks you will use logging and configuration management frameworks and you will comment your code extensively and supply logging statements at different logging levels (e.g., TRACE, INFO, WARN, ERROR) to record information at some salient points in the executions of your programs. All input configuration variables/parameters must be supplied through configuration files -- hardcoding these values in the source code is prohibited and will be punished by taking a large percentage of points from your total grade! You are expected to use [Logback](https://logback.qos.ch/) and [SLFL4J](https://www.slf4j.org/) for logging and [Typesafe Conguration Library](https://github.com/lightbend/config) for managing configuration files. These and other libraries should be imported into your project using your script [build.sbt](https://www.scala-sbt.org).

When creating your Scala applications you should avoid using **var**s and **while/for** loops that iterate over collections using [induction variables](https://en.wikipedia.org/wiki/Induction_variable). Instead, you should learn to use the monadic collection methods **map**, **flatMap**, **foreach**, **filter** and many others with lambda functions, which make your code linear and easy to understand. Also, you should avoid mutable variables that expose the internal states of your modules at all cost. Points will be deducted for having unreasonable **var**s and inductive variable loops without explanation why mutation is needed in your code unless it is confined to method scopes - you can always do without using mutable states.

## Overview
All three homeworks are created under the general umbrella of a course project that requires CS441 students to create and train an LLM using cloud computing tools and frameworks, which is an extremely valuable skill in today's AI-driven economy. The first phase of the project was to build an LLM by preparing and sampling the input data and learning vector embeddings  that is a term designating the conversion of input categorical text data into a vector format of the continuous real values and implement the attention mechanism for LLMs whereas the second phase involves implementing the attention mechanism with a training loop using Spark and evaluating the resulting model. In this second homework, you will create a distributed program for parallel processing of the large corpus of data using vector embedding that you obtained in the previous homework to learn an LLM with an attention mechanism.

This and all future homework scripts are written using a retroscripting technique, in which the homework outlines are generally and loosely drawn, and the individual students improvise to create the implementation that fits their refined objectives. In doing so, students are expected to stay within the basic requirements of the homework while free to experiment. Asking questions is important to clarify the requirements or to solve problems, so please ask away at [MS Teams](https://teams.microsoft.com/l/team/19%3A9rv9jqRlilNpSrbWQYfv94QkA-KpnOg3B2xOy7RUpM01%40thread.tacv2/conversations?groupId=60ea78dc-5092-47cd-9117-2bd5a5e35d99&tenantId=e202cd47-7a56-4baa-99e3-e3b71a7c77dd)!

## Functionality
Your homework assignment is to create a program for parallel distributed processing of big data to create an LLM, so that you can then use it to generate text in response to a query. In the first homework you already selected a dataset for your work, so if you haven't done so please consult the description of homework 1. Of course, many datasets are too large for a homework and may require significant computational resources, so students should carve a manageable subset of the data. Previous experiments show that it is possible to build a somewhat useful LLM with less than ten gigabytes of data depending on the quality of the data. However, as attractive and cool as it may be to create a useful LLM it is not the main goal of these homeworks that are designed specifically for learning cloud computing tools and frameworks.

The ultimate goal of the homeworks is to produce and use LLMs to generate text using the theory and practice of cloud computing. Let us begin with the end in mind to show how we can use an LLM if it is somehow obtained from the input data. Suppose for brevity that there is only one following sentence in the text corpus: "the cat sat on the mat." When the user submit the query "the cat" via some RPC invocation it is transmitted to the server that hosts a learned LLM. The query is tokenized and embedded into the vector space of the LLM. Using the method ```argMax``` of the object ```Nd4j``` the model returns the token that has the maximum probability match to follow the words of the query, i.e., the word "sat" and the resulting sentence becomes "the cat sat" and the process continues with the resulting sentence becoming the next input query. Upon completion of the process the resulting generated sentence is likely going to be the original sentence. The java-like pseudocode for generating a sentence is shown below.

```java
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.io.File;
import java.io.IOException;

public class TransformerModel {
    // Method to load the pretrained model from disk
    public static MultiLayerNetwork loadPretrainedModel(String modelPath) throws IOException {
        // Load the pretrained model from the specified file
        File file = new File(modelPath);
        MultiLayerNetwork model = ModelSerializer.restoreMultiLayerNetwork(file);
        return model;
    }

    // Method to generate the next word based on the query using the pretrained model
    public static String generateNextWord(String[] context, MultiLayerNetwork model) {
        // Tokenize context and convert to embedding (tokenization + embedding is done as part of homework 1)
        INDArray contextEmbedding = tokenizeAndEmbed(context);  // Create embeddings for the input
        // Forward pass through the transformer layers (pretrained)
        INDArray output = model.output(contextEmbedding);
        // Find the word with the highest probability (greedy search) or sample
        int predictedWordIndex = Nd4j.argMax(output, 1).getInt(0);  // Get the index of the predicted word
        return convertIndexToWord(predictedWordIndex);  // Convert index back to word
    }

    // Method to generate a full sentence based on the seed text
    public static String generateSentence(String seedText, MultiLayerNetwork model, int maxWords) {
        StringBuilder generatedText = new StringBuilder(seedText);
        // Initialize the context with the seed text
        String[] context = seedText.split(" ");
        for (int i = 0; i < maxWords; i++) {
            // Generate the next word
            String nextWord = generateNextWord(context, model);
            // Append the generated word to the current text
            generatedText.append(" ").append(nextWord);
            // Update the context with the new word
            context = generatedText.toString().split(" ");
            // If the generated word is an end token or punctuation, stop
            if (nextWord.equals(".") || nextWord.equals("END")) break;
        }

        return generatedText.toString();
    }

    // Helper function to tokenize and embed text (dummy function)
    private static INDArray tokenizeAndEmbed(String[] words) {
        // here we generate a dummy embedding for the input words and you need to use a real LLM
        // in reality, once an LLM is learned you can save and then load the embeddings
        return Nd4j.rand(1, 128);  // Assuming a 128-dimensional embedding per word
    }

    // Helper function to map word index to actual word (dummy function)
    private static String convertIndexToWord(int index) {
        // Example mapping from index to word (based on a predefined vocabulary)
        String[] vocabulary = {"sat", "on", "the", "mat", ".", "END"};
        return vocabulary[index % vocabulary.length];  // Loop around for small example vocabulary
    }

    public static void main(String[] args) throws IOException {
        // Load the pretrained transformer model from file
        String modelPath = "path/to/your/pretrained_model.zip";  // Path to the pretrained model file
        MultiLayerNetwork model = loadPretrainedModel(modelPath);

        // Generate text using the pretrained model
        String query = "The cat";
        String generatedSentence = generateSentence(query, model, 5);  // Generate a sentence with max 5 words
        System.out.println("Generated Sentence: " + generatedSentence);
    }
}
```
You can see that the first step in the method ```main``` is to load a model that is learned from the data obtained in homework 1. In order for this generative process to work the LLM must incorporate parameters or weights that guide the selection of the next word to complete the sentence. We could just use the dot product between the embedding vectors of the query words and all other words in the text corpus as it is done in the basic information retrieval, but the results are atrocious in general. Instead a special mechanism is used called **attention** to incorporate the positions of words in the input to weight the relevancy to all other positions in the same sentence or a bigger context. In our example the word cat is related to the words sat and mat and the previous occurences of the words cat and sat could be used as predictors for the word mat in the same sentence. There are several attention mechanisms and interested students can shop around to choose the ones they like even though it is not important for this homework.

LLMs can be viewed as trained neural networks whose weights are learned through the process that is described in the supplementary book on constructing LLM models from scratch, so we are not going to repeat it here. Instead, we concentrate on the computational steps for learning an LLM in this homework and the first step is to to compute the sliding window data samples with the input shifted by one as shown in the example image below. These data samples are based on the tokenized output from the previous step and they contain training data for predicting the final token given a number of previously occuring tokens, e.g., the word *learn* can be predicted by the previous occurence of the word *LLM*. In general, for certain types of embeddings or models, the sliding window is not necessary: if your task is to map individual tokens to vectors without considering their context (e.g., one-hot encoding), you don’t need a sliding window. However, our task is to construct LLM models where we do consider the context! In the context of the first homework the sliding window is superfluous, but we will require it in this homework.

![img_1.png](img_1.png)

The sliding window algorithm in the context of large language models (LLMs) works to handle long texts that exceed the model's maximum token limit by processing the text in chunks. Here's how it typically functions:
1. Define the Window Size: The model has a maximum number of tokens it can process in one go, say 1024 tokens.
2. Initialize the Window: The first window covers the first 1024 tokens of the input text.
3. Process the Initial Window: The model processes this window to generate output or extract information.
4. Sliding the Window: After processing, the window "slides" forward by a certain number of tokens, typically with some overlap to maintain context. For example, it might move forward by 512 tokens, meaning the next window will cover tokens 513 to 1536.
5. Repeat Until End: This process continues until the window reaches the end of the text.

If the text length is shorter than the maximum token limit, or when the sliding window reaches the end of the text, there are no more tokens to fill the window. Here's how it works:
1. Incomplete Final Window: If there aren't enough tokens left to fill the window completely, the final window may contain fewer tokens than the initial window size. The model processes whatever data remains in this window.
2. Stop Condition: The algorithm detects that there are no more tokens to include in the next window and stops processing. This prevents any further sliding or unnecessary computations.
3. Handling Edge Cases: In cases where the last chunk of text is crucial for context (like summarization or continuation tasks), the overlap between windows ensures that the model retains context from the previous segment, even when the last window is smaller.
   In essence, when there is no more data to slide, the algorithm concludes its processing as it has covered the entire input text.

Consider an input to an LLM where we are processing a batch of sequences of token embeddings. This involves multiple dimensions:
* Batch size: The number of sequences (e.g., sentences or documents) processed simultaneously;
* Sequence length: The number of tokens (words or subwords) in each sequence;
* Embedding size: The dimensionality of the embedding for each token, representing its features (e.g., 512 or 1024 dimensions).

To represent this, we would need a 3D tensor:
* Batch size: N (the number of sequences in a batch);
* Sequence length: L (the number of tokens in each sequence);
* Embedding size: E (the dimensionality of each token's vector).

So, we would represent this data as a tensor of shape (N,L,E). For instance, if:
* N=322 (32 sequences in the batch);
* L=128 (each sequence has 128 tokens);
* E=512 (each token is represented by a 512-dimensional vector).

Hence, we need a tensor of shape (32,128,512), which is a 3D tensor. This goes beyond what a matrix (which can only represent two dimensions) can handle. The term tensor is necessary to describe this multi-dimensional structure.

Below we show an example using Java-like pseudocode to create a sliding window dataset for training with positional embedding. Please keep in mind that you should use the embeddings and tokens that you created as part of homework 1. This code serves demonstration purposes only to describe one of many ways how a sliding window can be constructed for training a neural network. The first part of your homework 2 assignment is to create an algorithm how to parallelize the construction of the sliding window dataset using Spark.
```java
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import java.util.List;

public class SlidingWindowWithPositionalEmbedding {
    // Create sliding windows for inputs and targets with positional embeddings
    public static List<DataSet> createSlidingWindowsWithPositionalEmbedding(String[] tokens, int windowSize) {
        List<DataSet> dataSetList = new ArrayList<>();
        
        for (int i = 0; i <= tokens.length - windowSize; i++) {
            // Extract the input window (windowSize tokens)
            String[] inputWindow = new String[windowSize];
            System.arraycopy(tokens, i, inputWindow, 0, windowSize);

            // Extract the target token (the token right after the window)
            String targetToken = tokens[i + windowSize];

            // Convert input tokens into embeddings
            INDArray inputEmbeddings = tokenizeAndEmbed(inputWindow);  // Embedding for words
            // Add positional embeddings to the word embeddings
            INDArray positionalEmbeddings = computePositionalEmbedding(windowSize);
            INDArray positionAwareEmbedding = inputEmbeddings.add(positionalEmbeddings);

            // Convert the target token into an embedding
            INDArray targetEmbedding = tokenizeAndEmbed(new String[]{targetToken});

            // Add to the dataset (input is the window with positional embeddings, target is the next word)
            dataSetList.add(new DataSet(positionAwareEmbedding, targetEmbedding));
        }

        return dataSetList;
    }

    // Dummy method to simulate tokenization and embedding (replace with actual embedding code)
    private static INDArray tokenizeAndEmbed(String[] tokens) {
        // For simplicity, let's assume each word is embedded as a 1x128 vector
        return Nd4j.rand(tokens.length, 128);  // Generate random embeddings
    }

    // Compute sinusoidal positional embeddings for a given window size
    private static INDArray computePositionalEmbedding(int windowSize) {
        int embeddingDim = 128;  // Dimensionality of word embeddings
        INDArray positionalEncoding = Nd4j.zeros(windowSize, embeddingDim);

        for (int pos = 0; pos < windowSize; pos++) {
            for (int i = 0; i < embeddingDim; i += 2) {
                double angle = pos / Math.pow(10000, (2.0 * i) / embeddingDim);
                positionalEncoding.putScalar(new int[]{pos, i}, Math.sin(angle));
                positionalEncoding.putScalar(new int[]{pos, i + 1}, Math.cos(angle));
            }
        }

        return positionalEncoding;
    }

    public static void main(String[] args) {
        // Example sentence (tokenized)
        String[] sentence = {"The", "cat", "sat", "on", "a", "mat"};

        // Create sliding windows of size 4 with positional embeddings
        int windowSize = 4;
        List<DataSet> slidingWindows = createSlidingWindowsWithPositionalEmbedding(sentence, windowSize);

        // Output the number of sliding windows created
        System.out.println("Number of sliding windows with positional embeddings: " + slidingWindows.size());
    }
}
```

The idea of using Spark to parallelize the construction of the sliding window data is shown in the Java-like pseudocode below.
```java
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.SparkConf;
import java.util.ArrayList;
import java.util.List;

public class SlidingWindowSparkExample {
    public static void main(String[] args) {
        // Set up Spark configuration and context
        SparkConf conf = new SparkConf().setAppName("Sliding Window Dataset").setMaster("local[*]");
        JavaSparkContext sc = new JavaSparkContext(conf);

        // Example input data (could be sentences, tokens, etc.)
        String[] sentences = {
            "The quick brown fox jumps over the lazy dog",
            "This is another sentence for testing sliding windows"
        };

        // Parallelize the input data (convert array to an RDD)
        JavaRDD<String> sentenceRDD = sc.parallelize(Arrays.asList(sentences));

        // Apply the sliding window logic to create the dataset
        JavaRDD<WindowedData> slidingWindowDataset = sentenceRDD.flatMap(sentence -> createSlidingWindows(sentence, 4).iterator());

        // Collect and print the results (for demonstration)
        slidingWindowDataset.collect().forEach(window -> {
            System.out.println("Input: " + Arrays.toString(window.getInput()));
            System.out.println("Target: " + window.getTarget());
        });

        // Stop the Spark context
        sc.stop();
    }
}

class SlidingWindowUtils {
   class WindowedData {
      private String[] input;
      private String target;

      public WindowedData(String[] input, String target) {
         this.input = input;
         this.target = target;
      }

      public String[] getInput() {
         return input;
      }

      public String getTarget() {
         return target;
      }
   }

   // Create sliding windows for a given sentence
   public static List<WindowedData> createSlidingWindows(String sentence, int windowSize) {
      String[] tokens = sentence.split(" ");
      List<WindowedData> windowedDataList = new ArrayList<>();

      // Create sliding windows
      for (int i = 0; i <= tokens.length - windowSize; i++) {
         String[] inputWindow = new String[windowSize];
         System.arraycopy(tokens, i, inputWindow, 0, windowSize);

         String target = tokens[i + windowSize];
         windowedDataList.add(new WindowedData(inputWindow, target));
      }

      return windowedDataList;
   }
}
```

Once the sliding window dataset is produced you can use it to train a neural network locally as it is shown in the Java-like pseudocode below.
```java
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.optimize.api.IterationListener;
import org.deeplearning4j.datasets.iterator.impl.ListDataSetIterator;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.DataSet;
import java.util.List;

public class TrainingWithSlidingWindow {
    public static void main(String[] args) {
        // Create sliding windows with positional embeddings (as shown in the previous example)
        String[] sentence = {"The", "quick", "brown", "fox", "jumps", "over", "the", "lazy", "dog"};
        int windowSize = 4;
        //this part is done using our example about with Spark
        List<DataSet> slidingWindows = SlidingWindowWithPositionalEmbedding.createSlidingWindowsWithPositionalEmbedding(sentence, windowSize);

        // Create a DataSetIterator to feed the sliding windows into the neural network
        int batchSize = 2;  // Specify the batch size
        DataSetIterator dataSetIterator = new ListDataSetIterator<>(slidingWindows, batchSize);

        // Load or create a model (assuming the model has been defined, e.g., TransformerModel.createModel())
        MultiLayerNetwork model = TransformerModel.createModel(128, 64, 10);  // Input size, hidden size, output size

        // Train the model using the DataSetIterator
        int numEpochs = 5;  // Number of epochs to train
        for (int epoch = 0; epoch < numEpochs; epoch++) {
            model.fit(dataSetIterator);
            System.out.println("Completed epoch " + epoch);
        }

       // Specify the file where the model will be saved
       File locationToSave = new File("path/to/your/trained_model.zip");

       // Whether or not to save the updater (optimizer state)
       boolean saveUpdater = true;

       // Save the trained model to the specified file
       ModelSerializer.writeModel(model, locationToSave, saveUpdater);
       System.out.println("Training complete.");
    }
}
```

Once you tested your LLM generation program locally you can start working on migrating the program logic to parallelize the computationally intensive training of the neural network by using the Spark-integrated facilities of DeepLearning4j (DL4J). To run the training of your LLM on [Apache Spark](https://spark.apache.org/) using DL4J, you need to modify the code to take advantage of Spark's distributed capabilities. DL4J has built-in support for distributed training on Spark, which allows you to scale your training across a cluster of machines. Below are the outlines of the steps to adapt the training for a distributed environment using Spark.

1. Set Up Spark with DL4J
You first need to ensure that your environment is set up with Spark and DL4J's Spark dependencies. You will also need to configure your Spark cluster for distributed training and update the project dependencies to include the following: deeplearning4j-core, deeplearning4j-spark, spark-core_<version> where <version> is the one you use in your project and spark-mllib_<version>. You need to install both DL4J-Spark and Spark MLlib dependencies to enable distributed training and data handling with Spark.

Next, configure Spark to run on a cluster by setting up your SparkConf and JavaSparkContext. Here’s one example how you can set it up.
```java
import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaSparkContext;
import org.deeplearning4j.spark.impl.multilayer.SparkDl4jMultiLayer;

public class SparkLLMTraining {

    public static JavaSparkContext createSparkContext() {
        // Configure Spark for local or cluster mode
        SparkConf sparkConf = new SparkConf()
                .setAppName("DL4J-LanguageModel-Spark")
                .setMaster("local[*]");  // For local testing, or use "yarn", "mesos", etc. in a cluster
        
        // Create Spark context
        JavaSparkContext sc = new JavaSparkContext(sparkConf);
        return sc;
    }

    public static void main(String[] args) {
        // Initialize Spark context
        JavaSparkContext sc = createSparkContext();

        // Rest of the code for distributed training
    }
}
```

You can change `setMaster("local[*]")` to the specific cluster configuration you’re using (e.g., `yarn`, `mesos`, or `spark://...` for a Spark standalone cluster).

3. Convert Data to Spark RDD
For Spark, data should be distributed across the cluster. DL4J uses Resilient Distributed Dataset (RDD) to distribute datasets across multiple Spark workers. Next, you can convert your sliding window data into Spark RDDs and then train using DL4J’s `SparkDl4jMultiLayer` or use the results of the previously computed sliding window inputs.

You need to create a JavaRDD from your sliding window data. Spark will distribute this dataset across the cluster.
```java
import org.apache.spark.api.java.JavaRDD;
import org.deeplearning4j.spark.impl.multilayer.SparkDl4jMultiLayer;
import org.nd4j.linalg.dataset.DataSet;

import java.util.List;

public class SparkLLMTraining {

    public static JavaRDD<DataSet> createRDDFromData(List<DataSet> data, JavaSparkContext sc) {
        // Parallelize your data into a distributed RDD
        JavaRDD<DataSet> rddData = sc.parallelize(data);
        return rddData;
    }
}
```

4. Use Distributed Model Training
DL4J provides the `SparkDl4jMultiLayer` class, which is designed to handle training across Spark clusters.
```java
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.optimize.api.IterationListener;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.spark.api.TrainingMaster;
import org.deeplearning4j.spark.impl.multilayer.SparkDl4jMultiLayer;
import org.deeplearning4j.spark.impl.paramavg.ParameterAveragingTrainingMaster;
import org.deeplearning4j.spark.parameterserver.training.SharedTrainingMaster;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.apache.spark.api.java.JavaRDD;

import java.util.List;

public class SparkLLMTraining {

    public static void main(String[] args) {
        // Initialize Spark context
        JavaSparkContext sc = createSparkContext();

        // Create your LLM model using DL4J
        MultiLayerNetwork model = LLMModel.createModel(128, 10);  // Input size and output size

        // Prepare data (you can use the sliding window data from the previous step)
        List<DataSet> windows = SlidingWindowExample.createSlidingWindows(new double[1000], 128, 64, 10);
        JavaRDD<DataSet> rddData = createRDDFromData(windows, sc);

        // Set up the TrainingMaster configuration
        TrainingMaster trainingMaster = new ParameterAveragingTrainingMaster.Builder(32)
                .batchSizePerWorker(32)  // Batch size on each Spark worker
                .averagingFrequency(5)   // Frequency of parameter averaging
                .workerPrefetchNumBatches(2)
                .build();

        // Create a SparkDl4jMultiLayer with the Spark context and model
        SparkDl4jMultiLayer sparkModel = new SparkDl4jMultiLayer(sc, model, trainingMaster);

        // Set listeners to monitor the training progress
        model.setListeners(new ScoreIterationListener(10));

        // Train the model on the distributed RDD dataset
        sparkModel.fit(rddData);

        // Save the model after training
        // ModelSerializer.writeModel(sparkModel.getNetwork(), new File("LLM_Spark_Model.zip"), true);
        
        // Stop the Spark context after training
        sc.stop();
    }
}
```

5. Training Master Options
There are two main strategies for distributed training in DL4J: Parameter Averaging and Shared Parameter Server. The former trains the model on each worker and then averages the parameters after a set number of iterations and uses `ParameterAveragingTrainingMaster` for simple averaging of parameters across workers whereas the latter is a more advanced approach where workers share parameters via a central parameter server, which is useful for large-scale training, this method can be more efficient in some distributed setups - see the example below.
```java
TrainingMaster trainingMaster = new SharedTrainingMaster.Builder(32)
        .batchSizePerWorker(32)
        .workersPerNode(4)  // Define number of workers per node in the cluster
        .build();
```

6. Run the Job on a Cluster
Once the code is set up, you can run this on your Spark cluster. Depending on your environment, you can submit the job using:
```bash
spark-submit --class com.example.SparkLLMTraining \
    --master spark://<your-spark-cluster> \
    --executor-memory 4G \
    --total-executor-cores 4 \
    path-to-your-jar-file.jar
```
This approach enables you to scale the training of your LLM across a cluster using Apache Spark, allowing you to handle larger datasets and benefit from distributed computing.    

Your goal is to produce an LLM file as well as the accompanying statistics and runtime measurements. Of course, you should also test how your learned LLM works to generate texts and you should submit the results of text generation! 

The output of your program is a data file in some format of your choice, e.g., Yaml or CSV with the required statistics. The explanation of the Spark model is given in the main textbook/research papers and covered in class lectures. After creating and testing your Spark program locally, you will deploy it and run it on the AWS Spark EMR - you can find plenty of [documentation online](https://docs.aws.amazon.com/emr/latest/ReleaseGuide/emr-spark.html). Just like for the first homework you will produce a short movie that documents all steps of the deployment and execution of your program with your narration and you will upload this movie to [youtube](www.youtube.com) and you will submit a link to your movie as part of your submission in the README.md file. To produce a movie, you may use an academic version of [Camtasia](https://www.techsmith.com/video-editor.html) or Zoom or some other cheap/free screen capture technology from the UIC webstore or an application for a movie capture of your choice. The captured web browser content should show your login name in the upper right corner of the AWS application and you should introduce yourself in the beginning of the movie speaking into the camera. The display of your passwords and your credit card numbers should be avoided :-).

## Baseline Submission
Your baseline project submission should include your implementation, a conceptual explanation in the document or in the comments in the source code of how your Spark/DL4J processing components work to solve the problem and the documentation that describe the build and runtime process, to be considered for grading. Your should use [markdown](https://github.com/adam-p/markdown-here/wiki/Markdown-Cheatsheet) for your project's Readme.md. Your project submission should include all your source code as well as non-code artifacts (e.g., configuration files), your project should be buildable using the SBT, and your documentation must specify how you paritioned the data and what input/outputs are. Also, when running LLM training using DL4J on Apache Spark, various statistics and performance metrics can be generated to monitor and evaluate the training process. These statistics help in understanding the efficiency, convergence, and performance of the model. You should gather and report the following measurements.

1. Training Loss and Accuracy
- Training Loss: The model's loss function measures how well the model is performing on the training dataset. This is usually a cross-entropy loss for LLMs.
- Training Accuracy: You can monitor the percentage of correctly predicted tokens during training, which indicates how well the model is learning.
```java
model.setListeners(new ScoreIterationListener(10));  // Log every 10 iterations
```
You can implement a custom listener to evaluate the model on the validation dataset after each epoch:
```java
model.setListeners(new EvaluativeListener(validationDataSetIterator, 1, InvocationType.EPOCH_END));
```

2. Gradient Statistics
- Gradient Norms: Monitoring the gradient norms (magnitude of weight updates) helps detect issues such as vanishing or exploding gradients, which can slow down or destabilize training.
- Gradient Histograms: Visualizing the distribution of gradients can provide insights into how the model is updating parameters over time.
You can add custom listeners in DL4J to capture gradient norms:
```java
model.setListeners(new GradientStatsListener(1));  // Log gradient stats every 1 iteration
```
3. Learning Rate
- Learning Rate: Tracking the learning rate helps in monitoring learning rate schedules or adaptive learning rates (e.g., Adam optimizer).
- Decay or Scheduling: In some cases, the learning rate decays over time, so it's important to track how this changes during training.
You can retrieve the learning rate from the optimizer configuration and log it periodically using the following Java-like pseudocode.
```java
System.out.println("Current Learning Rate: " + model.conf().getLearningRate());
```

4. Memory Usage
- Memory Consumption: Track how much memory (RAM and GPU memory) is used during the training process.
- Batch Size: Monitoring the memory usage for different batch sizes can help you optimize training.
Spark has built-in tools such as Spark UI to monitor memory usage for each executor and task.

5. Time per Epoch/Iteration
- Time per Iteration: Measure how long each iteration takes during training. This helps assess if the training process is efficient and whether there are any bottlenecks.
- Time per Epoch: Measure the total time taken for each epoch, which gives a sense of how long the entire training process will take.
You can measure and log the time before and after each epoch:
```java
long startTime = System.currentTimeMillis();
// After the epoch ends
long endTime = System.currentTimeMillis();
System.out.println("Epoch time: " + (endTime - startTime) + "ms");
```

6. Data Shuffling and Partitioning Statistics (Spark)
- Data Shuffling: Track how much data is being shuffled across executors, which can impact performance in distributed training.
- Partitioning: Monitor how data is partitioned across Spark workers. An imbalance in partitions can lead to stragglers (slow workers), negatively impacting training time.

You can use the Spark UI to track shuffling and partition statistics:
- Shuffle Read/Write: Monitors how much data is being transferred between nodes.
- Task Skew: Detects if some tasks are taking significantly longer than others, indicating partitioning issues.

7. CPU and GPU Utilization
- CPU/GPU Utilization: Monitoring how well the CPU or GPU resources are being utilized helps detect under-utilization or bottlenecks in hardware usage.
- Batch Processing Speed: Monitor how fast batches are processed, which can give insight into whether the hardware is being fully used.

8. Spark-Specific Metrics
- Task Execution Times: Measure how long each Spark task takes. This can be accessed through the Spark UI or programmatically.
- Executor Metrics: These include the memory, disk, and CPU usage for each executor.
- Task Failure Rates: Track how often tasks are failing and being retried, as excessive retries can significantly slow down training.

You can access these metrics via Spark’s `SparkContext` or Spark UI.
```java
JavaSparkContext sc = new JavaSparkContext(conf);
System.out.println("Total executors: " + sc.getExecutorMemoryStatus().size());
```

## Collaboration
You can post questions and replies, statements, comments, discussion, etc. on Teams using the corresponding channel. For this homework, feel free to share your ideas, mistakes, code fragments, commands from scripts, and some of your technical solutions with the rest of the class, and you can ask and advise others using Teams on where resources and sample programs can be found on the Internet, how to resolve dependencies and configuration issues. When posting question and answers on Teams, please make sure that you selected the appropriate channel, to ensure that all discussion threads can be easily located. Active participants and problem solvers will receive bonuses from [the big brother](https://www.cs.uic.edu/~drmark/) :-) who is watching your exchanges. However, *you must not describe your mappers/reducers or the CORBA architecture or other specific details related to how you construct your models!*

## Git logistics
**This is an individual homework.** Please remember to grant a read access to your repository to your TA and your instructor. You can commit and push your code as many times as you want. Your code will not be visible and it should not be visible to other students - your repository should be private. Announcing a link to your public repo for this homework or inviting other students to join your fork for an individual homework before the submission deadline will result in losing your grade. For grading, only the latest commit timed before the deadline will be considered. **If your first commit will be pushed after the deadline, your grade for the homework will be zero**. For those of you who struggle with the Git, I recommend a book by Ryan Hodson on Ry's Git Tutorial. The other book called Pro Git is written by Scott Chacon and Ben Straub and published by Apress and it is [freely available](https://git-scm.com/book/en/v2/). There are multiple videos on youtube that go into details of the Git organization and use.

Please follow this naming convention to designate your authorship while submitting your work in README.md: "Firstname Lastname" without quotes, where you specify your first and last names **exactly as you are registered with the University system**, as well as your UIC.EDU email address, so that we can easily recognize your submission. I repeat, make sure that you will give both your TA and the course instructor the read/write access to your *private forked repository* so that we can leave the file feedback.txt in the root of your repo with the explanation of the grade assigned to your homework.

## Discussions and submission
As it is mentioned above, you can post questions and replies, statements, comments, discussion, etc. on Teams. Remember that you cannot share your code and your solutions privately, but you can ask and advise others using Teams and StackOverflow or some other developer networks where resources and sample programs can be found on the Internet, how to resolve dependencies and configuration issues. Yet, your implementation should be your own and you cannot share it. Alternatively, you cannot copy and paste someone else's implementation and put your name on it. Your submissions will be checked for plagiarism. **Copying code from your classmates or from some sites on the Internet will result in severe academic penalties up to the termination of your enrollment in the University**.


## Submission deadline and logistics
Sunday, November, 3, 2024 at 10PM CST by submitting the link to your homework repo in the Teams Assignments channel. Your submission repo will include the code for the program, your documentation with instructions and detailed explanations on how to assemble and deploy your program along with the results of your program execution, the link to the video and a document that explains these results based on the characteristics and the configuration parameters you chose for your experiments, and what the limitations of your implementation are. Again, do not forget, please make sure that you will give both your TAs and your instructor the read access to your private repository. Your code should compile and run from the command line using the commands **sbt clean compile test** and **sbt clean compile run**. Also, you project should be IntelliJ friendly, i.e., your graders should be able to import your code into IntelliJ and run from there. Use .gitignore to exlude files that should not be pushed into the repo.

## There will be no deadline extension for this and the following homeworks.
You have one month to complete this homework, so use your time wisely. No extension will be granted after the specified deadline.

## Evaluation criteria
- the maximum grade for this homework is 15%. Points are subtracted from this maximum grade: for example, saying that 2% is lost if some requirement is not completed means that the resulting grade will be 15%-2% => 13%; if the core homework functionality does not work or it is not implemented as specified in your documentation, your grade will be zero;
- only some basic Spark examples from some repos are given and nothing else is done: zero grade;
- using Python or some other language instead of Scala: 10% penalty;
- having less than five unit and/or integration scalatests: up to 10% lost;
- missing comments and explanations from your program with clarifications of your design rationale: up to 10% lost;
- logging is not used in your programs: up to 5% lost;
- hardcoding the input values in the source code instead of using the suggested configuration libraries: up to 5% lost;
- for each used *var* for heap-based shared variables or mutable collections without explicitly stated reasons: 0.3% lost;
- for each used *while* or *for* or other loops with induction variables to iterate over a collection: 0.5% lost;
- no instructions in README.md on how to install and run your program: up to 10% lost;
- the program crashes without completing the core functionality: up to 15% lost;
- the documentation exists but it is insufficient to understand your program design and models and how you assembled and deployed all components of your mappers and reducers: up to 15% lost;
- the minimum grade for this homework cannot be less than zero.

That's it, folks!
