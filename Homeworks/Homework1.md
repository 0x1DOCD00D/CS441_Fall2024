# Homework 1
## The hands-on project for CS441 is divided into three homeworks to build a [Large Language Model (LLM)](https://en.wikipedia.org/wiki/Large_language_model) from scratch. The first homework requires students to implement an LLM encoder using massively parallel distributed computations in the cloud, the goal of the second homework is to train the decoder using a neural network library as part of a cloud-based computational platform called Spark and the third, final homework is to create an LLM-based generative system using [Amazon Bedrock](https://aws.amazon.com/bedrock) to respond to clients' requests using cloud-deployed lambda functions. Much of the background information is based on the book [Build a Large Language Model (From Scratch)](https://github.com/rasbt/LLMs-from-scratch) that provides an example of the LLM implementation in Python and it is available from [Safari Books Online](https://learning.oreilly.com/videos/build-a-large/) that you can access with your academic subscription. All images in this homework description are used from this book.
### The goal of this homework is for students to gain experience with solving a distributed computational problem using cloud computing technologies. The main textbook group (option 1) will design and implement an instance of the map/reduce computational model using AWS EMR whereas the alternative textbook group (option 2) will use the CORBA model. You can check your textbook option in the corresponding column of the gradebook on the Blackboard.
### Grade: 15%
#### This Git repo contains the homework description that uses an open-source implementation of a [tokenizer toolkit JTokkit](https://github.com/knuddelsgmbh/jtokkit?tab=readme-ov-file) and [deeplearning4j](https://github.com/deeplearning4j/deeplearning4j) that is a suite of tools for deploying and training deep learning models using the JVM. Students may invest some time to learn how the use cases of JTokkit that are shown in [tests](https://github.com/knuddelsgmbh/jtokkit/tree/main/lib/src/test/java/com/knuddels/jtokkit), however, it is not required for completing this homework.

## Preliminaries
As part of this first homework assignment students are going to learn how to create and manage Git project repository, create an application in Scala, create tests using widely popular Scalatest framework, and expand on the provided SBT build and run script for their applications. Your job as a student in this course is to create randomly generated graphs that represent big data, apply specially designed perturbation operators to produce modified graphs, parse and analyze them to determine differences between the original and perturbed graphs by processing them in the cloud using distributed big data analysis frameworks.

First things first, if you haven't done so, you must create your account at [Github](https://github.com/), a Git repo management system. Please make sure that you write your name in your README.md in your repo as it is specified on the class roster. Since it is a large class, please use your UIC email address for communications and for signing your projects and you should avoid using emails from other accounts like funnybunny2003@gmail.com. As always, the homeworks class' Teams channel is the preferred way to exchange information and ask questions. If you don't receive a response within a few hours, please contact your TA or the professor by tagging our names. If you use emails it may be a case that your direct emails went to the spam folder.

Next, if you haven't done so, you will install [IntelliJ](https://www.jetbrains.com/student/) with your academic license, the JDK, the Scala runtime and the IntelliJ Scala plugin and the [Simple Build Toolkit (SBT)](https://www.scala-sbt.org/1.x/docs/index.html) and make sure that you can create, compile, and run Java and Scala programs. Please make sure that you can run [various Java tools from your chosen JDK between versions 8 and 22](https://docs.oracle.com/en/java/javase/index.html).

In this and all consecutive homeworks you will use logging and configuration management frameworks. You will comment your code extensively and supply logging statements at different logging levels (e.g., TRACE, INFO, WARN, ERROR) to record information at some salient points in the executions of your programs. All input configuration variables/parameters must be supplied through configuration files -- hardcoding these values in the source code is prohibited and will be punished by taking a large percentage of points from your total grade! You are expected to use [Logback](https://logback.qos.ch/) and [SLFL4J](https://www.slf4j.org/) for logging and [Typesafe Conguration Library](https://github.com/lightbend/config) for managing configuration files. These and other libraries should be imported into your project using your script [build.sbt](https://www.scala-sbt.org). These libraries and frameworks are widely used in the industry, so learning them is the time well spent to improve your resumes. Also, please set up your account with [AWS Educate](https://aws.amazon.com/education/awseducate/). Using your UIC email address may enable you to receive free credits for running your jobs in the cloud. Preferably, you should create your developer account for a small fee of approximately $29 per month to enjoy the full range of AWS services. Some students I know created business accounts to receive better options from AWS and some of them even started companies while taking this course using their AWS account and applications they created and hosted there!

From many example projects on Github you can see how to use Scala to create a fully functional (not imperative) implementation with subprojects and tests. As you see from the StackOverflow survey, knowledge of Scala is highly paid and in great demand, and it is expected that you pick it relatively fast, especially since it is tightly integrated with Java. I recommend using the book on [Programming in Scala Fourth and Fifth Editions by Martin Odersky et al](https://www.amazon.com/Programming-Scala-Fourth-Updated-2-13-ebook/dp/B082T2ZNJG). You can obtain this book using the academic subscription on [Safari Books Online](https://learning.oreilly.com/home-new/). There are many other books and resources available on the Internet to learn Scala. Those who know more about functional programming can use the book on [Functional Programming in Scala published in 2023 by Michael Pilquist, Rúnar Bjarnason, and Paul Chiusano](https://www.manning.com/books/functional-programming-in-scala-second-edition?new=true&experiment=C).

When creating your Map/Reduce in Scala or CORBA program code in C++ you should avoid using **var**s and **while/for** loops that iterate over collections using [induction variables](https://en.wikipedia.org/wiki/Induction_variable). Instead, you should learn to use collection methods **map**, **flatMap**, **foreach**, **filter** and many others with lambda functions, which make your code linear and easy to understand as we studied it in class. Also, avoid mutable variables that expose the internal states of your modules at all cost. Points will be deducted for having unreasonable **var**s and inductive variable loops without explanation why mutation is needed in your code unless it is confined to method scopes - you can always do without using mutable states.

## Overview
All three homeworks are created under the general umbrella of a course project that allows students to create and train an LLM using cloud computing tools and frameworks, which is an extremely valuable skill in today's AI-driven economy. The first phase of the project is to build an LLM by preparing and sampling the input data and implementing the attention mechanism for a given LLM architecture whereas the second phase involves implementing the training loop and evaluating the resulting model. In this homework, you will create a distributed program for parallel processing of the large corpus of data starting with data embedding that is a term designating the conversion of input categorical text data into a vector format of the continuous real values and implement the attention mechanism for LLMs.

This and all future homework scripts are written using a retroscripting technique, in which the homework outlines are generally and loosely drawn, and the individual students improvise to create the implementation that fits their refined objectives. In doing so, students are expected to stay within the basic requirements of the homework while free to experiment. Asking questions is important to clarify the requirements or to solve problems, so please ask away at [MS Teams](https://teams.microsoft.com/l/team/19%3A9rv9jqRlilNpSrbWQYfv94QkA-KpnOg3B2xOy7RUpM01%40thread.tacv2/conversations?groupId=60ea78dc-5092-47cd-9117-2bd5a5e35d99&tenantId=e202cd47-7a56-4baa-99e3-e3b71a7c77dd)!

## Functionality
Your homework assignment is to create a program for parallel distributed processing of large corpus of text. First things first, you should select a dataset for your work. Open-source text corpora are collections of texts that are freely available for public use, modification, and distribution. These corpora are typically released under open licenses that allow researchers, developers, and linguists to access, study, and use them for various purposes, such as natural language processing (NLP), machine learning, language modeling, and linguistic analysis. I recommend that you choose your text corpus from the following datasets.

* [OpenWebText2](https://openwebtext2.readthedocs.io/en/latest/) contains Reddit submissions from 2005 up until April 2020;
* [Common Crawl](https://commoncrawl.org/) is a massive corpus of web pages crawled from the internet, widely used for NLP tasks;
* [WikiText](https://paperswithcode.com/dataset/wikitext-2) is a dataset of Wikipedia articles, often used in language modeling tasks;
* [Project Gutenberg](https://www.gutenberg.org/) contains a large collection of public domain books, often used for text mining and NLP research;
* [OpenSubtitles](https://www.opensubtitles.com/) is a dataset of movie subtitles in multiple languages, useful for tasks like machine translation and dialogue modeling.

Of course, many datasets are too large for a homework and may require significant computational resources, so students should carve a manageable subset of the data. Previous experiments show that it is possible to build a useful LLM with ten Gb of data depending on the quality of the data.

The main steps of this homework are the following. First, you split the initial text corpus in shards for parallel processing. The size of the shard can be chosen experimentally. Next, for each shard you will convert the text into numerical tokens using the [Byte Pair Encoding (BPE) JTokkit](https://github.com/knuddelsgmbh/jtokkit?tab=readme-ov-file) as it is shown in the image from the LLM book below. You can decide how to handle special context tokens.

![img.png](img.png)

Next, you need to compute the sliding window data samples with the input shifted by one as shown in the example image below. These data samples are based on the tokenized output from the previous step and they contain training data for predicting the final token given a number of previously occuring tokens, e.g., the word *learn* can be predicted by the previous occurence of the word *LLM*.

![img_1.png](img_1.png)

In the context of constructing an LLM, a [tensor](https://en.wikipedia.org/wiki/Tensor) is a fundamental data structure used to represent multi-dimensional arrays of data. Tensors generalize matrices (2D arrays) to higher dimensions, and are key components in the field of machine learning, particularly in the implementation of neural networks, including LLMs like GPT models. The term [matrix](https://en.wikipedia.org/wiki/Matrix_(mathematics)) is not sufficient since we need to work with data that has more than two dimensions. Matrices are limited to 2D arrays, meaning they have rows and columns. However, in many machine learning and deep learning applications like the one we are creating in this course, especially in neural networks like those used in LLMs, the data is inherently multi-dimensional, requiring more than just rows and columns to represent it. This is where the term tensor becomes necessary.

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

Next, you will compute token embeddings, a fancy term for converting token IDs that you computed in the previous steps into embedding vectors. An embedding vector for a token in LLMs represents a dense, continuous, high-dimensional vector that encodes the semantic meaning of that token. It is part of the model's internal representation of words or tokens in a way that allows them to capture the relationships between words, such as their meanings, context, and syntactic roles. At this point, text is divided into smaller units called tokens. These can be individual words, subwords, or even characters, depending on the tokenization strategy. Then each token is mapped to a numerical vector through a process called embedding. The embedding vector typically has hundreds or thousands of dimensions, each representing different aspects of the token's meaning or usage in context. These embeddings are learned during the training process of the model. As the model processes vast amounts of text, it adjusts the embedding vectors so that tokens with similar meanings are placed closer together in this high-dimensional space. The embedding vector thus allows the model to handle tokens in a more context-sensitive way. Rather than treating each token as an isolated entity, the embedding provides a rich, nuanced representation that reflects the token's meaning within the model's learned knowledge.

For example, the words "king" and "queen" would have similar but distinct embeddings, capturing both their semantic similarity (royalty) and their differences (biological sexes). Similarly, context can modify the embedding of a word, so "bank" in "river bank" will have a different embedding than "bank" in "financial institution." If represented in a multidimensional space as vectors the words "king" and "queen" will be almost collinear or the cos(angle(vector(king), vector(queen)) will be close to 1. The steps are schematically represented in the image below.

![img_2.png](img_2.png)

Initially, all embeddings are given random values/weights and these weights will be updated later as part of the learning/training process using neural networks with the backpropagation learning algorithm. It is assumed that the vocabulary size of your dataset could be at least a few thousand words and the number of the output embedding dimensions can be determined experimentally, e.g., three dimensions may be a very small and unrealistic number whereas choosing millions of dimensions may be computationally prohibitive for this homework. In Deeplearning4j you can compute token embeddings using [Word2Vec](https://en.wikipedia.org/wiki/Word2vec) or other embedding models. Below are the steps shown in Java-like pseudocode compiled from various published programs to compute token embeddings using Word2Vec in Deeplearning4j. [Goldman Sacks](https://github.com/goldmansachs/MRWord2Vec) published an example of the map/reduce implementation of token embedding and you can study and emulate it. Once the embeddings are computed you can compute lists of semantically related words based on the closeness between their vector embeddings.

```java
public class Word2VecExample {

    public static void main(String[] args) throws Exception {
        // Load text data
        File file = new File("path_to_text_corpus.txt"); // Replace with your file path
        LineSentenceIterator sentenceIterator = new LineSentenceIterator(file);

        // Tokenizer configuration
        TokenizerFactory tokenizerFactory = new DefaultTokenizerFactory();

        // Build Word2Vec model
        Word2Vec word2Vec = new Word2Vec.Builder()
                .minWordFrequency(5)      // Minimum frequency of words to be included
                .iterations(10)           // Number of training iterations
                .layerSize(100)           // Size of the word vectors
                .seed(42)
                .windowSize(5)            // Context window size for embeddings
                .iterate(sentenceIterator)
                .tokenizerFactory(tokenizerFactory)
                .build();

        // Train the model
        word2Vec.fit();

        // Save the model for later use
        WordVectorSerializer.writeWord2VecModel(word2Vec, new File("word2vec_model.bin"));

        // Get embedding for a token
        double[] embedding = word2Vec.getWordVector("example"); // Replace "example" with your token

        // Print embedding
        if (embedding != null) {
            System.out.println("Embedding for 'example': ");
            for (double value : embedding) {
                System.out.print(value + " ");
            }
        } else {
            System.out.println("Word not in the vocabulary!");
        }
    }
}
```

Your goal is to produce files with token embeddings and various statistics about the data. First, you will compute a Yaml or an CSV file that shows the vocabulary as the list of words, their numerical tokens and the frequency of occurences of these words in the text corpus. Second, for each token embedding you can output other tokens/words that are semantically close them it based on the computed vector embeddings. Finally, you will produce an estimate of the optimal number of dimensions based on your experiments. Determining the optimal number of dimensions (also known as the embedding size or latent vector size) in embedding models like Word2Vec in Deeplearning4j involves balancing between capturing enough semantic information and keeping the model computationally efficient. There’s no one-size-fits-all rule, but here are strategies to guide the selection of the optimal number of dimensions, one of which is called a set of word analogy tasks, word similarity tasks, or clustering tasks to see how well the learned vectors capture word relationships. Some datasets like [WordSim-353](https://aclweb.org/aclwiki/WordSimilarity-353_Test_Collection_(State_of_the_art)) or [other datasets](https://github.com/vecto-ai/word-benchmarks) can be used for this purpose.

* Word Analogy: Check how well your embeddings capture relationships like "king" - "man" + "woman" ≈ "queen".
* Word Similarity: Evaluate cosine similarity between known pairs of words (e.g., "cat" and "dog") and see if their embeddings capture the right degree of similarity.

### Assignment for the main textbook group
Your job is to create the mapper and the reducer for each task, explain how they work, and then to implement them and run on the big data graphs that you will generate using your predefined configuration parameters. The output of your map/reduce is a Yaml or an CSV file with token embeddings and the required statistics. The explanation of the map/reduce model is given in the main textbook and covered in class lectures.

You will create and run your software application using [Apache Hadoop](http://hadoop.apache.org/), a framework for distributed processing of large data sets across multiple computers (or even on a single node) using the map/reduce model. Next, after creating and testing your map/reduce program locally, you will deploy it and run it on the Amazon Elastic MapReduce (EMR) - you can find plenty of [documentation online](https://aws.amazon.com/emr). You will produce a short movie that documents all steps of the deployment and execution of your program with your narration and you will upload this movie to [youtube](www.youtube.com) and you will submit a link to your movie as part of your submission in the README.md file. To produce a movie, you may use an academic version of [Camtasia](https://www.techsmith.com/video-editor.html) or Zoom or some other cheap/free screen capture technology from the UIC webstore or an application for a movie capture of your choice. The captured web browser content should show your login name in the upper right corner of the AWS application and you should introduce yourself in the beginning of the movie speaking into the camera. The display of your passwords and your credit card numbers should be avoided when possible :-).

### Assignment for the alternative textbook group
Your job is to create the distributed objects using [omniOrb CORBA framework](http://omniorb.sourceforge.net/omni42/omniORB/) for each task, explain how they work, and then to implement them and run on the generated log message dataset. The output of your distributed system is a Yaml or an CSV file with the required statistics. The explanation of the CORBA is given in the alternative textbook in Chapter 7 -Guide to Reliable Distributed Systems: Building High-Assurance Applications and Cloud-Hosted Services 2012th Edition by Kenneth P. Birman. You can complete your implementation using C++ or Python.

Next, after creating and testing your program locally, you will deploy it and run it on the AWS EC2 IaaS. You will produce a short movie that documents all steps of the deployment and execution of your program with your narration and you will upload this movie to [youtube](www.youtube.com) and you will submit a link to your movie as part of your submission in the README.md file. To produce a movie, you may use an academic version of [Camtasia](https://www.techsmith.com/video-editor.html) or some other cheap/free screen capture technology from the UIC webstore or an application for a movie capture of your choice. The captured web browser content should show your login name in the upper right corner of the AWS application and you should introduce yourself in the beginning of the movie speaking into the camera.

## Baseline Submission
Your baseline project submission should include your implementation, a conceptual explanation in the document or in the comments in the source code of how your mapper and reducer work to solve the problem for Option 1 group or how your CORBA distributed object work for Option 2 group, and the documentation that describe the build and runtime process, to be considered for grading. Your should use [markdown](https://github.com/adam-p/markdown-here/wiki/Markdown-Cheatsheet) for your project's Readme.md. Your project submission should include all your source code as well as non-code artifacts (e.g., configuration files), your project should be buildable using the SBT, and your documentation must specify how you paritioned the data and what input/outputs are.

## Collaboration
You can post questions and replies, statements, comments, discussion, etc. on Teams using the corresponding channel. For this homework, feel free to share your ideas, mistakes, code fragments, commands from scripts, and some of your technical solutions with the rest of the class, and you can ask and advise others using Teams on where resources and sample programs can be found on the Internet, how to resolve dependencies and configuration issues. When posting question and answers on Teams, please make sure that you selected the appropriate channel, to ensure that all discussion threads can be easily located. Active participants and problem solvers will receive bonuses from [the big brother](https://www.cs.uic.edu/~drmark/) :-) who is watching your exchanges. However, *you must not describe your mappers/reducers or the CORBA architecture or other specific details related to how you construct your models!*

## Git logistics
**This is an individual homework.** Please remember to grant a read access to your repository to your TA and your instructor. You can commit and push your code as many times as you want. Your code will not be visible and it should not be visible to other students - your repository should be private. Announcing a link to your public repo for this homework or inviting other students to join your fork for an individual homework before the submission deadline will result in losing your grade. For grading, only the latest commit timed before the deadline will be considered. **If your first commit will be pushed after the deadline, your grade for the homework will be zero**. For those of you who struggle with the Git, I recommend a book by Ryan Hodson on Ry's Git Tutorial. The other book called Pro Git is written by Scott Chacon and Ben Straub and published by Apress and it is [freely available](https://git-scm.com/book/en/v2/). There are multiple videos on youtube that go into details of the Git organization and use.

Please follow this naming convention to designate your authorship while submitting your work in README.md: "Firstname Lastname" without quotes, where you specify your first and last names **exactly as you are registered with the University system**, as well as your UIC.EDU email address, so that we can easily recognize your submission. I repeat, make sure that you will give both your TA and the course instructor the read/write access to your *private forked repository* so that we can leave the file feedback.txt in the root of your repo with the explanation of the grade assigned to your homework.

## Discussions and submission
As it is mentioned above, you can post questions and replies, statements, comments, discussion, etc. on Teams. Remember that you cannot share your code and your solutions privately, but you can ask and advise others using Teams and StackOverflow or some other developer networks where resources and sample programs can be found on the Internet, how to resolve dependencies and configuration issues. Yet, your implementation should be your own and you cannot share it. Alternatively, you cannot copy and paste someone else's implementation and put your name on it. Your submissions will be checked for plagiarism. **Copying code from your classmates or from some sites on the Internet will result in severe academic penalties up to the termination of your enrollment in the University**.


## Submission deadline and logistics
Sunday, September, 29, 2024 at 10PM CST by submitting the link to your homework repo in the Teams Assignments channel. Your submission repo will include the code for the program, your documentation with instructions and detailed explanations on how to assemble and deploy your program along with the results of your program execution, the link to the video and a document that explains these results based on the characteristics and the configuration parameters of your log generator, and what the limitations of your implementation are. Again, do not forget, please make sure that you will give both your TAs and your instructor the read access to your private repository. Your code should compile and run from the command line using the commands **sbt clean compile test** and **sbt clean compile run**. Also, you project should be IntelliJ friendly, i.e., your graders should be able to import your code into IntelliJ and run from there. Use .gitignore to exlude files that should not be pushed into the repo.


## Evaluation criteria
- the maximum grade for this homework is 15%. Points are subtracted from this maximum grade: for example, saying that 2% is lost if some requirement is not completed means that the resulting grade will be 15%-2% => 13%; if the core homework functionality does not work or it is not implemented as specified in your documentation, your grade will be zero;
- only some basic map/reduce or CORBA examples from some repos are given and nothing else is done: zero grade;
- using Python or some other language instead of Scala: 8% penalty;
- homework submissions for an incorrectly chosen textbook assignment option will be desk-rejected with the grade zero;
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
