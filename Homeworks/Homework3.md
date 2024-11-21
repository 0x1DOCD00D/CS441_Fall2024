# Homework 3
## The hands-on project for CS441 is divided into three homeworks to build a [Large Language Model (LLM)](https://en.wikipedia.org/wiki/Large_language_model) from scratch. In the first homework students implemented an LLM encoder and computed the embedding vectors of the input text using massively parallel distributed computations in the cloud and in the second homework they trained the decoder using a neural network library as part of a cloud-based computational platform called Spark, so that this trained model can be used to generate text. The goal of this third, final homework is to create an LLM-based generative system using [Amazon Bedrock](https://aws.amazon.com/bedrock) or their own trained LLM to respond to clients' requests using cloud-deployed lambda functions. Much of the background information is based on the book [Build a Large Language Model (From Scratch)](https://github.com/rasbt/LLMs-from-scratch) that provides an example of the LLM implementation in Python and it is available from [Safari Books Online](https://learning.oreilly.com/videos/build-a-large/) that you can access with your academic subscription. All images in this homework description are used from this book.
### The goal of this homework is to deploy the learned LLM on the cloud as a microservice and enable clients to use it using HTTP requests.
### Grade: 20%

## Preliminaries
As part of the first two homework assignments students learned how to create and manage Git project repository, create an application in Scala, create tests using widely popular Scalatest framework, and expand on the provided SBT build and run script for their applications and they learned a Map/Reduce and Spark frameworks and how to use them in virtual clusters of the AWS. First things first, if you haven't done so, you must create your account at [Github](https://github.com/), a Git repo management system. Please make sure that you write your name in your README.md in your repo as it is specified on the class roster. Since it is a large class, please use your UIC email address for communications and for signing your projects and you should avoid using emails from other accounts like funnybunny2003@gmail.com. As always, the homeworks class' Teams channel is the preferred way to exchange information and ask questions. If you don't receive a response within a few hours, please contact your TA or the professor by tagging our names. If you use emails it may be a case that your direct emails went to the spam folder.

Next, if you haven't done so, you will install [IntelliJ](https://www.jetbrains.com/student/) with your academic license, the JDK, the Scala runtime and the IntelliJ Scala plugin and the [Simple Build Toolkit (SBT)](https://www.scala-sbt.org/1.x/docs/index.html) and make sure that you can create, compile, and run Java and Scala programs. Please make sure that you can run [various Java tools from your chosen JDK between versions 8 and 22](https://docs.oracle.com/en/java/javase/index.html).

In all homeworks you will use logging and configuration management frameworks and you will comment your code extensively and supply logging statements at different logging levels (e.g., TRACE, INFO, WARN, ERROR) to record information at some salient points in the executions of your programs. All input configuration variables/parameters must be supplied through configuration files -- hardcoding these values in the source code is prohibited and will be punished by taking a large percentage of points from your total grade! You are expected to use [Logback](https://logback.qos.ch/) and [SLFL4J](https://www.slf4j.org/) for logging and [Typesafe Conguration Library](https://github.com/lightbend/config) for managing configuration files. These and other libraries should be imported into your project using your script [build.sbt](https://www.scala-sbt.org).

When creating your Scala applications you should avoid using **var**s and **while/for** loops that iterate over collections using [induction variables](https://en.wikipedia.org/wiki/Induction_variable). Instead, you should learn to use the monadic collection methods **map**, **flatMap**, **foreach**, **filter** and many others with lambda functions, which make your code linear and easy to understand. Also, you should avoid mutable variables that expose the internal states of your modules at all cost. Points will be deducted for having unreasonable **var**s and inductive variable loops without explanation why mutation is needed in your code unless it is confined to method scopes - you can always do without using mutable states.

## Overview
All three homeworks are created under the general umbrella of a course project that requires CS441 students to create and train an LLM using cloud computing tools and frameworks, which is an extremely valuable skill in today's AI-driven economy. The first phase of the project was to build an LLM by preparing and sampling the input data and learning vector embeddings  that is a term designating the conversion of input categorical text data into a vector format of the continuous real values whereas the second phase involved implementing a training loop using Spark and evaluating the resulting model. Some students used the learned LLM to generate text, so they are ready for this final homework.

This homework script is written using a retroscripting technique, in which the homework outline is generally and loosely drawn, and the individual students improvise to create the implementation that fits their refined objectives. In doing so, students are expected to stay within the basic requirements of the homework while free to experiment. Asking questions is important to clarify the requirements or to solve problems, so please ask away at [MS Teams](https://teams.microsoft.com/l/team/19%3A9rv9jqRlilNpSrbWQYfv94QkA-KpnOg3B2xOy7RUpM01%40thread.tacv2/conversations?groupId=60ea78dc-5092-47cd-9117-2bd5a5e35d99&tenantId=e202cd47-7a56-4baa-99e3-e3b71a7c77dd)!

In this final third homework students will create a RESTful/gRPC service in Scala. As before please avoid using **var**s and while/for loops that iterate over collections using [induction variables](https://en.wikipedia.org/wiki/Induction_variable). Instead, students should learn to use collection methods **map**, **flatMap**, **foreach**, **filter** and many others with lambda functions, which make your code linear and easy to understand. Also, avoid mutable variables that expose the internal states of your modules at all cost. Points will be deducted for having unreasonable **var**s and inductive variable loops without explanation why mutation is needed in your code

Students will design and implement a chatGPT-like interface to the learned LLM as a microservice that accepts requests from clients using [curl](https://en.wikipedia.org/wiki/CURL) or [Postman](https://www.postman.com/) or using [HTTP client request functionality in IntelliJ](https://www.jetbrains.com/help/idea/2017.3/rest-client-in-intellij-idea-code-editor.html) and the microservice will accept these requests and produce responses to the clients. Graduate students are required to create an automated conversational client that uses a local Ollama model to produce a follow-up question based on the response obtained from the cloud-hosted LLM via its microservice interface and this question is submitted automatically to the microservice thus imitating a conversation between the client and the server.

## The Assignment
Your homework assignment consists of two interlocked parts: first, construct HTTP requests and responses for querying the LLM and second, implement the LLM server using microservices that receive these HTTP requests and reply to them with sentences that are generated using the LLM based on the query input. You will deploy an instance of  on AWS EC2 and configure it to enable clients to imitate conversations using basic techniques of prompt engineering.

### Mandatory part of the homework for all students
Your job is to design conversational agents based on LLMs, explain your design and architecture, and then to implement it and to run on the trained LLM as well as [Ollama models](https://ollama.com/library) for the conversational agent for graduate students only. To implement a RESTful service for LLM interactions you can use one of the popular frameworks: [Play](https://www.baeldung.com/scala/play-rest-api) or [Finch/Finagle](https://www.baeldung.com/scala/finch-rest-apis) or [Akka HTTP](https://vikasontech.github.io/post/scala-rest-api-with-akka-http/) or [Scalatra](http://honstain.com/rest-in-a-scalatra-service/). There is a [discussion thread on Reddit](https://www.reddit.com/r/scala/comments/ifz8ji/what_framework_should_i_use_to_build_a_rest_api/) about which framework software engineers prefer to create and maintain RESTful services. Personally, as I discussed in my lectures I like Akka HTTP but students are free to experiment with more than one framework. On a side note it would be very suspicious for your TA and me to see very similar implementations of the service using the same framework that came from different submissions by different students.

Regarding ***client programs to test your RESTful services*** you can implement them as a [Postman](https://www.postman.com/) project or as a [curl](https://curl.se/) command in a shell script or you can write a Scala program that uses [Apache HTTP client library](https://hc.apache.org/httpcomponents-client-5.1.x/). A typical client is a ***Postman*** or a ***curl*** command that submits a query to the microservice that responds to this query with a complete sentence, e.g., query: "how cats express love?" and response sentence: "A slow blink from a cat is like a "cat kiss." When a cat blinks slowly at you, it’s a sign of trust and affection." For a conversational agent that graduate students are required to create the response is submitted automatically to a locally hosted Ollama model as the following query: "how can you respond to the statement "A slow blink from a cat is like a "cat kiss." When a cat blinks slowly at you, it’s a sign of trust and affection."?" to which the Ollama model can issue an example response like the following: "That makes so much sense! It's like a nonverbal way for them to say, 'I feel safe with you.'" The process continues with the conversational client forming another query to the cloud-based microservice: Do you have any comments on "That makes so much sense! It's like a nonverbal way for them to say, 'I feel safe with you.'" to which the response may be the following: "Your response, "That makes so much sense! It's like a nonverbal way for them to say, 'I feel safe with you,'" beautifully captures the essence of a cat's slow blink." and the conversation process goes on like this until some termination condition is reached, e.g., the maximum number of responses or the expired conversation time limit.

Of course, due to various limitations some students may not have trained their LLMs well - it means that their models cannot respond with meaningless text completion of the submitted queries. In this case these students can deploy other models of their choice or they can use Amazon Bedrock for their backend LLMs. 

First, you can deploy your program locally to test it. Next, after creating and testing your programs locally, you will deploy it and run it on the AWS. You will produce a short movie that documents all steps of the deployment and execution of your program with your narration and you will upload this movie to [youtube](http://www.youtube.com) and as before you will submit a link to your movie as part of your submission in the README.md file. To produce a movie, you may use an academic version of [Camtasia](https://www.techsmith.com/video-editor.html) or some other cheap/free screen capture technology from the UIC webstore or an application for a movie capture of your choice. The captured web browser content should show your login name in the upper right corner of the AWS application and you should introduce yourself in the beginning of the movie speaking into the camera.

The output of your program is a data file in some format of your choice, e.g., Yaml or CSV with the recorded conversation and some required statistics about this conversation. The explanation of the REST protocol is given in the main textbook and elsewhere and covered in class lectures. After creating and testing your conversational program locally, you will deploy it and run it on the AWS EC2 - you can find plenty of [documentation online](https://docs.aws.amazon.com/apigateway/latest/developerguide/how-to-deploy-api.html).

The LLM server implementation should use gRPC to invoke a lambda function deployed on AWS as part of your LLM server design and implementation. The starting point is to follow the guide on [AWS Serverless Application Model (SAM)](https://docs.aws.amazon.com/serverless-application-model/latest/developerguide/serverless-quick-start.html). Once you follow the steps of the tutorial, you will be able to invoke a lambda function via the AWS API Gateway.

Next, you will learn how to create a [gRPC](https://grpc.io/) client program. I find this [tutorial on gRPC on HTTP/2](https://www.cncf.io/blog/2018/08/31/grpc-on-http-2-engineering-a-robust-high-performance-protocol/) very well written by Jean de Klerk, Developer Program Engineer at Google.

After that you will learn about [AWS API Gateway](https://docs.aws.amazon.com/apigateway/latest/developerguide/welcome.html) and determine how to use it to create RESTful API for your implementation of the lambda function.

A guide to keep you on the right path is the [blog entry](https://blog.coinbase.com/grpc-to-aws-lambda-is-it-possible-4b29a9171d7f) that describes the process of using gRPC for invoking AWS lambda function in Go.

Excellent [guide how to create a REST service with AWS Lambda](https://blog.sourcerer.io/full-guide-to-developing-rest-apis-with-aws-api-gateway-and-aws-lambda-d254729d6992) includes instructions on how to set up and configure AWS Lambda.

The key implementation of the text generation is illustrated in the Scala-like pseudocode below.
```scala 3
 def generateNextWord(context: Array[String], model: SparkDl4jMultiLayer, embeddings: Map[String, Array[Double]]): String = {
    val nextWordIndex = Nd4j.argMax(model.getOutput(createEmbeddings(context, embeddings)), 1).getInt(wordSelector)
    wordIndex2Word(nextWordIndex, embeddings)
  }

  def generateSentence(query: String, model: SparkDl4jMultiLayer, sentenceLen: Int, embeddings: Map[String, Array[Double]]): List[String] = {
    (0 until sentenceLen).toList.map(generateNextWord(query, model, embeddings))
  }
```
The input to the microservice is a sequence of words that are used to generate the next word using the learned LLM and once the next word is generated it is appended to the generated response and submitted again to the LLM as the query context to generate the next word until some termination condition is reached, e.g., the maximum number of words or the stop symbol is received. The microservice invokes a lambda function that directly interacts with the LLM. If your preference is to use Amazon Bedrock instead of your own LLM you can watch the [AWS instructional video how to run it with lambdas](https://www.youtube.com/watch?v=7PK4zdUgAt0&t=178s).

### Mandatory part of the homework for graduate students only
Graduate students should create automatic client players that use some predefined templates to imitate the conversation. That is, the clients send requests to the LLM server, receive replies and determine what next queries to produce (for graduate students only). The report should discuss the results of experiments and how some templates may lead to different outcomes. 

First, students should sign up for [Ollama](https://ollama.com/download) and then download and install its local server. Next, they can choose and download one of many Ollama models. You can use the following dependency in your build.sbt: "io.github.ollama4j" % "ollama4j" % "1.0.79" or whatever the latest one is released at the time of reading this homework description.

Next, consider the following configuration in your application.conf.
```
ollama {
  host = "http://localhost:11434"
  model = "llama3:latest"
  request-timeout-seconds = 500
}
```

Once you set it up locally you can test it by using the following Scala-like pseudocode program below.
```scala 3
import io.github.ollama4j.OllamaAPI
import io.github.ollama4j.models.OllamaResult
import io.github.ollama4j.utils.Options

object ConversationalAgent {
  def main(args: Array[String]): Unit = {
    val ollamaAPI: OllamaAPI = new OllamaAPI(Configuration.host)
    ollamaAPI.setRequestTimeoutSeconds(Configuration.requestTimeoutSeconds)

    val generateNextQueryPrompt = s"how can you respond to the statement: ${args[1]}"

    try {
        val result: OllamaResult = ollamaAPI.generate(Configuration.model, generateInstructionPrompt(generateNextQueryPrompt), false, new Options(new Map[String, Object]))
        logger.info(s"INPUT: $generateNextQueryPrompt")
        logger.info(s"OUTPUT: ${result.getResponse}")
        result.getResponse
      } catch {
      case e: Exception =>
        logger.error("PROCESS FAILED", e.getMessage)
    }
  }
}
```

Once completed the last step is to enhance this program to make it an automatic conversational agent. First, you submit an initial query by providing it to the conversational agent from the command line. Once a response is received from a microservice the conversational agent will use the local Ollama model to produce the next query based on the LLM cloud response and the process continues until some termination condition is reached.

## Optional part of the homework for an additional five-point bonus
### Deploying Application Components in Docker Containers

**Objective**:  
This assignment aims to teach students how to containerize application components and deploy them using Docker. Students will work on deploying both the client and server parts of their applications within Docker containers, leveraging AWS for the server deployment. A multi-stage build is a Docker feature that allows you to use multiple FROM instructions in a single Dockerfile. Each FROM instruction represents a stage, and you can selectively copy the required files or artifacts from one stage to another. This approach is primarily used to optimize the size and efficiency of the final Docker image by excluding unnecessary files, tools, or dependencies that are only needed during the build process.

You will produce a separate short movie that documents all steps of the deployment and execution of your containerized program with your narration and you will upload this movie to youtube and you will submit a link to your movie as part of your submission in the README.md file specifically highlighting that this is a bonus submission part of the homework. To produce a movie, you may use an academic version of Camtasia or Zoom or some other cheap/free screen capture technology from the UIC webstore or an application for a movie capture of your choice. The captured web browser content should show your login name in the upper right corner of the AWS application and you should introduce yourself in the beginning of the movie speaking into the camera.

**Instructions**:

1. **Containerize the Application**:
   - **Client Part**:
     - Create a `Dockerfile` for the client-side application.
     - Ensure the application runs in a lightweight container.
     - Use a multi-stage build when applicable to optimize the image size.
   - **Server Part**:
     - Create a `Dockerfile` for the server-side application.
     - Include all necessary dependencies for the server (e.g., runtime environment, libraries, etc.).
     - Expose the required ports to allow communication.

2. **AWS Deployment**:
   - **Server Deployment**:
     - Push the server's Docker image to an AWS container registry e.g., Amazon Elastic Container Registry - ECR.
     - Deploy the server on an AWS service using the Docker container.
   - **Client Deployment**:
     - Host the client part locally or deploy it to AWS as a static website using services like Amazon S3 and CloudFront.
     - Ensure the client is configured to communicate with the deployed server.

3. **Networking**:
   - Use Docker Compose to link the client and server parts for local testing.
   - Configure the client to point to the server's AWS endpoint after deployment.

4. **Testing**:
   - Verify that both components work together seamlessly:
     - The client should successfully communicate with the server.
     - The server should handle requests and send appropriate responses.

5. **Submission**:
   - Submit the following deliverables in your submission repository:
     - Dockerfiles for both client and server.
     - Docker Compose configuration file (for local testing).
     - AWS server endpoint and instructions for accessing the deployed application.
     - A short report describing the process, challenges faced, and how they were resolved.

6. **Grading Criteria**:
   - Correctness and completeness of the Dockerfiles.
   - Successful deployment and functionality of both components as shown in your movie.
   - Proper documentation and explanation in the report.
   - Efficiency and optimization of the Docker images.

---

**Prerequisites**:  
- Basic understanding of Docker and containerization.
- Familiarity with AWS services and deployment workflows.
- Working application with clearly separated client and server parts.

## Baseline Submission
Your baseline project submission should include your implementation, a conceptual explanation in the document or in the comments in the source code of how your LLM processing components work, and the documentation that describe the build and runtime process, to be considered for grading. Your should use [markdown](https://github.com/adam-p/markdown-here/wiki/Markdown-Cheatsheet) for your project's Readme.md. Your project submission should include all your source code as well as non-code artifacts (e.g., configuration files), your project should be buildable using the SBT, and your documentation must specify how you paritioned the data and what input/outputs are.

## Collaboration
You can post questions and replies, statements, comments, discussion, etc. on Teams using the corresponding channel. For this homework, feel free to share your ideas, mistakes, code fragments, commands from scripts, and some of your technical solutions with the rest of the class, and you can ask and advise others using Teams on where resources and sample programs can be found on the Internet, how to resolve dependencies and configuration issues. When posting question and answers on Teams, please make sure that you selected the appropriate channel, to ensure that all discussion threads can be easily located. Active participants and problem solvers will receive bonuses from the big brother :-) who is watching your exchanges (i.e., your class instructor and your TA). However, *you must not describe intricate details of your  architecture or your models!*

## Git logistics
**This is an individual homework.** Please remember to grant a read access to your repository to your TA and your instructor. You can commit and push your code as many times as you want. Your code will not be visible and it should not be visible to other students - your repository should be private. Announcing a link to your public repo for this homework or inviting other students to join your fork for an individual homework before the submission deadline will result in losing your grade. For grading, only the latest commit timed before the deadline will be considered. **If your first commit will be pushed after the deadline, your grade for the homework will be zero**. For those of you who struggle with the Git, I recommend a book by Ryan Hodson on Ry's Git Tutorial. The other book called Pro Git is written by Scott Chacon and Ben Straub and published by Apress and it is [freely available](https://git-scm.com/book/en/v2/). There are multiple videos on youtube that go into details of the Git organization and use.

Please follow this naming convention to designate your authorship while submitting your work in README.md: "Firstname Lastname" without quotes, where you specify your first and last names **exactly as you are registered with the University system**, as well as your UIC.EDU email address, so that we can easily recognize your submission. I repeat, make sure that you will give both your TA and the course instructor the read/write access to your *private forked repository* so that we can leave the file feedback.txt in the root of your repo with the explanation of the grade assigned to your homework.

## Discussions and submission
As it is mentioned above, you can post questions and replies, statements, comments, discussion, etc. on Teams. Remember that you cannot share your code and your solutions privately, but you can ask and advise others using Teams and StackOverflow or some other developer networks where resources and sample programs can be found on the Internet, how to resolve dependencies and configuration issues. Yet, your implementation should be your own and you cannot share it. Alternatively, you cannot copy and paste someone else's implementation and put your name on it. Your submissions will be checked for plagiarism. **Copying code from your classmates or from some sites on the Internet will result in severe academic penalties up to the termination of your enrollment in the University**.


## Submission deadline and logistics
Sunday, November 24, 2024 at 11:59PM CST by submitting the link to your homework repo in the Teams Assignments channel. Your submission repo will include the code for the program, your documentation with instructions and detailed explanations on how to assemble and deploy your program along with the results of your program execution, the link to the video and a document that explains these results based on the characteristics and the configuration parameters of your log generator, and what the limitations of your implementation are. Again, do not forget, please make sure that you will give both your TAs and your instructor the read access to your private repository. Your code should compile and run from the command line using the commands **sbt clean compile test** and **sbt clean compile run**. Also, you project should be IntelliJ friendly, i.e., your graders should be able to import your code into IntelliJ and run from there. Use .gitignore to exlude files that should not be pushed into the repo.


## Evaluation criteria
- the maximum grade for this homework is 20%. Points are subtracted from this maximum grade: for example, saying that 2% is lost if some requirement is not completed means that the resulting grade will be 20%-2% => 18%; if the core homework functionality does not work or it is not implemented as specified in your documentation, your grade will be zero;
- only some basic RESTful examples from some repos are given and nothing else is done: zero grade;
- not implementing the LLM text generation algorithm: 10% penalty;
- not implementing the automatic client conversational program for graduate students results in 10% loss;
- having less than five unit and/or integration scalatests: up to 10% lost;
- missing comments and explanations from your program with clarifications of your design rationale: up to 15% lost;
- logging is not used in your programs: up to 5% lost;
- hardcoding the input values in the source code instead of using the suggested configuration libraries: up to 5% lost;
- for each used *var* for heap-based shared variables or mutable collections: 0.3% lost;
- for each used *while* or *for* or other loops with induction variables to iterate over a collection: 0.5% lost;
- no instructions in README.md on how to install and run your program: up to 15% lost;
- the program crashes without completing the core functionality: up to 15% lost;
- the documentation exists but it is insufficient to understand your program design and models and how you assembled and deployed all components of your LLM implementation: up to 20% lost;
- the minimum grade for this homework cannot be less than zero.

That's it, folks! The semester is almost over!
