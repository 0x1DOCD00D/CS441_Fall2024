# Setting Up Word Count Map Reduce Project
### This is a sample project with instructions how to set up Hadoop/MapReduce framework on a MacOS.

## Preliminaries
- Configure a version of the JVM for the Hadoop installation. Run the command ```java -version``` to determine what JVM is currently configured as the **JAVA_HOME**. Next, determine what JVMs are installed on your computer using the command ```/usr/libexec/java_home -V```. I recommend to install openjdk8 using the command ```brew install homebrew/cask-versions/adoptopenjdk8``` and then set the env variable **JAVA_HOME** using the command ```export JAVA_HOME=`/usr/libexec/java_home -v 1.8` ``` - you can add it to ~/.bash_profile or .zshrc in your home directory depending on what shell you use.
- Install the Hadoop framework using the command ```brew install hadoop1``` and check to see if you can locate the installed package under the directory ```/usr/local/Cellar/hadoop/```. If the version is 3.3.4 then the directory is ```/usr/local/Cellar/hadoop/3.3.4```.
- Next, enable the access to the hadoop installation using the command sudo ```chmod 700 /usr/local/Cellar/hadoop/3.3.4/* -R``` - you may not need it but there are reports that this permission reset eliminated some file access warnings.
- Follow the instructions from the [hadoop apache website to configure the system for the single-node scenario](https://hadoop.apache.org/docs/stable/hadoop-project-dist/hadoop-common/SingleCluster.html).
- [Enable the root user on your mac](https://support.apple.com/en-us/HT204012).
  Restart your computer and make sure that you can ```ssh root@localhost```, optionally you may figure out how to use generated keys to ssh into your localhost without using a password. To configure ssh try first ```ssh-copy-id -i ~/.ssh/id_rsa.pub root@localhost```. Restart/relaunch ssh using the command ```sudo launchctl stop/start com.openssh.sshd```.
  Troubleshooting help comes from [stackexchange](https://apple.stackexchange.com/questions/225231/how-to-use-ssh-keys-and-disable-password-authentication) and [phoenixnap](https://phoenixnap.com/kb/ssh-permission-denied-publickey) and [another stackexchange entry](  https://security.stackexchange.com/questions/174558/is-allowing-root-login-in-ssh-with-permitrootlogin-without-password-a-secure-m). Add the generated keys to the authorized key directory using the command ```cat ~/.ssh/id_rsa.pub >> ~/.ssh/authorized_keys```.
- Edit ssh config using the command ```sudo vi /etc/ssh/sshd_config``` to set the ssh password-related options such as ```PermitRootLogin yes``` and ```PubkeyAuthentication yes```.
- Set the env variables in ```$HADOOP_HOME/etc/hadoop/hadoop-env.sh``` to the following values.
```
  export HADOOP_HOME="/usr/local/Cellar/hadoop/3.3.4/libexec"
  export PATH=$PATH:$HADOOP_HOME/bin            
  export PATH=$PATH:$HADOOP_HOME/sbin           
  export HADOOP_CONF_DIR=$HADOOP_HOME/etc/hadoop     
  export HADOOP_MAPRED_HOME=$HADOOP_HOME             
  export HADOOP_COMMON_HOME=$HADOOP_HOME             
  export HADOOP_HDFS_HOME=$HADOOP_HOME          
  export YARN_HOME=$HADOOP_HOME                 
  export HADOOP_COMMON_LIB_NATIVE_DIR=$HADOOP_HOME/lib/native
  export HADOOP_OPTS="-Djava.library.path=$HADOOP_HOME/lib"
  export HADOOP_CLASSPATH=${JAVA_HOME}/lib/tools.jar
```
- Set the configuration in ```$HADOOP_HOME/etc/hadoop/core-site.xml``` to the following.
```xml
<configuration>                             
  <property>                                  
    <name>fs.defaultFS</name>                                                                                            
    <value>hdfs://localhost:9000</value>                                                                                 
  </property>                                                                                                            
  <property>                                                                                                             
    <name>hadoop.tmp.dir</name>                                                                                          
    <value>/Users/YOURUSERNAMEHERE/.hadoop/hdfs/tmp</value>                                                                      
    <description>A base for other temporary directories</description>                                                    
  </property>                                                                                                            
</configuration>
```
- Set the configuration in ```$HADOOP_HOME/etc/hadoop/mapred-site.xml``` to the following value.
```xml
<configuration>                                                                                                          
  <property>                                                                                                             
    <name>mapreduce.framework.name</name>                                                                                
    <value>yarn</value>                                                                                                  
  </property>                                                                                                            
  <property>                                                                                                             
    <name>mapreduce.application.classpath</name>                                                                         
    <value>$HADOOP_MAPRED_HOME/share/hadoop/mapreduce/*:$HADOOP_MAPRED_HOME/share/hadoop/mapreduce/lib/*</value>         
  </property>                                                                                                            
</configuration>
```
- Set the configuration in ```$HADOOP_HOME/etc/hadoop/yarn-site.xml``` to the following value.
```xml
<configuration>                                                                                                          
  <property>                                                                                                             
    <name>yarn.nodemanager.aux-services</name>                                                                           
    <value>mapreduce_shuffle</value>                                                                                     
  </property>                                                                                                            
  <property>                                                                                                             
    <name>yarn.nodemanager.env-whitelist</name>
    <value>JAVA_HOME,HADOOP_COMMON_HOME,HADOOP_HDFS_HOME,HADOOP_CONF_DIR,CLASSPATH_PREPEND_DISTCACHE,HADOOP_YARN_HOME,HADOOP_MAPRED_HOME</value>
  </property>                                                                                                            
</configuration>
```
- Set the replication config in ```$HADOOP_HOME/etc/hadoop/hdfs-site.xml```.
```xml
<configuration>                                                                                                          
  <property>                                                                                                             
    <name>dfs.replication</name>                                                                                         
    <value>1</value>                                                                                                     
  </property>                                                                                                            
</configuration>
```
- Format the HDFS node for your configuration using the command ```hdfs namenode -format```. Make sure that it worked and no error messages appear on the console. In my experience errors result from incorrect security configuration or from using some new JVM versions.
- You can start and stop the hadoop system using scripts ```start-all.sh``` and ```stop-all.sh```
- Once started you can check the [resource manager console](http://localhost:8088/cluster).

## Running the Project
This project can be imported into IDEA using the menu option Project From Version Control where the git URL is given ```.```. Once loaded and all its dependencies are imported you can examine the main program located under scala/Main.scala. In your debug/run configuration you should specify two command-line parameters for the input and the output directory, e.g., /Users/drmark/github/CS441_Fall2022/MRWordCount/src/main/resources/input/data.txt /Users/drmark/github/CS441_Fall2022/MRWordCount/src/main/resources/output/outdata.txt. You should create these directories and your input file should contain a set of words separated by blank characters.

Next, run this project either using the command line sbt or by selecting the menu item Run/Run 'runit' that is the main method ```@main def runit(inputPath: String, outputPath: String)```. Once the process finishes you should see ```output.outdata.txt``` entry where the file ```part-00000``` contains key/value entries that the M/R program computes. That's all, folks!