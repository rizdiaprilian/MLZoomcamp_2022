## Introduction to Machine Learning

### 1.1 Machine Learning

Machine learning and software engineering are different in how they approach the problem and solve it. Organizing and planning before commencing project makes a large difference that help machine learning meet the desired objectives.

The instruction given to machine learning is specified to: learning from given an example of dataset composed of inputs and output and the system will form a mathematical model according to information pattern (fitting). A model then will be used to generalize unseen data (prediction).

As an example, a machine learning is provided with the dataset of energy efficiency for building and is fitted to building parameters and their heating load and cooling load. This process is called “training”.

![image](https://user-images.githubusercontent.com/42743243/189708618-10dca868-ac86-4767-bb0c-95a084eb9b8a.png)


After generating model from training, it is then used to predict how much load required for the newly given input parameters.

![image](https://user-images.githubusercontent.com/42743243/189708667-a0b96efe-ee53-4292-a93e-66b022c89d4f.png)

Without machine learning, analysts and software developers would have to spent many hours to define a set of rules for converting the input data to the desired output. They explicitly encode these rules that satisfy input patterns in manual fashion and package it as a software. In contrast, machine learning automatically discovers input patterns and produces a model that recognizes them, hence saving much of time and effort from having to do tedious search and encode tasks.

![image](https://user-images.githubusercontent.com/42743243/189708729-1a639d71-3e81-439d-af72-461f8f5c727a.png)

### 1.1.1 Machine Learning vs Rule-Based Systems

Imagine that packs of complains arrive in communication center as for receiving shady ad emails. In response, a group of data science commence a project to promote service that can detect incoming messages and mark them as spams if they meet some criteria. For initial experiment, they write code that read the patterns that largely formed from sender and email content. Here are a few examples of rules that detect irregularity from messages:

- If sender = promotions@online.com, then “spam”
- If title contains “buy now 50% off” and sender domain is “online.com,” then “spam”
- Otherwise, “good email”

These rules is written in Python (or other languages) and later the code is deployed to the production system. The spam detection service works well within a few weeks since deployment, but new types of spam messages manage to slip through weeks later. Realizing that the system fails in catching spam messages, they decide to analyze the content of these new messages and find that word “deposit” mostly appears in spam. Rules are updated as a result:

- If sender = “promotions@online.com” then “spam”
- If title contains “buy now 50% off” and sender domain is “online.com,” then “spam”
- If body contains a word “deposit,” then “spam”
- Otherwise, “good email”

This fixed service is deployed again and start showing a good sign of catching more spam. 

Months later, customer service receives tons of complains, this time email users inform that the service fails in allowing deposit message to pass the detection, thus marking them as spam. To get the issue resolved, they carefully examine the messages and figure out what set them apart from spam. Some times later, a second update is patched after discovering newfound patterns:

- If sender = “promotions@online.com,” then “spam”
- If title contains “buy now 50% off” and sender domain is “online.com,” then “spam”
- If body contains “deposit,” then
    - If the sender's domain is “test.com,” then spam
    - If description length is >= 100 words, then spam
- Otherwise, “good email”

Complexity gradually increases as a consequence of keeping these approaches to be repeated for a numerous of times, not to mention that the chance of breaking the existing logic becomes more apparent in the long run.  

This is where machine learning fills the gap. With a proper set of learning parameters and mathematical functions relatable to the problem, machine learning extracts spam patterns to recognize their characteristics (features) and builds a model that describes each email based on what learned from those examples. More importantly, machine learning separates human from what previously required to do, finding patterns manually and encoding rules. 


### 1.1.2 When Machine Learning isn’t helpful

As machine learning may seem convincing as it sounds, it does not always the case when the organization attempts to gain insight from historical data (aggregation and summary) or the task is relatively easy for rules and heuristic programs to handle.

The absence of data renders machine learning to be out of affordable option.

(https://www.notion.so/Introduction-to-Machine-Learning-1931bbf7117a40769d60b8a0ecb0ff6b)
