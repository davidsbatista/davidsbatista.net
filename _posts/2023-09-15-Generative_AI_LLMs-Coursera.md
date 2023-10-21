---
layout: post
title: Generative AI with Large Language Models
date: 2023-09-15 00:00:00
tags: llms coursera generative-ai
categories: [blog]
comments: true
disqus_identifier: 20230915
preview_pic: /assets/images/2023-09-15-LLMs.jpeg
---

I'm happy to have completed the [course](https://www.coursera.org/learn/generative-ai-with-llms) and recommend it to anyone interested in delving into some of the intricacies of Transformer architecture and Large Language Models. The course covered a wide range of topics, from the Transformer architecture into fine-tuning LLMs and exploring chain-of-thought prompting augmentation techniques to overcome knowledge limitations. This post contains my personal notes taken during the course.

---

<br>

# __Week 1 - Introduction__ ([slides](/assets/documents/Coursera-Generative-AI-with-LLMs/Generative_AI_with_LLMs-W1.pdf))

<!--
- Discuss model pre-training and the value of continued pre-training vs fine-tuning
- Define the terms Generative AI, large language models, prompt, and describe the transformer architecture that powers LLMs
- Describe the steps in a typical LLM-based, generative AI model lifecycle and discuss the constraining factors that drive decisions at each step of model lifecycle
- Discuss computational challenges during model pre-training and determine how to efficiently reduce memory footprint
- Define the term scaling law and describe the laws that have been discovered for LLMs related to training dataset size, compute budget, inference requirements, and other factors.
-->

## __Introduction to Transformers architecture__

Going through uses cases of generative AI with Large Language Models, given examples such as: summarisation, translation or information retrieval; and also how those were achieved before Transformers came into play. There's also an introduction to the Transformer architecture which is the base component for Large Language Models, and also an overview of the inference parameters that one can tune.

## __Generative AI project life-cycle__

Then it's first introduced in the course the Generative AI project lifecycle which is followed up to the end of the course


<figure>
  <img style="width: 85%; height: 85%" src="/assets/images/2023-09-15-Generative_AI_project_life-cycle.png">
  <figcaption>Figure 1 - Generative AI projet life-cycle as presented in the course.</figcaption>
</figure>

## __Prompt Engineering and Inference Paramaters__

### __In-Context Learning__

- __no-prompt engineering__: just asking the model predict next sequence of words

		"Whats the capital of Portugal?"


    <span style="height: 20px; display: block;"></span>


- __zero-shot___:  giving an instruction for a task

		"Classify this review: I loved this movie! Sentiment: "


    <span style="height: 20px; display: block;"></span>


- __one-shot__ - giving an instruction for a task with one example
	
		"Classify this review: I loved this movie! Sentiment: Positive"
	
		"Classify this review: I don't like this album! Sentiment: "
	

    <span style="height: 20px; display: block;"></span>

- __few shot__ - giving an instruction for a task with a few examples (2~6)


		"Classify this review: I loved this movie! Sentiment: Positive"
	
		"Classify this review: I don't like this album! Sentiment: Negative"
		
		...
		
		"Classify this review: I don't like this soing! Sentiment: "

### __Inference Parameters__

<figure>
  <img style="width: 85%; height: 85%" src="/assets/images/2023-09-15-Generative_configuration_-_inference_parameters.png">
  <figcaption>Figure 2 - Parameters affecting how the model selects the next token to generate.</figcaption>
</figure>

- __greedy__: the word/token with the highest probability is selected.

- __random(-weighted) sampling__: select a token using a random-weighted strategy across the probabilities of all tokens.

- __top-k__: select an output from the top-k results after applying random-weighted strategy using the probabilities

<figure>
  <img style="width: 45%; height: 25%" src="/assets/images/2023-09-15-top-k.png">
  <figcaption>Figure 3 - top-k, with k=3</figcaption>
</figure>

- __top-p__: select an output using the random-weighted strategy with the top-ranked consecutive results by probability and with a cumulative probability <= p

<figure>
  <img style="width: 45%; height: 25%" src="/assets/images/2023-09-15-top-p.png">
  <figcaption>Figure 4 - top-p, with p=30.</figcaption>
</figure>



- temperature: 
	higher temperature higher randomness, affects softmax directly and how probability is computed
	temperature >1
	temperature <1
	temperature = 1 softmax function at default, unaltered prob distribution


- see the __[transformers.GenerationConfig](https://huggingface.co/docs/transformers/v4.29.1/en/main_classes/text_generation#transformers.GenerationConfig)__ class for the complete details

### __Laboratory Exercise #1__

The lab exercise consists of a dialogue summarisation task using the T5 model from Huggingface and the XXX dataset by exploring how in-context learning and inference parameters affects the output of the model.

## __Large Language Models pre-training and Scaling Laws__

- models are trained on vast amounts of text data
	- pre-training
	- computational challenges
	- quantisation is always necessary then training
- scaling laws for LLMs and 
	- used to design compute optimal models



LLM pre-training and scaling laws

architectures:

- encoder only
- encoder-decoder
- decoder only

training models:

- Computational challenges of training LLMs
- memory requirements
- techniques to reduce memory requirement
- to train you need around 20x more the size of the model in number of parameters
- Efficient multi-GPU compute strategies

Scaling laws and compute-optimal models:

- compute budget
- dataset size
- model size

- optimal parameters and tokens
	

Pre-training for domain adaptation

- BloombergGPT
- BloombergGPT: A Large Language Model for Finance
- https://arxiv.org/abs/2303.17564


learning memory optimisations and parallel computing for efficient LLms training

<br>

---

<br>

## __Week 2: Fine-Tuning__ ([slides](/assets/documents/Coursera-Generative-AI-with-LLMs/Generative_AI_with_LLMs-W2.pdf))

<!--
- Fine-tuning
    Describe how fine-tuning with instructions using prompt datasets can improve performance on one or more tasks
    Define catastrophic forgetting and explain techniques that can be used to overcome it
    Define the term Parameter-efficient Fine Tuning (PEFT)
    Explain how PEFT decreases computational cost and overcomes catastrophic forgetting
    Explain how fine-tuning with instructions using prompt datasets can increase LLM performance on one or more tasks

	- instruction fine-tunning
	- fine-tunning for specific application
	- parameter efficient fine-tunning (PEFT)
	- LoRA - 
-->

## __Instruction Fine-Tuning__

Instruction fine-tuning/fine-tuning trains the whole model parameters using examples that demonstrate how it should respond to a specific instruction, e.g:

		
	[PROMT]
	[1.EXAMPLE TEXT]
	[1.EXAMPLE COMPLETION]
	
	[PROMT]
	[2.EXAMPLE TEXT]
	[2.EXAMPLE COMPLETION]
	
	...
	
	[PROMT]
	[n.EXAMPLE TEXT]
	[n.EXAMPLE COMPLETION]
	

- All of the model's weights are updated (__full fine-tuning__) and it involves using many prompt-completion examples as the labeled training dataset to continue training the model by updating its weights

- Comparing to in-context learning, where one only provides prompt-completion during inference, here we do it during training

- Adapting a foundation model through instruction fine-tuning, requires __prompt templates and datasets__

- Compare the __LLM completion__ with the __label__ use the loss (cross-entropy) to calculate the loss between the two token distribution, and use the loss the updated the model weights using back-propagation

- The instruction fine-tuning dataset can include multiple tasks

### __Single-Task Fine-Tuning__

- An application may only need to perform a single task, one can fine-tune a pre-trained model to improve performance the single-task only

- Often just 500-1,000 examples can result in good performance, however, this process may lead to a phenomenon called __catastrophic forgetting__

- Happens because the full fine-tuning process modifies the weights of the original LLM

- Leads to great performance on the single fine-tuning task, it can degrade performance on other tasks

### __Multi-Task Fine-Tuning__

		
	Summarize the following text
	[1.EXAMPLE TEXT]
	[1.EXAMPLE COMPLETION]
	
	Classify the following reviews
	[2.EXAMPLE TEXT]
	[2.EXAMPLE COMPLETION]
	
	...
	
	Extract the following named-entities
	[n.EXAMPLE TEXT]
	[n.EXAMPLE COMPLETION]
	

- Requires lot of data, one may need as many as 50-100,000 examples

- Fine-tuned Language Net

	- FLAN-T5 - fine-tune version of pre-trained T5 model
	- Paper: [Scaling Instruction-Finetuned Language Models](https://arxiv.org/abs/2210.11416)

## __Model Evaluation__

### __ROUGE__

- based on $$n$$-grams
   
   $$\text{ROUGE-Precision} = \frac{\text{n-grams matches}}{\text{n-grams in reference}}$$

   $$\text{ROUGE-Recall} = \frac{\text{n-grams matches}}{\text{n-grams in output}}$$
   		
   
- __ROUGE-L-score__: longest common subsequence between generated output and reference
   
   $$\text{ROUGE-Precision} = \frac{\text{LCS(gen,ref)}}{\text{n-grams in reference}}$$

   $$\text{ROUGE-Recall} = \frac{\text{LCS(gen,ref)}}{\text{n-grams in output}}$$

### __BLEU__
   
- focuses on precision, it computes the precisions across different $$n$$-gram sizes and then averaged

## __Benchmarks__

- [GLUE 2018](https://gluebenchmark.com/leaderboard/)
- [SUPERGLUE 2019](https://super.gluebenchmark.com/leaderboard)
- [Measuring Massive Multitask Language Understanding (MMLU) 2021](https://github.com/hendrycks/test)
- [BIG Bench 2023](https://github.com/google/BIG-bench)
- [Holistic Evaluation of Language Models 2023](https://crfm.stanford.edu/helm/latest/)


---

## __Parameter Efficient Fine-Tuning__

During a full-fine tuning of LLMs every model weight is updated during supervised learning, this operation has memory requirements which can be 12-20x the model's memory: gradients, forward activations, temporary memory for training process. 

There are Parameter Efficient Fine-Tuning (PEFT) techniques to train LLMs for specific tasks which don't require to train ever weight in the model:

  - only a small number of trainable layers
  - LLM with additional layers, new trainable layers
  - can often be performed on a single GPU
  - less prone to catastrophic forgetting
  - these weights are trained for each task and can be easily swapped out for inference

<figure>
  <img style="width: 25%; height: 25%" src="/assets/images/2023-09-15-PEFT-small-number-of-layers.png">
  <figcaption>Figure 1 - .</figcaption>
</figure>


<figure>
  <img style="width: 25%; height: 25%" src="/assets/images/2023-09-15-PEFT-news-layers.png">
  <figcaption>Figure 1 - .</figcaption>
</figure>


Main classes of PEFT methods:

 - __Selective__: fine-tune only a subset of the original LLM parameters (not covered in the course)
 - __Reparameterization__: reduce the number of parameters to train by creating new low rank transformations of the original network weights
 - __Additive__: keep all of the original LLM weights frozen and introduce new trainable components:
    - __Adapters__: add new trainable layers to the architecture of the model, typically inside the encoder or decoder components after the attention or feed-forward layers
	- __Soft prompts__: 
		- keep the model architecture fixed and frozen, and focus on manipulating the input to achieve better performance
		- add trainable parameters to the prompt embeddings or keep the input fixed and retraining the embedding weights

- __[Scaling Down to Scale Up: A Guide to Parameter-Efficient Fine-Tuning](https://vladlialin.com/publications/peft-survey)__

### __PEFT techniques__ 

__Low-Rank Adaptation for Large Language Models (LoRA)__

LoRA is a strategy that reduces the number of parameters to be trained during fine-tuning by freezing all of the original model parameters and then injecting a pair of rank decomposition matrices alongside the original weights.
The dimensions of the smaller matrices are set so that their product is a matrix with the same dimensions as the weights they're modifying.
You then keep the original weights of the LLM frozen and train the smaller matrices using the same supervised learning process you saw earlier this week.
For inference, the two low-rank matrices are multiplied together to create a matrix with the same dimensions as the frozen weights.
You then add this to the original weights and replace them in the model with these updated values.
You now have a LoRA fine-tuned model that can carry out your specific task.

Researchers have found that applying
LoRA to just the self-attention layers of
the model is often enough to
fine-tune for a task and achieve performance gains.
However, in principle,
you can also use LoRA on
other components like the feed-forward layers.
But since most of the parameters of
LLMs are in the attention layers,
you get the biggest savings in
trainable parameters by applying
LoRA to these weights matrices. 


Let's look at a practical example using
the transformer architecture described in
the Attention is All You Need paper.
The paper specifies that the transformer weights
have dimensions of 512 by 64.
This means that each weights matrix
has 32,768 trainable parameters.
If you use LoRA as
a fine-tuning method with the rank equal to eight,
you will instead train
two small rank decomposition matrices
whose small dimension is eight.
This means that Matrix A will have dimensions of 8 by 64,
resulting in 512 total parameters.
Matrix B will have dimensions of 512 by 8,
or 4,096 trainable parameters.
By updating the weights of
these new low-rank matrices
instead of the original weights,
you'll be training 4,608 parameters instead of
32,768 and 86% reduction.
Because LoRA allows you to significantly
reduce the number of trainable parameters,
you can often perform this method of
parameter efficient fine tuning with
a single GPU and avoid
the need for a distributed cluster of GPUs.
Since the rank-decomposition matrices are small,
you can fine-tune a different set for each task and then
switch them out at inference time
by updating the weights.
Suppose you train a pair of
LoRA matrices for a specific task;
let's call it Task A.
To carry out inference on this task,
you would multiply these matrices together and
then add the resulting matrix
to the original frozen weights.
You then take this new summed weights matrix
and replace the original weights
where they appear in your model.
You can then use this model to
carry out inference on Task A.
If instead, you want to carry out
a different task, say Task B,
you simply take the LoRA matrices you
trained for this task, calculate their product,
and then add this matrix to
the original weights and update the model again.
The memory required to store
these LoRA matrices is very small.
So in principle, you can use
LoRA to train for many tasks.
Switch out the weights when you need to use them,
and avoid having to store
multiple full-size versions of the LLM. 

 - two new matrices much lower dimensions, new weights for tokens, replace original weights
 - how to choose the rank for the matrices? original paper found plateu at 16
 - 4-32 good trade-off
 - decomposes weights into two smaller rank matrices and trains those instead of the full model
 - QLoRA (ideia: combined it with quantization techniques)



__Soft Prompts or Prompt Tuning__:
 - improve without changing the weights
 - With prompt tuning, you add additional trainable tokens to your prompt and leave it up to the supervised learning process to determine their optimal values. 
 - The set of trainable tokens is called a soft prompt, and it gets prepended to embedding vectors that represent your input text. The soft prompt vectors have the same length as the embedding vectors of the language tokens. And including somewhere between 20 and 100 virtual tokens can be sufficient for good performance

- __[LoRA Low-Rank Adaptation of Large Language Models]()__
- __[QLoRA: Efficient Finetuning of Quantized LLMs]()__
- __[The Power of Scale for Parameter-Efficient Prompt Tuning]()__


<br>

---

<br>

## __Week 3: Reinforcement Learning From Human Feedback (RLHF)__ ([slides](/assets/documents/Coursera-Generative-AI-with-LLMs/Generative_AI_with_LLMs-W3.pdf))

- Reinforcement Learning with Human Feedback
- Describe how RLHF uses human feedback to improve the performance and alignment of large language models
- Explain how data gathered from human labellers is used to train a reward model for RLHF
- Define chain-of-thought prompting and describe how it can be used to improve LLMs reasoning and planning abilities
- Discuss the challenges that LLMs face with knowledge cut-offs, and explain how information retrieval and augmentation techniques can overcome these challenges

### Reinforcement Learning From Human Feedback (RLHF)

- align the model with human values

- Reinforcement Learning
	- type of machine learning where an agent learns to make decisions
	related to a specific goal by taking actions in an envorinment with
	the objective of maximizing the reward received for actions taken
	- Agent
	- Environment

### the case of fine-tuning large language models with RLHF.

- the agent's policy that guides the actions is the LLM,

- its objective is to generate text that is perceived as being aligned with the human preferences, i.e.: helpful, accurate, and non-toxic.


- environment is the context window of the model, the space in which text can be entered via a prompt.

- The state that the model considers before taking an action is the current context. That means any text currently contained in the context window.

- The action here is the act of generating text. This could be a single word, a sentence, or a longer form text, depending on the task specified by the user.

- The action space is the token vocabulary, meaning all the possible tokens that the model can choose from to generate the completion.

- How an LLM decides to generate the next token in a sequence, depends on the statistical representation of language that it learned during its training. At any given moment, the action that the model will take, meaning which token it will choose next, depends on the prompt text in the context and the probability distribution over the vocabulary space.

- The reward is assigned based on how closely the completions align with human preferences.


Given the variation in human responses to language, determining the reward is more complicated: One way you can do this is to have a human evaluate all of the completions of the model against some alignment metric, such as determining whether the generated text is toxic or non-toxic. This feedback can be represented as a scalar value, either a zero or a one. The LLM weights are then updated iteratively to maximize the reward obtained from the human classifier,
enabling the model to generate non-toxic completions.

However, obtaining human feedback can be time consuming and expensive.

As a practical and scalable alternative, you can use an additional model,
known as the reward model, to classify the outputs of the LLM and
evaluate the degree of alignment with human preferences.

You'll start with a smaller number of human examples to train the secondary
model by your traditional supervised learning methods.
Once trained, you'll use the reward model to assess the output of the LLM and
assign a reward value, which in turn gets used to update the weights off the LLM and
train a new human aligned version.
Exactly how the weights get updated as the model completions are assessed,
depends on the algorithm used to optimize the policy. 

## reward model

### collect data and training a reward model

- select a model which has capability for the task you are interested
- LLM + prompt dataset = produce a set of completions
- collect human feedback from the produced completions 
- humans rank completions to prompts for a task

- ranking to pairwise for supervised learning

- ranking gives more training data to train the reward model in comparison for instance to a thumbs up/down approach

- use the model as a binary classifier
- a reward model can be as well an LLM such as BERT for instance

### RLHF: Fine-tuning with reinforcement learning

- using a reward model within the RLHF
- remember one should start with an LLM that already has good performance on your task of interests

1) pass prompt P to an instruct LLM get the output X
2) pass the pair (P,X) to the reward model, and the get reward score
3) passs the reward value to the RL algorithm to updarted the wieght os the LLM
4) RL-updated LLM

- this is repeat and the LLM should converge to a human-aligned LLM and the reward should improve after each iteration

- stop when some defined threshold value for helpfulness is reached or this is repeated for a number n of steps

### RL Algorithm

- Proximal Policy Optimization (PPO)

- PPO makes updates to the LLM. The updates are small and within a bounded region, resulting in an updated LLM that is close to the previous version, hence the name Proximal Policy Optimization. 

- You start PPO with your initial instruct LLM, then at a high level, each cycle of PPO goes over two phases:

	- In Phase I, the LLM, is used to carry out a number of experiments, completing the given prompts.
		
		- These experiments allow you to update the LLM against the reward model in Phase II
		- The reward model captures the human preferences, can define how helpful, harmless, and honest the responses are. 
		- The expected reward of a completion is an important quantity used in the PPO objective.
		- We estimate this quantity through a separate head of the LLM called the value function.
		- Calculate value loss
	
	- In Phase II you make a small updates to the model and evaluate 	the impact of those updates on	your alignment goal for the model.

        - The model weights updates are 	guided by the prompt completion,	losses, and rewards.
		- PPO also ensures to keep the model updates within 	a certain small region called the trust region. 	This is where the proximal aspect of PPO comes into play. 	Ideally, this series of small updates 	will move the model towards higher rewards.



In PPO, the goal is to find an improved policy for an agent by iteratively updating its parameters based on the rewards received from interacting with the environment. However, updating the policy too aggressively can lead to unstable learning or drastic policy changes. To address this, PPO introduces a constraint that limits the extent of policy updates. This constraint is enforced by using KL-Divergence.

To understand how KL-Divergence works, imagine we have two probability distributions: the distribution of the original LLM, and a new proposed distribution of an RL-updated LLM. KL-Divergence measures the average amount of information gained when we use the original policy to encode samples from the new proposed policy. By minimizing the KL-Divergence between the two distributions, PPO ensures that the updated policy stays close to the original policy, preventing drastic changes that may negatively impact the learning process.

https://huggingface.co/blog/trl-peft

KL-divergence.png

### Reward Hacking

- As the policy tries to optimize the reward, it can diverge too much from the initial language model.
 
- For example the model can start to generate completions that will lead to very low toxicity scores by including phrases like most awesome, most incredible, sounding very exaggerated.

- The model could also start generating nonsensical, grammatically incorrect text that just happens to maximize the rewards in a similar way, outputs like this are definitely not very useful. To prevent our board hacking from happening, you can use the initial instruct LLM as performance reference. Let's call it the reference model. The weights of the reference model are frozen and are not updated during iterations of RHF. This way, you always maintain a single reference model to compare to. During training, each prompt is passed to both models, generating a completion by the reference LLM and the intermediate LLM updated model. At this point, you can compare the two completions and calculate a value called the Kullback-Leibler divergence, or KL divergence for short. KL divergence is a statistical measure of how different two probability distributions are. You can use it to compare the completions off the two models and determine how much the updated model has diverged from the reference. 


- KL divergence is calculated for each generate a token across the whole vocabulary off the LLM. This can easily be tens or hundreds of thousands of tokens. However, using a softmax function, you've reduced the number of probabilities to much less than the full vocabulary size. Keep in mind that this is still a relatively compute expensive process. You will almost always benefit from using GPUs. 

- Once you've calculated the KL divergence between the two models, you added as a term to the reward calculation. This will penalize the RL updated model if it shifts too far from the reference LLM and generates completions that are two different.

- Note that you now need to full copies of the LLM to calculate the KL divergence, the frozen reference LLM, and the oral updated PPO LLM.

- By the way, you can benefit from combining our relationship with puffed. In this case, you only update the weights of a path adapter, not the full weights of the LLM. This means that you can reuse the same underlying LLM for both the reference model and the PPO model, which you update with a trained path parameters. This reduces the memory footprint during training by approximately half.

- Once you have completed your RHF alignment of the model, you will want to assess the model's performance. The number you'll use here is the toxicity score, this is the probability of the negative class, in this case, a toxic or hateful response averaged across the completions. If RHF has successfully reduce the toxicity of your LLM, this score should go down. First, you'll create a baseline toxicity score for the original instruct LLM by evaluating its completions off the summarization data set with a reward model that can assess toxic language.

- Then you'll evaluate your newly human aligned model on the same data set and compare the scores.

### Scaling Human Feedback

Although you can use a reward model to eliminate the need for human evaluation during RLHF fine tuning, the human effort required to produce the trained reward model in the first place is huge. 

The labeled data set used to train the reward model typically requires large teams of labelers, sometimes many thousands of people to evaluate many prompts each. This work requires a lot of time and other resources which can be important limiting factors. As the number of models and use cases increases, human effort becomes a limited resource.

Methods to scale human feedback are an active area of research.

One idea to overcome these limitations is to scale through model self supervision.

Constitutional AI is one approach of scale supervision.

First proposed in 2022 by researchers at Anthropic, Constitutional AI is a method for training models using a set of rules and principles that govern the model's behavior.
Together with a set of sample prompts, these form the constitution.

You then train the model to self critique and revise its responses to comply with those principles.

Constitutional AI is useful not only for scaling feedback, it can also help address some unintended consequences of RLHF.

For example, depending on how the prompt is structured, an aligned model may end up revealing harmful information as it tries to provide the most helpful response it can.

As an example, imagine you ask the model to give you instructions on how to hack your neighbor's WiFi.

Because this model has been aligned to prioritize helpfulness, it actually tells you about an app that lets you do this, even though this activity is illegal.
 
Providing the model with a set of constitutional principles can help the model balance these competing interests and minimize the harm.
Here are some example rules from the research paper
that Constitutional AI I asks LLMs to follow.
For example, you can tell the model to choose
the response that is the most
helpful, honest, and harmless.
But you can play some bounds on this,
asking the model to prioritize harmlessness by
assessing whether it's response encourages illegal,
unethical, or immoral activity.
Note that you don't have to
use the rules from the paper,
you can define your own set of rules that is best
suited for your domain and use case.
When implementing the Constitutional AI method,
you train your model in two distinct phases.
In the first stage, you carry out supervised learning,
to start your prompt the model in ways that
try to get it to generate harmful responses,
this process is called red teaming.
You then ask the model to critique
its own harmful responses according to
the constitutional principles and
revise them to comply with those rules.
Once done, you'll fine-tune
the model using the pairs of red team
prompts and the revised constitutional responses.
Let's look at an example of
how one of these prompt completion pairs is generated.
Let's return to the WiFi hacking problem.

As you saw earlier,
this model gives you a harmful response
as it tries to maximize its helpfulness.
To mitigate this, you augment the prompt
using the harmful completion and
a set of predefined instructions that
ask the model to critique its response.
Using the rules outlined in the Constitution,
the model detects the problems in its response.
In this case, it correctly acknowledges
that hacking into someone's WiFi is illegal.
Lastly, you put all the parts
together and ask the model to write
a new response that removes
all of the harmful or illegal content.
The model generates a new answer
that puts the constitutional principles
into practice and does not
include the reference to the illegal app.
The original red team prompt,
and this final constitutional response
can then be used as training data.
You'll build up a data set of
many examples like this to create
a fine-tuned NLM that has learned how
to generate constitutional responses.
The second part of
the process performs reinforcement learning.
This stage is similar to RLHF,
except that instead of human feedback,
we now use feedback generated by a model.

This is sometimes referred to as reinforcement learning from AI feedback or RLAIF. Here you use the fine-tuned model from the previous step to generate a set of responses to your prompt. You then ask the model which of the responses is preferred according to the constitutional principles.

The result is a model generated preference dataset that you can use to train a reward model. With this reward model, you can now fine-tune your model further using a reinforcement learning algorithm like PPO, as discussed earlier.

## Laboratory Exercise 3


 2 - Load FLAN-T5 Model, Prepare Reward Model and Toxicity Evaluator

    2.1 - Load Data and FLAN-T5 Model Fine-Tuned with Summarization Instruction
    2.2 - Prepare Reward Model
    2.3 - Evaluate Toxicity

3 - Perform Fine-Tuning to Detoxify the Summaries

    3.1 - Initialize PPOTrainer
    3.2 - Fine-Tune the Model
    3.3 - Evaluate the Model Quantitatively
    3.4 - Evaluate the Model Qualitatively


## Large Language Models-powered Applications


LLM optimisation techniques

### Distillation:

1. Freeze the teacher model's weights and use it to generate completions for your training data. At the same time, you generate completions for the training data using your student model.

2. The knowledge distillation between teacher and student model is achieved by __minimizing a loss function called the distillation loss__. To calculate this loss, distillation __uses the probability distribution over tokens that is produced by the teacher model's softmax layer__.

3. Now, the teacher model is already fine tuned on the training data. So the probability distribution likely closely matches the ground truth data and won't have much variation in tokens. That's why Distillation applies a little trick adding a temperature parameter to the softmax function. As you learned in lesson one, a higher temperature increases the creativity of the language the model generates. With a temperature parameter greater than one, the probability distribution becomes broader and less strongly peaked. This softer distribution provides you with a set of tokens that are similar to the ground truth tokens.

- __soft labels__: freeze the teacher model's weights and use it to generate completions for your training data (adding a temperature parameter)

- __soft predictions__: generate completions for the training data using your student model (adding a temperature parameter)

In parallel, you train the student model to generate the correct predictions based on your ground truth training data.
Here, you don't vary the temperature setting and instead use the standard softmax function.

__hard predictions__: train the student model to generate the correct predictions based on your ground truth training data, don't vary the temperature setting use the standard softmax function

__hard labels__: ground truth

The loss between these two is the __student loss__. The combined __distillation and student losses__ are used to update the weights of the student model via back propagation.__

The key benefit of distillation methods is that the smaller student model can be used for inference in deployment instead of the teacher model.

In practice, distillation is not as effective for generative decoder models. It's typically more effective for encoder only models, such as Burt that have a lot of representation redundancy. Note that with Distillation, you're training a second, smaller model to use during inference. You aren't reducing the model size of the initial LLM in any way.

### Quantisation:

- quantization quantization-training
- post-training quantization

### Pruning: 

- remove weights with values close or equal to zero
- full model-retraining 
- PEFT/LoRA
- Post-Training

### Generative AI Project Lifecycle Cheat Sheet

## Using the LLM in Applications

- Augment LLM knowledge with external components

- Retrieval augmented generation (RAG)

	- Retriever
		- Query Encoder - encodes the data in the same format as the external documents
		- External information sources

	- Extended prompt that contains information retrieved from external documents is then passed to the LLM
	

- Reasoning tasks with multiple steps
- Chain of thought prompting

- Program-aided language models (PAL)

- ReAct: Combining reasoning and action
	- https://arxiv.org/abs/2210.03629
	- ReAct: Synergizing Reasoning and Acting in Language Models

- LangChainn:
	- tools
	- prompt templates
	- memory
	- agents: PAL, ReAct

## LLM application architectures

LLM is only one part of the history...


### Issues with LLM

toxiticiy
hallucination
use of intellectual property
