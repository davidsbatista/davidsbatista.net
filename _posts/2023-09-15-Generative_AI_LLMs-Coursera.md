---
layout: post
title: Generative AI with Large Language Models
date: 2023-09-15 00:00:00
tags: llms coursera
categories: [blog]
comments: true
disqus_identifier: 20230915
preview_pic: /assets/images/2023-09-15-2001.png
---

I'm happy to have completed this course and recommend it to anyone interested in delving into the intricacies of Transformer architecture and Large Language Models. The course covered a wide range of topics, including the Transformer architecture, the Generative AI lifecycle, it provided in-depth insights into fine-tuning LLMs with parameter-efficient fine-tuning (PEFT), LoRA, and soft-prompts. The course also explored advanced topics such as Reinforcement Learning with Human Feedback, chain-of-thought prompting, and the use of information retrieval and augmentation techniques to overcome knowledge limitations.


## __Week 1 - Introduction__

Discuss model pre-training and the value of continued pre-training vs fine-tuning
Define the terms Generative AI, large language models, prompt, and describe the transformer architecture that powers LLMs
Describe the steps in a typical LLM-based, generative AI model lifecycle and discuss the constraining factors that drive decisions at each step of model lifecycle
Discuss computational challenges during model pre-training and determine how to efficiently reduce memory footprint
Define the term scaling law and describe the laws that have been discovered for LLMs related to training dataset size, compute budget, inference requirements, and other factors.


### __Introduction to Transformers architecture__

Going through uses cases of generative AI with Large Language Models, given examples such as:  summarisation, translation or information retrieval; and also how those were achieved before Transformers came into play. There's also an introduction to the Transformer architecture which is the base component for Large Language Models, and also an overview of the inference parameters that one can tune.


### __Generative AI project life-cycle__

Then it's first introduced in the course the Generative AI project lifecycle which is followed up to the end of the course

__TODO__: imagem

- Generative AI project life-cycle ?
	- Scope: define use case
		- multi-tasks? specific tasks?
	- Select pre-trained or trained-own
	- Adapt and align model - iterative process
		- prompt engineering
		- fine-tuning
		- align with human feedback
		===========================
		- Evaluate		
	- Application Integration
		- OPtimize inference
		- agument model and build LLM-pxoewr applications


### __Prompt Engineering and Inference paramaters__

- Prompt Engineering 
	in-context learning
		- zero-shot
		- one-shot
		- few shot

- Inference
	- generative config: 
		- greedy vs. random sampling
		- top-k: select only from the top-k tokens
		- top-p: select from top results by probability and with a cumulative probability <= p
		- temperature: 
			higher temperature higher randomness, affects softmax directly and how probability is computed
			temperature >1
			temperature <1
			temperature = 1 softmax function at default, unaltered prob distribution


### __Laboratory Exercise about Prompt Engineering__


- Lab exercise

T5 model from huggingface

__ Generative AI Use Case: Summarize Dialogue __

Welcome to the practical side of this course. In this lab you will do the dialogue summarization task using generative AI. You will explore how the input text affects the output of the model, and perform prompt engineering to direct it towards the task you need. By comparing zero shot, one shot, and few shot inferences, you will take the first step towards prompt engineering and see how it can enhance the generative output of Large Language Models.


https://huggingface.co/blog/few-shot-learning-gpt-neo-and-inference-api

1 - without any prompt engineering - essentially asking the predict next sequence of words
2 - zero-shot - giving an instruction
3 - one-shot - giving an instruction with one example
4 - few shot - giving an instruction with a few examples (2~6)

5 - Generative Configuration Parameters for Inference
	pretty cool
	
	https://huggingface.co/docs/transformers/v4.29.1/en/main_classes/text_generation#transformers.GenerationConfig


Summarize Dialogue without Prompt Engineering

In this use case, you will be generating a summary of a dialogue with the pre-trained Large Language Model (LLM) FLAN-T5 from Hugging Face. The list of available models in the Hugging Face `transformers` package can be found [here](https://huggingface.co/docs/transformers/index). 

Let's upload some simple dialogues from the [DialogSum](https://huggingface.co/datasets/knkarthick/dialogsum) Hugging Face dataset. This dataset contains 10,000+ dialogues with the corresponding manually labeled summaries and topics. 


### __LLM pre-training and Scaling Laws__

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


learning memory optimizaions and parallel computing for efficient LLms training




## __Week 2 - Fine-Tuning__

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


###  Instruction Fine-Tuning

Instruction fine-tuning or simple fine-tuning trains the whole model parameters using examples that demonstrate how it should respond to a specific instruction.

Example:
	
	PROMPT, EXAMPLE TEXT, EXAMPLE COMPLETION

Since all the all of the model's weights are updated is known as: __full fine-tuning__ and it involves using many prompt-completion examples as the labeled training dataset to continue training the model by updating its weights.

Comparing to in-context learning, one only provides prompt-completion during inference, here we do it during training.

Adapt a foundation model through instruction fine-tuning, requires prompt templates and datasets


### Single-Task Fine-Tuning

The output of an LLM is a prob distribution across tokens, so you can compare the distribution of the completion and the training label and use standard cross-entropy function to calculate loss between the two token distributions, and use that loss to 
to update the model weights in standard back propagation.

This type of tuning, on a single task, leads to __catastrophic forgetting__. Fine-tuning the weights of the original LLM, yields great results on a single task degrads on other tasks, if the model was trained for several tasks


### Multi-Task Fine-Tuning

- FLAN-T5
- FLAN-PALM

https://arxiv.org/abs/2210.11416


### Model Evaluation

- ROUGE-n ngrams bla bla
- ROUGE-L longest common subsequence between generatedoutput and reference

- BLEU metric - avg(precision across range of n-gram sizes)

- BLEU focues on precision in. matching genearted output to the refernec text


### Benchmarks

- GLUE 2018
- SUPERGLUE 2019
- Leaderboards
- HELM Language Models
		- rouge and helm


###  Parameter Efficient Fine-Tuning (PEFT)

Full-fine tuning Large Language Models is challenging, you need lots of memory:

- not only the model in memory
- optimize states
- gradients
- forward activates
- temporary memory for training process
- this can be 12-20x the model's memory
	
Parameter Efficient Fine-Tuning (PEFT)
 - only a small number of trainable layers
 - LLM with additional layers for PEFT, new trainable layers
 - can often be performed on a single GPU
 - less prone to catraspojhic forgetting
	
PEFT methods
 - selective: select a subset of initial LLM parameters to fine-tune
 - LoRA:
	- reparamterize model weights using a low-rank representation
 - Additive
	- add trainable layers or parameters to model
		- adapters
		- soft prompts: prompt tuning
		
LoRA:
 - Low-Rank Adaptation for Large Language Models (LoRA)
 - two new matrices much lower dimensions, new weights for tokens, replace original weights
 - how to choose the rank for the matrices? original paper found plateu at 16
 - 4-32 good trade-off
 - decomposes weights into two smaller rank matrices and trains those instead of the full model
 - QLoRA (ideia: combined it with quantization techniques)


Soft Prompts:
 - improve without changing the weights
 - prompt tunning
 - not promot enginerrong 
	- prompt enginerrong: work on the language of input prompt
 - With prompt tuning, you add additional trainable tokens to your prompt and leave it up to the supervised learning process to determine their optimal values. The set of trainable tokens is called a soft prompt, and it gets prepended to embedding vectors that represent your input text. The soft prompt vectors have the same length as the embedding vectors of the language tokens. And including somewhere between 20 and 100 virtual tokens can be sufficient for good performance. The tokens that represent natural language are hard in the sense that they each correspond to a fixed loc
	- soft prompt
		a set of trainable tokens that are added to a prompt and whose values are updated during additional training to improve performance on specific tasks



review:
=======


### Laboratory Exercises


You just run code nothing is expected - although you can play around with the parameters


### Reading material


Multi-task, instruction fine-tuning

    Scaling Instruction-Finetuned Language Models

 - Scaling fine-tuning with a focus on task, model size and chain-of-thought data.

Introducing FLAN: More generalizable Language Models with Instruction Fine-Tuning

     - This blog (and article) explores instruction fine-tuning, which aims to make language models better at performing NLP tasks with zero-shot inference.

Model Evaluation Metrics

    HELM - Holistic Evaluation of Language Models

 - HELM is a living benchmark to evaluate Language Models more transparently. 

General Language Understanding Evaluation (GLUE) benchmark

 - This paper introduces GLUE, a benchmark for evaluating models on diverse natural language understanding (NLU) tasks and emphasizing the importance of improved general NLU systems.

SuperGLUE

 - This paper introduces SuperGLUE, a benchmark designed to evaluate the performance of various NLP models on a range of challenging language understanding tasks.

ROUGE: A Package for Automatic Evaluation of Summaries

 - This paper introduces and evaluates four different measures (ROUGE-N, ROUGE-L, ROUGE-W, and ROUGE-S) in the ROUGE summarization evaluation package, which assess the quality of summaries by comparing them to ideal human-generated summaries.

Measuring Massive Multitask Language Understanding (MMLU)

 - This paper presents a new test to measure multitask accuracy in text models, highlighting the need for substantial improvements in achieving expert-level accuracy and addressing lopsided performance and low accuracy on socially important subjects.

BigBench-Hard - Beyond the Imitation Game: Quantifying and Extrapolating the Capabilities of Language Models

     - The paper introduces BIG-bench, a benchmark for evaluating language models on challenging tasks, providing insights on scale, calibration, and social bias.

Parameter- efficient fine tuning (PEFT)

    Scaling Down to Scale Up: A Guide to Parameter-Efficient Fine-Tuning

 - This paper provides a systematic overview of Parameter-Efficient Fine-tuning (PEFT) Methods in all three categories discussed in the lecture videos.

On the Effectiveness of Parameter-Efficient Fine-Tuning

     - The paper analyzes sparse fine-tuning methods for pre-trained models in NLP.

LoRA

    LoRA Low-Rank Adaptation of Large Language Models

 -  This paper proposes a parameter-efficient fine-tuning method that makes use of low-rank decomposition matrices to reduce the number of trainable parameters needed for fine-tuning language models.

QLoRA: Efficient Finetuning of Quantized LLMs

     - This paper introduces an efficient method for fine-tuning large language models on a single GPU, based on quantization, achieving impressive results on benchmark tests.

Prompt tuning with soft prompts

    The Power of Scale for Parameter-Efficient Prompt Tuning

 - The paper explores "prompt tuning," a method for conditioning language models with learned soft prompts, achieving competitive performance compared to full fine-tuning and enabling model reuse for many tasks.




## __Week 3__

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













### LLM-powered Applications


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





### ISsues with LLM

toxiticiy
hallucination
use of intellectual property







<!--
<figure>
  <img style="width: 65%; height: 65%" src="/assets/images/2023-09-14-power_law_ent_freq.png">
  <figcaption>Source: rawpixel https://www.rawpixel.com/image/9975355/photo-image-art-space-vintage.</figcaption>
</figure>
-->









