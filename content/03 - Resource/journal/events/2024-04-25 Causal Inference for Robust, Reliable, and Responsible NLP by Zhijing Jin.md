---
title: Causal Inference for Robust, Reliable, and Responsible NLP by Zhijing Jin
allDay: false
startTime: 12:00
endTime: 13:00
date: 2024-04-25
completed: null
---
# Prev work
Paradigm shift in research of NLP, HCI, robotics to LLM. 

## Correlation vs Causation confusion
For example, in image classification, Cow correlates with green bg, and camel with yellow bg. We want the model to learn animal correlation with its label, but the model learns a **spurious correlation**. 
## Issues
- Model's implicit decision-making mechanisms
- model's explicit texture reasoning
# Aims 
## Robust NLP by a causal framework
### LLM testing
It is not about testing them on human exams. It is not the right abstraction level. 
Quantify the difference between real casual relationship and spurious correlation. 

### 2020: is BERT really robust?
A textfooler that paraphrases text to induce spurious correlation

### 2020: Invariance and sensitivity tests
- to make sure the correct causal relationship is invariant 
- to make sure the spurious one stay low. 
### 2023: a causal framework to quantify the robustness of mathematical reasoning in language models
1. list all the casual variables
	1. operands
	2. text relevant to operation and text not relevant
2. draw the desired vs model-learned causal variables
3. draw paths 
4. quantify the causal effect on paths 
## Reliable causal reasoning 
### Tasks 
1. Causal discovery (qualitative)
2. Casual Effect Reasoning (quantitative effect of one variable to another)
### 2023: Cladder 
On task 2. For example, vaccination is correlated with severity. There is one confounder variable: healthy condition of the patient.
Use causal graph and available data to form a causal query, and query the LLM. 
#### Causal Chain of thought: 
How to improve LLM correctness on this task: inject a 5 step prompt pipeline.
## Responsible NLP for social good 
### 2021: Mining the Cause of Political Decision-making from Social Media

Casual Policy analysis. 
X: Public Opinion (in terms of Tweets) 
Y: Social Distancing Policies. 
One confounder is the Covid 19 cases 

# Future work
## Interpretability, failness, robust
## GPT for science 

CausalNLP Tutorial: An Introduction to Causality for Natural Language Processing
- [Causal nlp: a path towards opening the black box of nlp](youtube.com/watch?v=Auls8ap0oA0)

