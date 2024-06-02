---
title: Shuyan Zhou_Solving Real-World Tasks with AI Agents
allDay: false
startTime: 12:00
endTime: 13:00
date: 2024-03-11
completed: null
---
Shuyan Zhou
# Areas
## AI agent evaluation
### WebArena: 
800ish tasks in realistic, interactive app
#### Conclusion
GPT4 is bad at tool use, arithematic, abstract reasoning, doesn't have update-to-date knowledge with cut-off knowledge base. 

## Speak "AI" language
To improve AI's ability in tool use & reasoning
### PaL
Replace "chain of thought" in natural language with symbolic language (interleave between natural language and programming language): python code with meanful variable name and comments
#### Training
- few-shot in context learning 
#### advantages
- offloading solving to API: can be extended to multi-modal domain
- output code template rather than example. 

## Learning by reading
Make GPT have up-to-date knowledge
### DocPrompting
Retrieval-then-generation
#### Training 
Contrast training the retriever to train the embeddings, then retrieve the k-nearest neighbor 


