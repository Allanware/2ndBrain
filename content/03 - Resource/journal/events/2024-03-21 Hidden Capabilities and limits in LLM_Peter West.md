---
title: Hidden Capabilities and limits in LLM_Peter West
allDay: false
startTime: 12:00
endTime: 13:00
date: 2024-03-21
completed: null
---
# Is scale all you need in LM
Sufficient? Linguistic Ambiguity, theory of mind, etc 
Necessary? 

## Hidden capabilities in Compact LM

### Inference-time algo: 
GPT-2 (1000x smaller than GPT-4, 2019)
Summarization task 
- Guided by information theory: the Information Bottleneck Method (need next sentence) , and 
- Discrete Search 
- No Fine-tuning, but need to generate 
- Another version: bottlesum_self: self-supervised learning by training bottle_sum on self-generated data
### Knowledge lacked by compact LM
Can we transfer the knowledge from LLM to compact LM
#### Symbolic knowledge distillation 
Knowledge distillation: teacher-student model using cross-entropy loss/objective. 
Issue: no access to proprietary model/probability distribution, and intractable number of strings 
Solution: Sampling some big knowledge source (GPT-3) and train a filter/critic model: we get a filtered, much larger set and more accurate knowledge source => then we use it to train a student model.

HE then goes on to apply to different domains.
## Counterintuitive Limits in Extreme-scale LM
### Generative Model Paradox
GM acquire generation abilities more effectively than understanding. On the other hand, generation requires understanding. 
#### To formalize 
GPT VS Human in Generation (short-form question answering) VS Understanding (multiple-choice question)

Or more stronger, does good generation suggest self-understanding (e.g. test on their own generated text)

## Future work
### Standardizing Symbolic Knowledge Distillation
Standardized Distillation process

### Improve Pretraining with Generated Knowledge
- Extract Linguistic Formalism
- Have more control over model 

### Explaining Limits of LLM
How models differ from human
- Memorization + composition 