---
title: Ramya Korlakai Vinayak
allDay: false
startTime: 12:30
endTime: 13:30
date: 2024-04-03
completed: null
---
# Towards Plurality: foundations of learning from diverse human preferences 

## Motivation
- Align pre-trained model to human preference
- Modeling/learning human preferences 

## Alignment
- learn a reward function that score the "outputs" from the pre-trained model that are preferred more by humans with higher score. Then, use the reward function to 
	- either fine-tune the pre-trained model
	- or generate a bunch and output the top ranked
- Deep reinforcement Learning from Human preferences (2017)
	- casting the preference (a preference choice over an output pair) score into a cross-entropy loss: it assumes a single ordering of the outputs by humans 

## Plurality of Alignment
- Before, it is learning an average preference among all population. Individual preference is seen as "noise". 
- A Roadmap to Pluralistic Alignment

## Preference models
### Ideal Point Model
- Coombs, 1950
- Casting preference to distance between the items and the ideal point within the feature space 

## Goal
### Simultaneously learn unknown metric 
we can obtain the feature space from the pre-trained model, but euclidian distance may not be the metric) and multiple user preferences (no longer assuming uniformity)
- One for all: Simultaneous metric and preference learning over multiple users (2022)
	- with no additional cost (in terms of number of queries) compared to the case where metric is known. 
	- E.g. Color preference data

### Leaning metric with limited budget