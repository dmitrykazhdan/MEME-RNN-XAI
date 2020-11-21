# MEME: Generating RNN Model Explanations via Model Extraction

This repository contains an implementation of MEME.
MEME is a (M)odel (E)xplanation via (M)odel (E)xtraction framework, which can be used for analysing 
RNN models via explainable concept-based extracted models, in order to explain and improve 
performance of RNN models, as well as to extract useful knowledge from them.
 
For further details, see our paper (link coming soon).

The experiments use the following open-source datasets:

- [Room Occupation Prediction](https://machinelearningmastery.com/how-to-predict-room-occupancy-based-on-environmental-factors/)
- [MIMIC-III In-Hospital Mortality Prediction](https://arxiv.org/pdf/1703.07771.pdf)


Abstract
---

Recurrent Neural Networks (RNNs) have achieved remarkable performance on a range of tasks. 
A key step to further empowering RNN-based approaches is improving their explainability and 
interpretability. In this work we present MEME: a model extraction approach capable of 
approximating RNNs with interpretable models represented by human-understandable concepts and 
their interactions. We demonstrate how MEME can be applied to two multivariate, continuous data case 
studies: Room Occupation Prediction, and In-Hospital Mortality Prediction. 
Using these case-studies, we show how our extracted models can be used to interpret RNNs both 
locally and globally, by approximating RNN decision-making via interpretable concept interactions.


Visual Abstract
---

![alt text](https://github.com/dmitrykazhdan/MEME-RNN-XAI/blob/master/figures/visual_abstract.png)

Given an RNN model, we: (1) approximate its hidden space by a set of concepts. 
(2) approximate its hidden space dynamics by a set of transition functions, one per concept. 
(3) approximate its output behaviour by a concept-class mapping, specifying an output class label for every concept. 
For every step in (1)-(3), the parts of the RNN being approximated are highlighted in red. 
In (a)-(c) we cluster the RNN's training data points in their hidden representation (assumed to be two-dimensional, in this example), 
and use the clustering to produce a set of concepts (in this case: _sick_, _healthy_ and _uncertain_, written as _unc._). 
In (d)-(f) we approximate the hidden function of the RNN by a function _F<sub>C</sub>_, which predicts transitions 
between the concepts. We represent this function by a set of functions, one per concept 
(in this case: _F<sub>s</sub>_, _F<sub>u</sub>_, _F<sub>h</sub>_). In (g)-(i) we approximate the output behaviour of the RNN by a 
function _S_, which predicts the output class from a concept. This function is represented by a concept-class mapping, 
specifying an output label for every concept (in this case: _healthy_&rarr;0, _sick_&rarr;1, and _unc_&rarr;1). Collectively, steps (1)-(3) are used to produce our extracted model, consisting of concepts, 
their interactions, and their corresponding class labels.


Processing Example
---

![alt text](https://github.com/dmitrykazhdan/MEME-RNN-XAI/blob/master/figures/local_explanation.png)

Extracted model sequence processing for three timesteps (_t = 0,1,2_), with _uncertain_ as the initial concept. 
For each timestep _t_, the concept the model is in at time _t_ is highlighted with a double border. 
We show the input data (**x**<sub>1</sub>, **x**<sub>2</sub>, **x**<sub>3</sub>), the corresponding concept 
transition sequence (_uncertain_ &rarr; _uncertain_ &rarr; _sick_ &rarr; _sick_), 
and the explanations for each transition function prediction. In this example, the class labels outputted by the model 
are not shown.


Prerequisites
---
TBC...

Citing
---

If you find this code useful in your research, please consider citing:

```
TBC...
```

