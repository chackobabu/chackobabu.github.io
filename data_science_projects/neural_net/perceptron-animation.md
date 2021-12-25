---
layout: default
title: Simple Perceptron
mathjax: true
---

#### **Perceptron Model (a simple a neural network)**
<br>
**Please note:**

This is an over-simplified neural network model. I have consciously used simple functions and avoided complex math operations and formulae.

However I hope that this model does justice in terms of depicting the basic working of neural network. That was the only intention. 

My main reference: <a href="https://natureofcode.com/book/chapter-10-neural-networks/" target="_blank" rel="noopener noreferrer"><https://natureofcode.com/book/chapter-10-neural-networks/></a>

```python
import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
import seaborn as sns
```


```python
random.seed(1)
```


```python
def activation_func(sum_):
# I define a simple activation function, which returns +1 if the sum (passed as an argument) is positive or equal to zero, or otherwise -1 is returned. 1 and -1 are the two targets in the data. The value returned by this function becomes the target predicted by the model.
    if sum_>=0:
        return 1
    else:
        return -1
```

**Creating a class (an object constructor) called Perceptron**

This does the following things for each set of inputs.

1. Initializes the object with random weights. by the __init__()
<div class="math-container">  
$$
\text{number of weights = number of features = n}\\
\text{weight}_{i} = w_{i},\ \begin{equation}
                            \left\{
                            \begin{aligned}
                                \text{where}\ w\in [-1,1] \\
                                i \in [1,\text{n}]
                            \end{aligned}
                            \right.
                            \end{equation}
$$
</div>


2. Forward propogation. by **forward_prop()**

<div class="math-container">      
$$\text{weighted sum} = \sum_{i=1}^{n}\text{weight}_{i} \times \text{feature}_{i}\\
\text{predicted target} = \begin{equation}
                          \left\{
                          \begin{aligned}
                              1,\ \text{if weighted sum} \ge 0\\
                              -1,\ \text{if weighted sum} < 0\\
                          \end{aligned}
                          \right.
                          \end{equation}
$$
</div>

    
3. Backward propogation. by **back_prop()**

<div class="math-container">   
$$
\text{error} = \text{target} - \text{predicted target}\\
\text{updated weights}_{i} = \text{weights}_{i} + (\text{error}\times \text{feature}_{i} \times \text{learning rate}),\ i \in [1,\text{n}]
$$
</div>

```python
class Perceptron(): 
    
    def __init__(self,size, learning_rate):
        
# Size denotes the number of features per input. The number of features and number of weights should be the same. i.e there should be one weight per feature. Here the number of features and number of weights equal to two. 

# learning rate defines the rate at which the model updates the weights after each iteration. There are risks of the model overshooting the targets if the rate is too high.

        self.size = size
        self.learning_rate = learning_rate
        self.weights = []
        
# this loop initializes random weights between -1 to 1. I have not used biases here, as weights and biases behave similarly in a neural  network. Since my intention is just to try to understand the basic working of a neural network, I felt the biases could be omitted for sake of simplicity. 

        for i in range(self.size): 
            w = random.uniform(-1,1)
            self.weights.append(w)
        self.updated_weights = self.weights


    def forward_prop(self, features, weights=False):
        
# this method predicts the target for the input. It calculates the sum of features*weights and pass it to the activation function which returns the sign of the sum (the sign becomes the predicted target). when the method is called for the first time the weights used are the ones with which object was initialized. when the method is called for the second and subsequent times, updated weights are passed. 

#This is an abstraction of forward propogation.

        sum_ = 0
        if weights == False:
            for i in range(self.size): 
                sum_ += features[i]*self.weights[i]
            return sum_, activation_func(sum_), self.weights
        else:
            for i in range(self.size): 
                sum_ += features[i]*weights[i]
            return sum_, activation_func(sum_), weights  
    

    def back_prop(self, features, target, predicted_target, old_weights):
        
# this method is where the weights are updated based on weights from the previous iteration. This is an abstraction of backward propogation where weights are adjusted based on the error value. Normally in neural networks weights are adjusted as per sensitivity of each weight to the error. (through a process called gradient descent). But this being only a demo, weights are adjusted as per the equation: new_weight = old_weight + learning_rate*input*error (for each input and weight).

        for i in range(self.size):
            error = target - predicted_target
            new_weight = old_weights[i] + self.learning_rate*error*features[i]
            self.updated_weights[i] = new_weight
        return self.updated_weights
```


```python
# A class to construct points with random x, y, and a target that depends on a simple condition. If x>=y, target = 1 else target = -1.

class Point(): 
    def __init__(self):
        self.x = random.uniform(0,100)
        self.y = random.uniform(0,100)
        
        if self.x>=self.y:
            self.target = 1
        else:
            self.target = -1
```


```python
# we create an array of 100 points, each of which are objects of Point, initalized with x,y and target. 
nof_points = 100 

p = []

for i in range(nof_points):
    p.append(Point()) # initializing points

```


```python
# save the 100 points to a dataframe.     
df_points = pd.DataFrame(columns=['x','y','target'])

for i in range(nof_points):
    df_points.loc[i] = [p[i].x, p[i].y, p[i].target]
```


```python
# This is our the training data. Actually this being an very simple example, we show only training of the model.  there is no testing data. 

df_points.head()
```



<pre>
<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>x</th>
      <th>y</th>
      <th>target</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>13.436424</td>
      <td>84.743374</td>
      <td>-1.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>76.377462</td>
      <td>25.506903</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>49.543509</td>
      <td>44.949106</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>65.159297</td>
      <td>78.872335</td>
      <td>-1.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>9.385959</td>
      <td>2.834748</td>
      <td>1.0</td>
    </tr>
  </tbody>
</table>
</div>
</pre>



```python
# plotting the data. 
plt.figure(dpi=100)
sns.scatterplot(x='x',y='y', data=df_points, hue=pd.Categorical(df_points['target']), legend=False)
sns.lineplot(x='x',y='x', data=df_points)
plt.xticks([])
plt.xlabel("")
plt.yticks([])
plt.ylabel("")
plt.show()
```


<img src="/assets/img/perceptron.png" alt="classification" width="75%">


This is the classification that the model will attempt to predict. All points with x >= y are shown as blue dots, and all points with x < y are shown as orange dots. 

This model will only be trained, and not tested.


```python
# initialize a perceptron object with size two and learning rate 0.001. I have kept the rate lower than usual, because it helps the animation (later) looks better. Usually learning rates are at 0.01, I guess.

Brain = Perceptron(2, 0.001) 
```


```python
# Here we take each of the 100 points, make the perceptron each's targets, by calling the forward_prop method with features of each point as arguments. 

# we also keep track of the weights used for each point, so as to use them later on for back-propagation. 

# this loop runs 100 times, once for each point.

for i in range(nof_points):
    features = [df_points.loc[i,'x'], df_points.loc[i,'y']]
    outcome = Brain.forward_prop(features)
    df_points.loc[i, 'iter0_weight1'] = outcome[2][0]
    df_points.loc[i, 'iter0_weight2'] = outcome[2][1]
    df_points.loc[i,'pred_0'] = outcome[1]
```


```python
df_points.head()
```



<pre>
<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>x</th>
      <th>y</th>
      <th>target</th>
      <th>iter0_weight1</th>
      <th>iter0_weight2</th>
      <th>pred_0</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>13.436424</td>
      <td>84.743374</td>
      <td>-1.0</td>
      <td>-0.349309</td>
      <td>0.743243</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>76.377462</td>
      <td>25.506903</td>
      <td>1.0</td>
      <td>-0.349309</td>
      <td>0.743243</td>
      <td>-1.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>49.543509</td>
      <td>44.949106</td>
      <td>1.0</td>
      <td>-0.349309</td>
      <td>0.743243</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>65.159297</td>
      <td>78.872335</td>
      <td>-1.0</td>
      <td>-0.349309</td>
      <td>0.743243</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>9.385959</td>
      <td>2.834748</td>
      <td>1.0</td>
      <td>-0.349309</td>
      <td>0.743243</td>
      <td>-1.0</td>
    </tr>
  </tbody>
</table>
</div>
</pre>



```python
iteration = 1
stop_iterations = False 

while(stop_iterations==False):
    
# in this loop for each points, train funtion is called with features, target, and the previously guessed values. based of these, errors for each points is calculated, and weights are updated for each point, and new guesses are made using the updated weights. Each of these values is stored as subsequent columns in the dataframe. 

# the loop ends after the iteration where all the predicted targets are equal to the correct targets.  

    for i in range(nof_points):
        
        features = [df_points.loc[i,'x'], df_points.loc[i,'y']]
        target = df_points.loc[i,'target']
        predicted_target = df_points.loc[i,f'pred_{iteration-1}']
        old_weights = [df_points.loc[i,f'iter{iteration-1}_weight1'],\
                       df_points.loc[i,f'iter{iteration-1}_weight2']]
        
        udpdated_weights = Brain.back_prop(features, target, predicted_target, old_weights)
        
        outcomes = Brain.forward_prop(features, udpdated_weights)
        
        df_points.loc[i, f'iter{iteration}_weight1'] = outcomes[2][0]
        df_points.loc[i, f'iter{iteration}_weight2'] = outcomes[2][1]
        df_points.loc[i,f'pred_{iteration}'] = outcomes[1]
        
    if all(df_points[f'pred_{iteration}'] == df_points['target']):
        print(f"our model predicts all targets correctly after {iteration+1} iterations")
        stop_iterations=True
        
    iteration+=1
```

    the model predicts all targets correctly after 17 iterations



```python
df_points.head()
```



<pre>
<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>x</th>
      <th>y</th>
      <th>target</th>
      <th>iter0_weight1</th>
      <th>iter0_weight2</th>
      <th>pred_0</th>
      <th>iter1_weight1</th>
      <th>iter1_weight2</th>
      <th>pred_1</th>
      <th>iter2_weight1</th>
      <th>...</th>
      <th>pred_13</th>
      <th>iter14_weight1</th>
      <th>iter14_weight2</th>
      <th>pred_14</th>
      <th>iter15_weight1</th>
      <th>iter15_weight2</th>
      <th>pred_15</th>
      <th>iter16_weight1</th>
      <th>iter16_weight2</th>
      <th>pred_16</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>13.436424</td>
      <td>84.743374</td>
      <td>-1.0</td>
      <td>-0.349309</td>
      <td>0.743243</td>
      <td>1.0</td>
      <td>-0.376182</td>
      <td>0.573756</td>
      <td>1.0</td>
      <td>-0.403054</td>
      <td>...</td>
      <td>-1.0</td>
      <td>-0.456800</td>
      <td>0.065296</td>
      <td>-1.0</td>
      <td>-0.456800</td>
      <td>0.065296</td>
      <td>-1.0</td>
      <td>-0.456800</td>
      <td>0.065296</td>
      <td>-1.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>76.377462</td>
      <td>25.506903</td>
      <td>1.0</td>
      <td>-0.349309</td>
      <td>0.743243</td>
      <td>-1.0</td>
      <td>-0.196554</td>
      <td>0.794257</td>
      <td>1.0</td>
      <td>-0.196554</td>
      <td>...</td>
      <td>1.0</td>
      <td>-0.196554</td>
      <td>0.794257</td>
      <td>1.0</td>
      <td>-0.196554</td>
      <td>0.794257</td>
      <td>1.0</td>
      <td>-0.196554</td>
      <td>0.794257</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>49.543509</td>
      <td>44.949106</td>
      <td>1.0</td>
      <td>-0.349309</td>
      <td>0.743243</td>
      <td>1.0</td>
      <td>-0.349309</td>
      <td>0.743243</td>
      <td>1.0</td>
      <td>-0.349309</td>
      <td>...</td>
      <td>1.0</td>
      <td>-0.349309</td>
      <td>0.743243</td>
      <td>1.0</td>
      <td>-0.349309</td>
      <td>0.743243</td>
      <td>1.0</td>
      <td>-0.349309</td>
      <td>0.743243</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>65.159297</td>
      <td>78.872335</td>
      <td>-1.0</td>
      <td>-0.349309</td>
      <td>0.743243</td>
      <td>1.0</td>
      <td>-0.479627</td>
      <td>0.585498</td>
      <td>1.0</td>
      <td>-0.609946</td>
      <td>...</td>
      <td>-1.0</td>
      <td>-0.609946</td>
      <td>0.427754</td>
      <td>-1.0</td>
      <td>-0.609946</td>
      <td>0.427754</td>
      <td>-1.0</td>
      <td>-0.609946</td>
      <td>0.427754</td>
      <td>-1.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>9.385959</td>
      <td>2.834748</td>
      <td>1.0</td>
      <td>-0.349309</td>
      <td>0.743243</td>
      <td>-1.0</td>
      <td>-0.330537</td>
      <td>0.748913</td>
      <td>-1.0</td>
      <td>-0.311765</td>
      <td>...</td>
      <td>1.0</td>
      <td>-0.217905</td>
      <td>0.782929</td>
      <td>1.0</td>
      <td>-0.217905</td>
      <td>0.782929</td>
      <td>1.0</td>
      <td>-0.217905</td>
      <td>0.782929</td>
      <td>1.0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 54 columns</p>
</div>
</pre>


```python
# creating another dataframe called df_animation to be used for animating the process. This dataframe contains the predictions that the model makes over iterations. 

df_animation = df_points.set_index(['x','y','target']).filter(regex='pred').reset_index()
```


```python
df_animation.head()
```

<pre>
<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>x</th>
      <th>y</th>
      <th>target</th>
      <th>pred_0</th>
      <th>pred_1</th>
      <th>pred_2</th>
      <th>pred_3</th>
      <th>pred_4</th>
      <th>pred_5</th>
      <th>pred_6</th>
      <th>pred_7</th>
      <th>pred_8</th>
      <th>pred_9</th>
      <th>pred_10</th>
      <th>pred_11</th>
      <th>pred_12</th>
      <th>pred_13</th>
      <th>pred_14</th>
      <th>pred_15</th>
      <th>pred_16</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>13.436424</td>
      <td>84.743374</td>
      <td>-1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>-1.0</td>
      <td>-1.0</td>
      <td>-1.0</td>
      <td>-1.0</td>
      <td>-1.0</td>
      <td>-1.0</td>
      <td>-1.0</td>
      <td>-1.0</td>
      <td>-1.0</td>
      <td>-1.0</td>
      <td>-1.0</td>
      <td>-1.0</td>
      <td>-1.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>76.377462</td>
      <td>25.506903</td>
      <td>1.0</td>
      <td>-1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>49.543509</td>
      <td>44.949106</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>65.159297</td>
      <td>78.872335</td>
      <td>-1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>-1.0</td>
      <td>-1.0</td>
      <td>-1.0</td>
      <td>-1.0</td>
      <td>-1.0</td>
      <td>-1.0</td>
      <td>-1.0</td>
      <td>-1.0</td>
      <td>-1.0</td>
      <td>-1.0</td>
      <td>-1.0</td>
      <td>-1.0</td>
      <td>-1.0</td>
      <td>-1.0</td>
      <td>-1.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>9.385959</td>
      <td>2.834748</td>
      <td>1.0</td>
      <td>-1.0</td>
      <td>-1.0</td>
      <td>-1.0</td>
      <td>-1.0</td>
      <td>-1.0</td>
      <td>-1.0</td>
      <td>-1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
    </tr>
  </tbody>
</table>
</div>
</pre>


```python
from matplotlib.animation import FuncAnimation
from IPython import display
```


```python
nof_frames = len(df_animation.columns)-3
```


```python
nof_frames
```




    17




```python
fig = plt.figure(dpi=150)
sns.set_style('white')
plt.plot(df_animation['x'], df_animation['x'], linestyle='--', linewidth=0.1, color='brown', alpha=0.5)
plt.xticks([])
plt.xlabel("")
plt.yticks([])
plt.ylabel("")

plt.annotate("Blue Points", (0.07,0.9), xycoords='figure fraction')
plt.annotate("Orange Points", (0.805,0.085), xycoords='figure fraction')

def animate(frames):
    hue_change = pd.Categorical(df_animation[f"pred_{frames}"])
    sns.scatterplot(x=df_animation['x'], y=df_animation['y'], hue=hue_change, alpha=0.75, legend=False)
                                                                            
anim = FuncAnimation(fig, animate, frames=nof_frames, interval=500, repeat=True)
video = anim.to_html5_video()
html = display.HTML(video)
display.display(html)
plt.close()                                                                        
```
**animation of the model classifying the dots over 17 iterations**


<video width="600" height="400" controls autoplay loop>
  <source type="video/mp4" src="data:video/mp4;base64,AAAAHGZ0eXBNNFYgAAACAGlzb21pc28yYXZjMQAAAAhmcmVlAAC8YW1kYXQAAAKuBgX//6rcRem9
5tlIt5Ys2CDZI+7veDI2NCAtIGNvcmUgMTQ4IHIyNjQzIDVjNjU3MDQgLSBILjI2NC9NUEVHLTQg
QVZDIGNvZGVjIC0gQ29weWxlZnQgMjAwMy0yMDE1IC0gaHR0cDovL3d3dy52aWRlb2xhbi5vcmcv
eDI2NC5odG1sIC0gb3B0aW9uczogY2FiYWM9MSByZWY9MyBkZWJsb2NrPTE6MDowIGFuYWx5c2U9
MHgzOjB4MTEzIG1lPWhleCBzdWJtZT03IHBzeT0xIHBzeV9yZD0xLjAwOjAuMDAgbWl4ZWRfcmVm
PTEgbWVfcmFuZ2U9MTYgY2hyb21hX21lPTEgdHJlbGxpcz0xIDh4OGRjdD0xIGNxbT0wIGRlYWR6
b25lPTIxLDExIGZhc3RfcHNraXA9MSBjaHJvbWFfcXBfb2Zmc2V0PS0yIHRocmVhZHM9MTIgbG9v
a2FoZWFkX3RocmVhZHM9MiBzbGljZWRfdGhyZWFkcz0wIG5yPTAgZGVjaW1hdGU9MSBpbnRlcmxh
Y2VkPTAgYmx1cmF5X2NvbXBhdD0wIGNvbnN0cmFpbmVkX2ludHJhPTAgYmZyYW1lcz0zIGJfcHly
YW1pZD0yIGJfYWRhcHQ9MSBiX2JpYXM9MCBkaXJlY3Q9MSB3ZWlnaHRiPTEgb3Blbl9nb3A9MCB3
ZWlnaHRwPTIga2V5aW50PTI1MCBrZXlpbnRfbWluPTIgc2NlbmVjdXQ9NDAgaW50cmFfcmVmcmVz
aD0wIHJjX2xvb2thaGVhZD00MCByYz1jcmYgbWJ0cmVlPTEgY3JmPTIzLjAgcWNvbXA9MC42MCBx
cG1pbj0wIHFwbWF4PTY5IHFwc3RlcD00IGlwX3JhdGlvPTEuNDAgYXE9MToxLjAwAIAAAGIWZYiE
ABX//vfJ78Cm61tbtb+Tz0j8LLc+wio/blsTtOoAAAMAAAMAAAMAAAMDx+xGrqKoZ4qx2AAAAwAF
J2Z/+W92h4ADlqB+dNY2ixW/1JCr+YF+EoJN5UORNPBKUyDYCC1uFUGFiHf9cgEOApCiv/KTFkoj
w2rLMY8kNOWjuOqPl1YhaAdxjfSmKU/Fc49SbHUSd2m5qW8mMIptlGVECsMysvKAND9let6/v1j4
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
">
  Your browser does not support the video tag.
</video>


<a id=1></a>
