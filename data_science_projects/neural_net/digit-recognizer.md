---
layout: default
title: Digit Recognizer
mathjax: true
---

#### **Digit Recognizer**
<br>

**Neural network to recognize hand written digits from MNIST dataset.**


```python
import pandas as pd
import numpy as np
import random 
import matplotlib.pyplot as plt
```


```python
random.seed(1)
```

source of data: <a href="http://yann.lecun.com/exdb/mnist/" target="_blank" rel="noopener noreferrer"><http://yann.lecun.com/exdb/mnist></a>

```python
from tensorflow.keras.datasets import mnist     
(X_train, y_train), (X_test, y_test) = mnist.load_data()
```


```python
print(f"Shape of X_train: {X_train.shape}")
print(f"Shape of y_train: {y_train.shape}")
print(f"Shape of X_test: {X_test.shape}")
print(f"Shape of y_test: {y_test.shape}")
```

    Shape of X_train: (60000, 28, 28)
    Shape of y_train: (60000,)
    Shape of X_test: (10000, 28, 28)
    Shape of y_test: (10000,)



```python
# converting to 2-D array
X_train = X_train.reshape(60000, 784)
```


```python
X_test = X_test.reshape(10000, 784)
```


```python
X_test
```




    array([[0, 0, 0, ..., 0, 0, 0],
           [0, 0, 0, ..., 0, 0, 0],
           [0, 0, 0, ..., 0, 0, 0],
           ...,
           [0, 0, 0, ..., 0, 0, 0],
           [0, 0, 0, ..., 0, 0, 0],
           [0, 0, 0, ..., 0, 0, 0]], dtype=uint8)




```python
# Converting each of these arrays to dataframe. Just for ease of viewing.
```


```python
columns = [f"pixel{x}" for x in list(range(0,784))]
```


```python
X_train = pd.DataFrame(X_train, columns=columns)
```


```python
X_train.head()
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
      <th>pixel0</th>
      <th>pixel1</th>
      <th>pixel2</th>
      <th>pixel3</th>
      <th>pixel4</th>
      <th>pixel5</th>
      <th>pixel6</th>
      <th>pixel7</th>
      <th>pixel8</th>
      <th>pixel9</th>
      <th>...</th>
      <th>pixel774</th>
      <th>pixel775</th>
      <th>pixel776</th>
      <th>pixel777</th>
      <th>pixel778</th>
      <th>pixel779</th>
      <th>pixel780</th>
      <th>pixel781</th>
      <th>pixel782</th>
      <th>pixel783</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 784 columns</p>
</div>
</pre>



```python
X_test = pd.DataFrame(X_test, columns=columns)
```


```python
X_test.head()
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
      <th>pixel0</th>
      <th>pixel1</th>
      <th>pixel2</th>
      <th>pixel3</th>
      <th>pixel4</th>
      <th>pixel5</th>
      <th>pixel6</th>
      <th>pixel7</th>
      <th>pixel8</th>
      <th>pixel9</th>
      <th>...</th>
      <th>pixel774</th>
      <th>pixel775</th>
      <th>pixel776</th>
      <th>pixel777</th>
      <th>pixel778</th>
      <th>pixel779</th>
      <th>pixel780</th>
      <th>pixel781</th>
      <th>pixel782</th>
      <th>pixel783</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 784 columns</p>
</div>
</pre>



```python
y_train =pd.DataFrame(y_train, columns=['label'])
```


```python
y_train.head()
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
      <th>label</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>5</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>4</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>9</td>
    </tr>
  </tbody>
</table>
</div>
</pre>



```python
y_test = pd.DataFrame(y_test, columns=['label'])
```


```python
y_test.head()
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
      <th>label</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>7</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4</td>
    </tr>
  </tbody>
</table>
</div>
</pre>


X_train contains 784 features of 60000 handwritten digits. Each row represents a 28 $\times$ 28 greyscale image of a handwritten image. 

**Examples of handwritten images represented in rows**


```python
random_image_num = random.randint(0,784)

# I am taking a random number between 0 and 783, and taking the row from the dataframe, then reshaping it into a 28*28 matrix, and passing it into the imshow() of matplotlib.pyplot

random_image = X_train.loc[random_image_num].values.reshape(28,28)
plt.imshow(random_image, cmap='gray')
plt.axis('off')
plt.show()
```

<img src="/assets/img/digit-recg1.png" alt="digit_rec_output1" width="50%">

The goal of the neural network will be to guess these handwritten images.

Y_train contains 60000 labels. The labels range from 0 to 9. It represents the correct digit of the image represented by each row of train dataframe


```python
y_train['label'].unique()
```




    array([5, 0, 4, 1, 9, 2, 3, 6, 7, 8], dtype=uint8)



**Normalizing the data**

noramlizing the data between 0 and 1 speeds up the training process


```python
from sklearn.preprocessing import MinMaxScaler
```


```python
scaler = MinMaxScaler()
```


```python
X_train = scaler.fit_transform(X_train)
```


```python
X_train.min(), X_train.max()
```




    (0.0, 1.0)




```python
X_test = scaler.transform(X_test) 
```

we transorm **test data** using mean and standard deviation **of train data** and **not of test data**.  This is because we want to pretend as if the testing dataset is unseen, like in an actual scenario. 


```python
X_test.min(), X_test.max()
```




    (0.0, 24.0)



The digits in the y_train and y_test datasets need to be one-hot coded. 

for example, digit 5 should be represented as 

$$
\begin{bmatrix} 0\\ 0\\ 0\\ 0\\ 0\\ 1\\ 0\\ 0\\ 0\\ 0\\ \end{bmatrix}
$$


pandas get_dummies() can be used to get one-hot coded digits as shown below. 


```python
pd.get_dummies(y_train, columns=['label'], prefix="digit")
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
      <th>digit_0</th>
      <th>digit_1</th>
      <th>digit_2</th>
      <th>digit_3</th>
      <th>digit_4</th>
      <th>digit_5</th>
      <th>digit_6</th>
      <th>digit_7</th>
      <th>digit_8</th>
      <th>digit_9</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>59995</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>59996</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>59997</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>59998</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>59999</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>60000 rows × 10 columns</p>
</div>
</pre>



```python
y_train_prepd = pd.get_dummies(y_train, columns=['label'], prefix="digit").values
```


```python
y_test_prepd = pd.get_dummies(y_test, columns=['label'], prefix="digit").values
```

so here are our final datasets that we will be feeding into the model


```python
X_train
```




    array([[0., 0., 0., ..., 0., 0., 0.],
           [0., 0., 0., ..., 0., 0., 0.],
           [0., 0., 0., ..., 0., 0., 0.],
           ...,
           [0., 0., 0., ..., 0., 0., 0.],
           [0., 0., 0., ..., 0., 0., 0.],
           [0., 0., 0., ..., 0., 0., 0.]])




```python
X_test
```




    array([[0., 0., 0., ..., 0., 0., 0.],
           [0., 0., 0., ..., 0., 0., 0.],
           [0., 0., 0., ..., 0., 0., 0.],
           ...,
           [0., 0., 0., ..., 0., 0., 0.],
           [0., 0., 0., ..., 0., 0., 0.],
           [0., 0., 0., ..., 0., 0., 0.]])




```python
y_train_prepd
```




    array([[0, 0, 0, ..., 0, 0, 0],
           [1, 0, 0, ..., 0, 0, 0],
           [0, 0, 0, ..., 0, 0, 0],
           ...,
           [0, 0, 0, ..., 0, 0, 0],
           [0, 0, 0, ..., 0, 0, 0],
           [0, 0, 0, ..., 0, 1, 0]], dtype=uint8)




```python
y_test_prepd
```




    array([[0, 0, 0, ..., 1, 0, 0],
           [0, 0, 1, ..., 0, 0, 0],
           [0, 1, 0, ..., 0, 0, 0],
           ...,
           [0, 0, 0, ..., 0, 0, 0],
           [0, 0, 0, ..., 0, 0, 0],
           [0, 0, 0, ..., 0, 0, 0]], dtype=uint8)



**creating the model**


```python
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Dropout
```


```python
model = Sequential()

model.add(Dense(units=512, input_shape=(784,), activation='relu', name='input_layer'))
# this layer takes in the input, 784 features and each of these 784 features is connected to all 512 nodes of layer 1 

model.add(Dropout(0.2))
# 2% of the nodes (selected randomly) of the previous layer will get turned off (dropped out) in each iteration. inorder to prevent overfitting of the model. 

model.add(Dense(units=512, activation='relu', name='hidden_layer'))
#outputs from the input layer goes into the hidden layer.
model.add(Dropout(0.2))

model.add(Dense(units=10, activation='softmax', name='output_layer'))
# outputs of the 'output layer' will be a vector of probabilities, each denote the probability of the image representing a particular digit

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
```


```python
from tensorflow.keras.callbacks import EarlyStopping
```


```python
# Stop training if the 'val_loss' does not decrease from a minimum for 5 consecutive epochs
early_stop = EarlyStopping(monitor='val_loss', mode='min', patience=5, verbose=1)
```


```python
model.fit(x=X_train, y=y_train_prepd, epochs=25, validation_data=(X_test,y_test_prepd), batch_size=128, verbose=2, callbacks=[early_stop])
```

    Epoch 1/25
    469/469 - 3s - loss: 0.2488 - accuracy: 0.9253 - val_loss: 0.1141 - val_accuracy: 0.9639
    Epoch 2/25
    469/469 - 2s - loss: 0.0999 - accuracy: 0.9690 - val_loss: 0.0775 - val_accuracy: 0.9751
    Epoch 3/25
    469/469 - 3s - loss: 0.0690 - accuracy: 0.9781 - val_loss: 0.0781 - val_accuracy: 0.9754
    Epoch 4/25
    469/469 - 3s - loss: 0.0563 - accuracy: 0.9815 - val_loss: 0.0646 - val_accuracy: 0.9799
    Epoch 5/25
    469/469 - 3s - loss: 0.0453 - accuracy: 0.9847 - val_loss: 0.0700 - val_accuracy: 0.9798
    Epoch 6/25
    469/469 - 3s - loss: 0.0387 - accuracy: 0.9871 - val_loss: 0.0666 - val_accuracy: 0.9799
    Epoch 7/25
    469/469 - 3s - loss: 0.0326 - accuracy: 0.9895 - val_loss: 0.0632 - val_accuracy: 0.9819
    Epoch 8/25
    469/469 - 3s - loss: 0.0309 - accuracy: 0.9900 - val_loss: 0.0723 - val_accuracy: 0.9802
    Epoch 9/25
    469/469 - 3s - loss: 0.0280 - accuracy: 0.9907 - val_loss: 0.0663 - val_accuracy: 0.9814
    Epoch 10/25
    469/469 - 3s - loss: 0.0247 - accuracy: 0.9920 - val_loss: 0.0629 - val_accuracy: 0.9825
    Epoch 11/25
    469/469 - 3s - loss: 0.0216 - accuracy: 0.9926 - val_loss: 0.0743 - val_accuracy: 0.9824
    Epoch 12/25
    469/469 - 3s - loss: 0.0232 - accuracy: 0.9923 - val_loss: 0.0646 - val_accuracy: 0.9839
    Epoch 13/25
    469/469 - 3s - loss: 0.0228 - accuracy: 0.9924 - val_loss: 0.0720 - val_accuracy: 0.9808
    Epoch 14/25
    469/469 - 3s - loss: 0.0218 - accuracy: 0.9929 - val_loss: 0.0627 - val_accuracy: 0.9835
    Epoch 15/25
    469/469 - 3s - loss: 0.0170 - accuracy: 0.9941 - val_loss: 0.0662 - val_accuracy: 0.9833
    Epoch 16/25
    469/469 - 3s - loss: 0.0175 - accuracy: 0.9942 - val_loss: 0.0719 - val_accuracy: 0.9832
    Epoch 17/25
    469/469 - 3s - loss: 0.0179 - accuracy: 0.9941 - val_loss: 0.0697 - val_accuracy: 0.9842
    Epoch 18/25
    469/469 - 3s - loss: 0.0162 - accuracy: 0.9945 - val_loss: 0.0673 - val_accuracy: 0.9847
    Epoch 19/25
    469/469 - 3s - loss: 0.0126 - accuracy: 0.9960 - val_loss: 0.0832 - val_accuracy: 0.9833
    Epoch 00019: early stopping





    <tensorflow.python.keras.callbacks.History at 0x7f708c5e5d90>




```python
losses = pd.DataFrame(model.history.history)
```


```python
losses.plot(figsize=(10,10))
```




    <matplotlib.axes._subplots.AxesSubplot at 0x7f70a9f61690>



<img src="/assets/img/digit-recg2.png" alt="digit_rec_output2" width="75%">


From the graph of losses and accuracy, we can see that model behaves as well on the testing set as it does on the training set. Meaning the model is well-fit. 


```python
print("Loss:", model.evaluate(X_test,y_test_prepd, verbose=0)[0])
print("Accuracy:", model.evaluate(X_test,y_test_prepd, verbose=0)[1])
```

    Loss: 0.08324258029460907
    Accuracy: 0.983299970626831



```python
from sklearn.metrics import classification_report, confusion_matrix, multilabel_confusion_matrix
```


```python
predicted = np.argmax(model.predict(X_test), axis=-1)
```


```python
confusion_matrices=multilabel_confusion_matrix(y_test, predicted)
```


```python
for i in range(10):
    print(f"confusion matrix for {i}:\n {confusion_matrices[i]}\n")
```

    confusion matrix for 0:
     [[9007   13]
     [   5  975]]
    
    confusion matrix for 1:
     [[8851   14]
     [   3 1132]]
    
    confusion matrix for 2:
     [[8946   22]
     [  14 1018]]
    
    confusion matrix for 3:
     [[8970   20]
     [  20  990]]
    
    confusion matrix for 4:
     [[8998   20]
     [  19  963]]
    
    confusion matrix for 5:
     [[9101    7]
     [  24  868]]
    
    confusion matrix for 6:
     [[9026   16]
     [  11  947]]
    
    confusion matrix for 7:
     [[8952   20]
     [  18 1010]]
    
    confusion matrix for 8:
     [[9018    8]
     [  31  943]]
    
    confusion matrix for 9:
     [[8964   27]
     [  22  987]]
    



```python
print(classification_report(y_test, predicted))
```

                  precision    recall  f1-score   support
    
               0       0.99      0.99      0.99       980
               1       0.99      1.00      0.99      1135
               2       0.98      0.99      0.98      1032
               3       0.98      0.98      0.98      1010
               4       0.98      0.98      0.98       982
               5       0.99      0.97      0.98       892
               6       0.98      0.99      0.99       958
               7       0.98      0.98      0.98      1028
               8       0.99      0.97      0.98       974
               9       0.97      0.98      0.98      1009
    
        accuracy                           0.98     10000
       macro avg       0.98      0.98      0.98     10000
    weighted avg       0.98      0.98      0.98     10000
    


**predicting the label of 10 random samples and comparing with the true label**


```python
for i in range(20):
    num = random.randint(0,len(X_test))
    image = X_test[num].reshape(-1,784)
    pred = np.argmax(model.predict(image), axis=-1)
    print(f"predicted digit: {pred[0]}")
    print(f"true digit: {y_test.loc[num][0]}\n")
    if pred[0]!=y_test.loc[num][0]:
        print("-------------------\ngot this one wrong!\n-------------------\n")
```

    predicted digit: 0
    true digit: 0
    
    predicted digit: 8
    true digit: 8
    
    predicted digit: 1
    true digit: 1
    
    predicted digit: 5
    true digit: 5
    
    predicted digit: 0
    true digit: 0
    
    predicted digit: 2
    true digit: 2
    
    predicted digit: 0
    true digit: 0
    
    predicted digit: 8
    true digit: 8
    
    predicted digit: 0
    true digit: 0
    
    predicted digit: 6
    true digit: 6
    
    predicted digit: 3
    true digit: 3
    
    predicted digit: 3
    true digit: 3
    
    predicted digit: 5
    true digit: 5
    
    predicted digit: 5
    true digit: 5
    
    predicted digit: 0
    true digit: 0
    
    predicted digit: 7
    true digit: 7
    
    predicted digit: 8
    true digit: 8
    
    predicted digit: 9
    true digit: 9
    
    predicted digit: 6
    true digit: 6
    
    predicted digit: 5
    true digit: 5
    

