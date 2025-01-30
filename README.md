# **Breast Cancer Classification with Neural Networks**

**Developed by:** Karan Bhosle  
**Contact:** [LinkedIn - Karan Bhosle](https://www.linkedin.com/in/karanbhosle/)  

### **Project Description:**
This project implements a neural network model to classify breast cancer data into two categories: malignant (cancerous) and benign (non-cancerous). The model uses a feedforward neural network (fully connected) to predict whether a tumor is malignant or benign based on 30 input features related to the characteristics of cell nuclei present in breast cancer biopsies. 

The dataset used in this project is the **Breast Cancer Wisconsin dataset**, which is available through the `sklearn` library. The neural network is trained using **TensorFlow** and **Keras**, and the performance is evaluated using accuracy metrics and visualizations.

### **Objective:**
- To train a neural network model to classify breast cancer tumors as malignant or benign.
- To preprocess the dataset, perform feature scaling, train a neural network, and visualize the performance over epochs.

---

### **Tools and Libraries Used:**
- **Python**: Programming language
- **pandas**: Data manipulation
- **numpy**: Numerical computations
- **matplotlib** & **seaborn**: Data visualization
- **sklearn**: Machine learning utilities, preprocessing, and evaluation
- **TensorFlow** & **Keras**: Neural network model creation and training

---

### **Data Description:**
The dataset used is the **Breast Cancer Wisconsin (Diagnostic) Dataset**, which contains 30 features per sample that describe characteristics of the cell nuclei present in breast cancer biopsies. The target label is binary:
- **0**: Benign
- **1**: Malignant

**Key Features**:
- Radius, texture, perimeter, area, smoothness, compactness, concavity, symmetry, fractal dimension of cell nuclei.

---

### **Steps Involved:**

#### **1. Import Libraries:**
All necessary libraries such as `pandas`, `numpy`, `matplotlib`, `seaborn`, and `sklearn` for preprocessing, building models, and visualization are imported. Additionally, `tensorflow` is used to build and train the neural network.

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
import warnings
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_breast_cancer
```

#### **2. Load and Prepare Data:**
The **Breast Cancer dataset** is loaded using `sklearn.datasets.load_breast_cancer`. The features are extracted and stored in a DataFrame along with the target label.

```python
breast_cancer_data = load_breast_cancer()
data_frame = pd.DataFrame(breast_cancer_data.data, columns=breast_cancer_data.feature_names)
data_frame['label'] = breast_cancer_data.target
```

#### **3. Data Exploration:**
Basic data exploration is done to check for null values, describe the dataset, and check the distribution of labels. 

```python
data_frame.info()  # Check data types and null values
data_frame.describe()  # Get basic statistics
data_frame['label'].value_counts()  # Check label distribution
```

#### **4. Feature Selection and Splitting:**
The features (X) are separated from the target variable (Y), and the dataset is split into training and testing datasets. **80%** of the data is used for training, and **20%** is used for testing.

```python
X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size=0.2, random_state=2)
```

#### **5. Data Standardization:**
StandardScaler is used to scale the features to have zero mean and unit variance, which helps improve the neural network's performance.

```python
scaler = StandardScaler()
scaler.fit(X_train)  # Fit the scaler on training data
X_train_std = scaler.transform(X_train)  # Transform training data
X_test_std = scaler.transform(X_test)  # Transform test data
```

#### **6. Neural Network Model:**
A simple **feedforward neural network** is created using **Keras**. The model consists of:
- **Input Layer**: Flattens the input data.
- **Hidden Layer**: A fully connected layer with 20 neurons and ReLU activation.
- **Output Layer**: A sigmoid activation function to predict binary outcomes (malignant/benign).

```python
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(30,)),  # Flatten input data
    keras.layers.Dense(20, activation='relu'),  # Hidden layer with 20 neurons
    keras.layers.Dense(2, activation='sigmoid')  # Output layer with 2 neurons
])
```

#### **7. Model Compilation:**
The model is compiled using **Adam** optimizer, **sparse categorical cross-entropy** loss, and **accuracy** as the evaluation metric.

```python
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
```

#### **8. Model Training:**
The model is trained on the training data for 100 epochs. Validation data is also used to track the performance during training.

```python
history = model.fit(X_train_std, Y_train, validation_split=0.1, epochs=100)
```

#### **9. Model Evaluation:**
After training, the model's performance is evaluated on the test set, and test accuracy and loss are printed.

```python
loss, accuracy = model.evaluate(X_test_std, Y_test)
```

#### **10. Visualization:**
The training and validation accuracy/loss are plotted to visualize the model's learning progress.

```python
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model Accuracy')
```

#### **11. Predictions:**
The model's predictions are made on the test data, and the predicted values are printed.

```python
Y_pred = model.predict(X_test_std)
```

---

### **Model Evaluation:**
The model performance is evaluated using the following metrics:
- **Accuracy**: Percentage of correct predictions.
- **Loss**: The loss function value showing the difference between predicted and actual values.

Additionally, visualizations such as accuracy and loss curves help to understand how well the model is learning.

---

### **Conclusion:**
This project demonstrates how to use neural networks for binary classification problems like breast cancer diagnosis. The model successfully predicts whether a tumor is malignant or benign using the **Breast Cancer Wisconsin dataset**. The neural network model achieves high accuracy, and the performance is tracked and visualized using various plots.

---

### **References:**
1. **Breast Cancer Wisconsin Dataset**: Available in `sklearn.datasets`.
2. **Keras**: High-level neural network API, running on top of TensorFlow.
3. **TensorFlow**: Framework for deep learning used for building neural networks.
