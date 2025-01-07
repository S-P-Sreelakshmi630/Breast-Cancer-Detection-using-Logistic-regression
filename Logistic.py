import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
from scipy import stats
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
warnings.filterwarnings('ignore')



data=pd.read_csv('/content/drive/MyDrive/Data/data.csv')

data.info()

data.describe()

data.head()

data.tail()

data.shape

data.isnull().sum

data[data.isnull().any(axis=1)]


print("Number of missing values before filling:", data.isnull().sum().sum())

data = data.fillna(data.mean())

data['diagnosis'].unique()

label_encoder = LabelEncoder()
data['diagnosis'] = label_encoder.fit_transform(data['diagnosis'])

y1 = data['diagnosis']
x1 = data.drop(['id', 'diagnosis', 'Unnamed: 32'], axis=1)
x1.columns

print("Number of samples after filling missing values:", len(data))

diagnosis_counts = data['diagnosis'].value_counts()
plt.pie(diagnosis_counts, labels=diagnosis_counts.index, autopct='%1.1f%%', startangle=90)
plt.title('Pie Chart of Diagnosis')
plt.show()

sns.countplot(x='diagnosis', data=data)
plt.show()

#Outliers detection and removal using Z- Score

data = data.drop('Unnamed: 32', axis=1)

# Extract numerical columns for outlier detection
numerical_cols = data.select_dtypes(include=['float64']).columns

# Calculate Z-scores for the numerical columns
z_scores = stats.zscore(data[numerical_cols])

# Set a threshold
threshold = 3

# Identify outliers
outliers = (abs(z_scores) > threshold).all(axis=1)

# Remove outliers from the DataFrame
df_no_outliers = data[~outliers]

# Show information about removed outliers
removed_outliers_count = sum(outliers)
print(f"Number of removed outliers: {removed_outliers_count}")


#correlation checking
df = data.drop('radius_se', axis=1)

# Calculate correlation matrix
correlation_matrix = df.corr()

# Display the correlation matrix
print(correlation_matrix)



# Visualize pairplots
sns.pairplot(data, x_vars=['radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean'], y_vars=['diagnosis'], hue='diagnosis')
plt.show()


sns.pairplot(data, x_vars=['smoothness_mean', 'symmetry_mean', 'concavity_mean', 'concave points_mean'], y_vars=['diagnosis'])
plt.show()

print("Number of samples before dropping missing values:", len(data))

x_train,x_test,y_train,y_test=train_test_split(x1,y1,train_size=0.7,random_state=123)

x_train.shape,x_test.shape,y_train.shape,y_test.shape

model=LogisticRegression(max_iter=5000)

model.fit(x_train,y_train)

y_pred=model.predict(x_test)

print(data['diagnosis'].unique())

#RandomForest Classifier
data['diagnosis'] = data['diagnosis'].map({'M': 1, 'B': 0})

feature_cols = ['radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean', 'smoothness_mean', 'compactness_mean', 'concavity_mean', 'concave points_mean', 'symmetry_mean', 'fractal_dimension_mean', 'texture_se', 'perimeter_se', 'area_se', 'smoothness_se', 'compactness_se', 'concavity_se', 'concave points_se', 'symmetry_se', 'fractal_dimension_se', 'radius_worst', 'texture_worst', 'perimeter_worst', 'area_worst', 'smoothness_worst', 'compactness_worst', 'concavity_worst', 'concave points_worst', 'symmetry_worst', 'fractal_dimension_worst']

x_train,x_test,y_train,y_test=train_test_split(x1,y1,train_size=0.7,random_state=2529)

rf_model = RandomForestClassifier(random_state=2529)

# Train the model
rf_model.fit(x_train, y_train)

# Predictions on the test set
y_pred = rf_model.predict(x_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
classification_rep = classification_report(y_test, y_pred)

# Display results
print(f"Accuracy: {accuracy}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"Confusion Matrix:\n{conf_matrix}")
print(f"Classification Report:\n{classification_rep}")


confusion_matrix(y_test,y_pred)
accuracy_score(y_test,y_pred)
print(classification_report(y_test,y_pred))
recall_score(y_test,y_pred)
precision_score(y_test,y_pred)
