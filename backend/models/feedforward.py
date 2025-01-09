import pickle
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import pandas as pd
import numpy as np
import sklearn
from sklearn.model_selection import train_test_split
device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
train_dataset=pd.read_csv('./datasets/breast+cancer+wisconsin+diagnostic/wdbc.csv',header=None)
column_names= ['ID', 'Diagnosis'] + [
    'Radius (mean)', 'Texture (mean)', 'Perimeter (mean)', 'Area (mean)', 
    'Smoothness (mean)', 'Compactness (mean)', 'Concavity (mean)', 'Concave points (mean)', 
    'Symmetry (mean)', 'Fractal dimension (mean)', 
    'Radius (SE)', 'Texture (SE)', 'Perimeter (SE)', 'Area (SE)', 
    'Smoothness (SE)', 'Compactness (SE)', 'Concavity (SE)', 'Concave points (SE)', 
    'Symmetry (SE)', 'Fractal dimension (SE)', 
    'Radius (worst)', 'Texture (worst)', 'Perimeter (worst)', 'Area (worst)', 
    'Smoothness (worst)', 'Compactness (worst)', 'Concavity (worst)', 'Concave points (worst)', 
    'Symmetry (worst)', 'Fractal dimension (worst)'
]
train_dataset.columns=column_names
from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
label=train_dataset['Diagnosis']
features=train_dataset.drop(columns=['Diagnosis'])
normalized_features=scaler.fit_transform(features)
normalized_df=pd.DataFrame(normalized_features,columns=features.columns)
combined=pd.concat([normalized_df,label],axis=1)
from scipy.stats import zscore
z_scores=np.abs(zscore(combined.drop(columns=['Diagnosis'])))
threshold=3
outliers=(z_scores>threshold).any(axis=1)
combined_cleaned=combined[~outliers]

features_cleaned = combined_cleaned.drop(columns=['Diagnosis'])
label_cleaned=combined_cleaned['Diagnosis']

Q1= normalized_df.quantile(0.25)
Q3= normalized_df.quantile(0.75)

IQR=Q3-Q1
lower_bound=Q1-1.5*IQR
upper_bound=Q3+1.5*IQR
outliers_iqr = (normalized_df < lower_bound) | (normalized_df > upper_bound)
df_cleaned_iqr = combined[~outliers_iqr.any(axis=1)]
final_df = df_cleaned_iqr.copy()
final_df[features.columns] = scaler.transform(final_df[features.columns])

from sklearn.preprocessing import LabelEncoder
encoder=LabelEncoder()
encoded_label=encoder.fit_transform(label_cleaned)
X=features_cleaned
y=encoded_label



x_train,x_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=42)
x_train_tensor=torch.tensor(x_train.to_numpy(),dtype=torch.float32).to(device)
x_test_tensor=torch.tensor(x_test.to_numpy(),dtype=torch.float32).to(device)
y_train_tensor=torch.tensor(y_train,dtype=torch.float32).to(device).view(-1, 1)
y_test_tensor=torch.tensor(y_test,dtype=torch.float32).to(device).view(-1,1)

from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN,self).__init__()
        self.fc1=nn.Linear(31,64)
        self.fc2=nn.Linear(64,32)
        self.fc3=nn.Linear(32,1)
    def forward(self,x):
        x=torch.relu(self.fc1(x))
        x=torch.relu(self.fc2(x))
        x=torch.relu(self.fc3(x))
        return x
model=SimpleNN()
criterion=nn.MSELoss()
optimizer=torch.optim.SGD(model.parameters(),lr=0.01)
model.to(device)
for epoch in range(100):
    model.train()
    optimizer.zero_grad()
    features=model(x_train_tensor)
    loss=criterion(features,y_train_tensor)
    loss.backward()
    optimizer.step()

with torch.no_grad():
    model.eval()
    train_features=model(x_train_tensor).cpu().numpy()
    test_features=model(x_test_tensor).cpu().numpy()


# svm=SVC(kernel='linear') 
# svm.fit(train_features,y_train)   
# preds=svm.predict(test_features)
# accuracy=accuracy_score(y_test,preds)
# print(f"the svm accuracy is:{accuracy*100:.2f}%")

from sklearn.neighbors import KNeighborsClassifier
KNN=KNeighborsClassifier(n_neighbors=9)
KNN.fit(train_features,y_train)
preds_k=KNN.predict(test_features)
accuracyk=accuracy_score(y_test,preds_k)
print(f"the KNN accuracy is:{accuracyk*100:.2f}%")
with open('./models/trained_model.pkl', 'wb') as f:
    pickle.dump(model, f)
