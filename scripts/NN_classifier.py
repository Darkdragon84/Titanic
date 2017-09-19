import pandas as pd
import numpy as np
from collections import namedtuple
import math
from keras import optimizers, initializers, regularizers
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.utils import np_utils  # for transforming data later
from keras.wrappers.scikit_learn import KerasClassifier

from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline

seed = 42  # for reproducability
np.random.seed(seed)

train = pd.read_csv( '../input/train.csv')
test = pd.read_csv('../input/test.csv')

n_train = train.shape[0]
n_test = test.shape[0]

print('bare data')
print(train.head(5))

## FEATURE ENGINEERING

full_data = [train, test]

# Feature that tells whether a passenger had a cabin on the Titanic
train['Has_Cabin'] = train["Cabin"].apply(lambda x: 0 if type(x) == float else 1)
test['Has_Cabin'] = test["Cabin"].apply(lambda x: 0 if type(x) == float else 1)

# some of the feature engineering steps taken from Sina
# Create new feature FamilySize as a combination of SibSp and Parch
for dataset in full_data:
    dataset['FamilySize'] = dataset['SibSp'] + dataset['Parch'] + 1
# Create new feature IsAlone from FamilySize
for dataset in full_data:
    dataset['IsAlone'] = 0
    dataset.loc[dataset['FamilySize'] == 1, 'IsAlone'] = 1
# Remove all NULLS in the Embarked column (use most common value 'S')
for dataset in full_data:
    dataset['Embarked'] = dataset['Embarked'].fillna('S')
# Remove all NULLS in the Fare column and fill with median
for dataset in full_data:
    dataset['Fare'] = dataset['Fare'].fillna(train['Fare'].median())
# Remove all NULLS in the Age column and fill random age in range[avg - std, avg + std] (questionable...)
for dataset in full_data:
    age_avg = dataset['Age'].mean()
    age_std = dataset['Age'].std()
    age_null_count = dataset['Age'].isnull().sum()  # counts the number of NaN is the Age column
    age_null_random_list = np.random.randint(age_avg - age_std, age_avg + age_std, size=age_null_count)
    # fill missing ages with random ages around mean age
    dataset.loc[dataset['Age'].isnull(), 'Age'] = age_null_random_list
    dataset['Age'] = dataset['Age'].astype(int)
# Map Sex to categorical [1,0]
for dataset in full_data:
    # Mapping Sex
    dataset['Sex'] = dataset['Sex'].astype("category").cat.codes
    # Mapping Embarked
    # dataset['Embarked'] = dataset['Embarked'].astype("category").cat.codes

# create dummies for Embarked (unordered categorical feature)
# why does this not work if done in the above for loop (it has no effect)
train = pd.get_dummies(train, columns=['Embarked'])
test = pd.get_dummies(test, columns=['Embarked'])

drop_elements = ['Name', 'Ticket', 'Cabin', 'SibSp', 'Parch']
train = train.drop(drop_elements, axis=1)
test = test.drop(drop_elements, axis=1)


wanted_features = ['Pclass', 'Sex', 'Age', 'Fare', 'Has_Cabin', 'IsAlone', 'Embarked_C', 'Embarked_Q', 'Embarked_S']

X_train = train[wanted_features].values
X_test = test[wanted_features].values

Y_train = train['Survived'].values

n_features = X_train.shape[1]

## Scale input and test parameters
scaler = StandardScaler()
scaler.fit(X_train)

X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

### NN parameters
rg = 1e-3

epochs = 400
batch_size = 10
optim = 'adam'

init = initializers.RandomUniform(minval=-rg, maxval=rg)
reg = regularizers.l2(0.002)

model = Sequential()
model.add(Dense(30, input_dim=n_features,
                activation='relu',
                kernel_initializer=init,
                bias_initializer=init,
                kernel_regularizer=reg,
                bias_regularizer=reg))

model.add(Dense(20, activation='relu',
                kernel_initializer=init,
                bias_initializer=init,
                kernel_regularizer=reg,
                bias_regularizer=reg))

model.add(Dense(1, activation='sigmoid',
                kernel_initializer=init,
                bias_initializer=init,
                kernel_regularizer=reg,
                bias_regularizer=reg))

print(model.summary())
model.compile(loss='binary_crossentropy', optimizer=optim, metrics=['accuracy'])

model.fit(X_train, Y_train, batch_size=batch_size, epochs=epochs, verbose=1)
score = model.evaluate(X_train, Y_train)

Y_final = model.predict(X_test)
surv_msk = Y_final > 0.5
Y_final = np.zeros(Y_final.shape)
Y_final[surv_msk] = 1

test['Survived'] = Y_final.astype('int64')
df_out = test[['PassengerId', 'Survived']]
df_out.head(10)
df_out.to_csv('../output/predictions_MLP_Keras.csv', index=False)
