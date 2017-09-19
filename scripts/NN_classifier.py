import pandas as pd
import numpy as np
import os
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

train0 = pd.read_csv('../input/train.csv')
test0 = pd.read_csv('../input/test.csv')

n_train = train0.shape[0]
n_test = test0.shape[0]

## FEATURE ENGINEERING ################################################################################################
# a map of more aggregated titles
Title_Dictionary = {
    "Capt": "Officer",
    "Col": "Officer",
    "Major": "Officer",
    "Jonkheer": "Royalty",
    "Don": "Royalty",
    "Sir": "Royalty",
    "Dr": "Officer",
    "Rev": "Officer",
    "the Countess": "Royalty",
    "Dona": "Royalty",
    "Mme": "Mrs",
    "Mlle": "Miss",
    "Ms": "Mrs",
    "Mr": "Mr",
    "Mrs": "Mrs",
    "Miss": "Miss",
    "Master": "Master",
    "Lady": "Royalty"
}

titles = set()
for _, title in Title_Dictionary.items():
    titles.add(title)

train = train0.copy()
test = test0.copy()

full_data = [train, test]

# Feature that tells whether a passenger had a cabin on the Titanic
train['Has_Cabin'] = train["Cabin"].apply(lambda x: 0 if type(x) == float else 1)
test['Has_Cabin'] = test["Cabin"].apply(lambda x: 0 if type(x) == float else 1)

# Create new feature FamilySize as a combination of SibSp and Parch
for dataset in full_data:
    dataset['FamilySize'] = dataset['SibSp'] + dataset['Parch'] + 1

# Create new feature IsAlone from FamilySize
for dataset in full_data:
    dataset['IsAlone'] = 0
    dataset.loc[dataset['FamilySize'] == 1, 'IsAlone'] = 1

# Remove all NULLS in the Embarked column (fill with most common category, which is 'S')
for dataset in full_data:
    dataset['Embarked'] = dataset['Embarked'].fillna('S')

# Remove all NULLS in the Fare column and create a new feature CategoricalFare
for dataset in full_data:
    dataset['Fare'] = dataset['Fare'].fillna(train['Fare'].median())

# create new feature title, extracted from full name, and map to one of 6 title categories
for dataset in full_data:
    dataset['Title'] = dataset['Name'].map(lambda name: name.split(',')[1].split('.')[0].strip())
    # we map each title
    dataset['Title'] = dataset.Title.map(Title_Dictionary)

# map categorical features to numerical (consider creating dummies for non ordinal!)
for dataset in full_data:
    # Mapping Sex
    # dataset['Sex'] = dataset['Sex'].map( {'female': 0, 'male': 1} ).astype(int)
    dataset['Sex'] = dataset['Sex'].astype('category').cat.codes
    # dataset['Embarked'] = dataset['Embarked'].astype('category').cat.codes

# For the age we do sth a bit more elaborate. Instead of using the median of the entire dataset
# we calculate the median age for a finer grouping, namely by (Sex, Pclass, Title)
# We then fill all the null age values with the median corresponding to (Sex, Pclass, Title) of that entry

# calculate median of ENTIRE dataset (including test)
combined = train.append(test)
combined.reset_index(inplace=True)  # probably not necessary, but do it anyway
combined.drop('index', inplace=True, axis=1)  # probably not necessary, but do it anyway
age_medians = combined[['Sex', 'Pclass', 'Title', 'Age']].groupby(['Sex', 'Pclass', 'Title']).median()

def fill_age(row):
    return age_medians.loc[(row['Sex'], row['Pclass'], row['Title']), 'Age']

for dataset in full_data:
    age_msk = dataset['Age'].isnull()  # select entries with missing age
    dataset.loc[age_msk, 'Age'] = dataset.loc[age_msk].apply(fill_age, axis=1)

train = pd.get_dummies(train, columns=['Embarked'])
test = pd.get_dummies(test, columns=['Embarked'])

train = pd.get_dummies(train, columns=['Title'])
test = pd.get_dummies(test, columns=['Title'])
########################################################################################################################

## These features and parameters gave me a LB score of 0.77990, which is not great unfortunately...
wanted_features = ['Pclass', 'Sex', 'Age',
                   'Fare', 'Has_Cabin',
                   'FamilySize',
                   'Embarked_Q', 'Embarked_S',
                   'Title_Miss', 'Title_Mr', 'Title_Mrs', 'Title_Officer', 'Title_Royalty']


### NN parameters

epochs = 800
batch_size = 50
optim = 'rmsprop'
dim1 = 50
dim2 = 50
dim3 = 10

# rg = 1e-3
# init = initializers.RandomUniform(minval=-rg, maxval=rg)
init_W = initializers.glorot_normal(seed)
init_b = initializers.Zeros()
# reg = None
# reg_val = 0.01
reg_val = 0.2
# reg = regularizers.l2(reg_val)
reg = None
input_dim = None


X_train = train[wanted_features].values
X_test = test[wanted_features].values

Y_train = train['Survived'].values

n_features = X_train.shape[1]

## Scale input and test parameters
scaler = StandardScaler()
scaler.fit(X_train)

X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

model = Sequential()

model.add(Dropout(reg_val, input_shape=(n_features,)))
model.add(Dense(dim1,
                activation='relu',
                kernel_initializer=init_W,
                bias_initializer=init_b,
                kernel_regularizer=reg,
                bias_regularizer=reg))

model.add(Dropout(reg_val))
model.add(Dense(dim2, activation='relu',
                kernel_initializer=init_W,
                bias_initializer=init_b,
                kernel_regularizer=reg,
                bias_regularizer=reg))

model.add(Dropout(reg_val))
model.add(Dense(dim3, activation='relu',
                kernel_initializer=init_W,
                bias_initializer=init_b,
                kernel_regularizer=reg,
                bias_regularizer=reg))

model.add(Dropout(reg_val))
model.add(Dense(1, activation='sigmoid',
                kernel_initializer=init_W,
                bias_initializer=init_b,
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

file_name = 'predictions_MLP_Keras_Dims' + str(dim1) + '_' + str(dim2) + '_' + str(dim3) + '_RegDropout' + str(reg_val) + '_Optim' + \
            optim.capitalize() + '_Epochs' + str(epochs) + '_BatchSize' + str(batch_size) + '.csv'
test['Survived'] = Y_final.astype('int64')
df_out = test[['PassengerId', 'Survived']]

df_out.to_csv(os.path.join('../output', file_name), index=False)
