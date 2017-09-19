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
    dataset['Embarked'] = dataset['Embarked'].astype("category").cat.codes

# create dummies for Embarked (unordered categorical feature)
# why does this not work if done in the above for loop (it has no effect)
# train = pd.get_dummies(train, columns=['Embarked'])
# test = pd.get_dummies(test, columns=['Embarked'])

drop_elements = ['Name', 'Ticket', 'Cabin', 'SibSp', 'Parch']
train = train.drop(drop_elements, axis = 1)
test = test.drop(drop_elements, axis = 1)

print('cleaned dataset')
print(train.head(10))

#wanted_features = ['Pclass', 'Sex', 'Age', 'Fare', 'Has_Cabin', 'FamilySize', 'IsAlone',\
                   #'Embarked_C', 'Embarked_Q', 'Embarked_S']
wanted_features = ['Pclass', 'Sex', 'Age', 'Fare', 'Has_Cabin', 'IsAlone', 'Embarked']
# wanted_features = ['Sex', 'Age', 'Fare', 'Has_Cabin', 'FamilySize']
# wanted_features = ['Pclass', 'Sex', 'Age', 'Fare', 'Has_Cabin', 'IsAlone', 'Embarked_C', 'Embarked_Q', 'Embarked_S']

X_train = train[wanted_features].values
X_test = test[wanted_features].values

Y_train = train['Survived'].values

n_features = X_train.shape[1]

reginfo = namedtuple('reginfo',['type', 'value'])

## model generator
def build_model(hidden_dims=[10], reginfos=[reginfo('dropout', 0)], optim='adam', verbose=True):
    n_hidden_layers = len(hidden_dims)

    if reginfos is None:
        reginfos = (n_hidden_layers + 1)*[None]

    if len(reginfos) != n_hidden_layers + 1:
        raise ValueError('# of regularization infos must be # hidden layers + 1')

    # all_dims = [n_features] + hidden_dims + [1]

    # rg = 1e-3
    # init = initializers.RandomUniform(minval=-rg, maxval=rg)
    init_W = initializers.glorot_normal(seed)
    init_b = initializers.Zeros
    model = Sequential()

    # rg = math.sqrt(6/(n_features + hidden_dims[0]))
    reg = None
    input_dim = n_features
    if reginfos[0] is not None:
        if reginfos[0].type == 'dropout':
            model.add(Dropout(reginfos[0].value, input_shape=(n_features,)))
            input_dim = None
        elif reginfos[0].type == 'l2':
            reg = regularizers.l2(reginfos[0].value)

    model.add(Dense(hidden_dims[0], input_dim=input_dim,
                    activation='relu',
                    kernel_initializer=init_W,
                    bias_initializer=init_b,
                    kernel_regularizer=reg,
                    bias_regularizer=reg))

    for i in range(1, n_hidden_layers):
        # rg = math.sqrt(6 / (hidden_dims[i-1] + hidden_dims[i]))
        # init = initializers.RandomUniform(minval=-rg, maxval=rg)

        if reginfos[i] is not None:
            if reginfos[i].type == 'dropout':
                model.add(Dropout(reginfos[i].value))
                reg = None
            elif reginfos[i].type == 'l2':
                reg = regularizers.l2(reginfos[i].value)

        model.add(Dense(hidden_dims[i], activation='relu',
                        kernel_initializer=init_W,
                        bias_initializer=init_b,
                        kernel_regularizer=reg,
                        bias_regularizer=reg))

    reg = None
    if reginfos[n_hidden_layers] is not None:
        if reginfos[n_hidden_layers].type == 'dropout':
            model.add(Dropout(reginfos[n_hidden_layers].value))
        elif reginfos[n_hidden_layers].type == 'l2':
            reg = regularizers.l2(reginfos[n_hidden_layers].value)

    model.add(Dense(1, activation='sigmoid',
                    kernel_initializer=init_W,
                    bias_initializer=init_b,
                    kernel_regularizer=reg,
                    bias_regularizer=reg))

    if verbose:
        print(model.summary())
    model.compile(loss='binary_crossentropy', optimizer=optim, metrics=['accuracy'])
    return model


## build pipeline
epochs = 400
batch_size = 10
optim = 'adam'
hidden_dims = [7, 3]
# reginfos = 3*[reginfo('dropout', 0.25)]
# reginfos = 3*[reginfo('l2', 0.02)]
# reginfos = None
reginfos = 3*[reginfo('l2', 0.001)]
# reginfos = 3*[reginfo('dropout', 0.25)]

sk_params = {'epochs': epochs,
             'batch_size': batch_size,
             'optim': optim,
             'hidden_dims': hidden_dims,
             'reginfos': reginfos}

estimators = []
estimators.append(('standardize', StandardScaler()))
estimators.append(('mlp',
                   KerasClassifier(build_fn=build_model, verbose=1, **sk_params)))

pipeline = Pipeline(estimators)
kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
results = cross_val_score(pipeline, X_train, Y_train, cv=kfold)

print('\n\n')
print('features:')
print(wanted_features)
print('parameters:')
print(sk_params)
print("Standardized: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))
