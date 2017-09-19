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

# create additional ordinal categorical age feature
n_age_bins = 5
train['CategoricalAge'], age_bins = pd.cut(train['Age'], n_age_bins, retbins=True)

train = pd.get_dummies(train, columns=['Embarked'], drop_first=True)
test = pd.get_dummies(test, columns=['Embarked'], drop_first=True)

train = pd.get_dummies(train, columns=['Title'], drop_first=True)
test = pd.get_dummies(test, columns=['Title'], drop_first=True)
########################################################################################################################

# drop_elements = ['Name', 'Ticket', 'Cabin', 'SibSp', 'Parch']
# train = train.drop(drop_elements, axis = 1)
# test = test.drop(drop_elements, axis = 1)



#wanted_features = ['Pclass', 'Sex', 'Age', 'Fare', 'Has_Cabin', 'FamilySize', 'IsAlone',\
                   #'Embarked_C', 'Embarked_Q', 'Embarked_S']
# wanted_features = ['Pclass', 'Sex', 'Age', 'Fare', 'Has_Cabin', 'IsAlone', 'FamilySize', 'Embarked']
# wanted_features = ['Pclass', 'Sex', 'Age', 'Fare', 'Has_Cabin', 'IsAlone', 'FamilySize',
#                    'Embarked_C', 'Embarked_Q', 'Embarked_S',
#                    'Title_Master', 'Title_Miss', 'Title_Mr', 'Title_Mrs', 'Title_Officer', 'Title_Royalty']
wanted_features = ['Pclass', 'Sex', 'Age',
                   'Fare', 'Has_Cabin',
                   'FamilySize',
                   'Embarked_Q', 'Embarked_S',
                   'Title_Miss', 'Title_Mr', 'Title_Mrs', 'Title_Officer', 'Title_Royalty']
# wanted_features = ['Sex', 'Age', 'Fare', 'Has_Cabin', 'FamilySize']
# wanted_features = ['Pclass', 'Sex', 'Age', 'Fare', 'Has_Cabin', 'IsAlone', 'Embarked_C', 'Embarked_Q', 'Embarked_S']

reginfo = namedtuple('reginfo',['type', 'value'])

## build pipeline
n_splits = 10
epochs = 800
batch_size = 50
optim = 'rmsprop'
hidden_dims = [50, 50, 10]
n_regs = len(hidden_dims) + 1
reginfos = n_regs*[reginfo('dropout', 0.2)]
# reginfos = n_regs*[reginfo('l2', 0.02)]
# reginfos = None
# reginfos = n_regs*[reginfo('l2', 0.01)]
# reginfos = n_regs*[reginfo('dropout', 0.25)]

sk_params = {'epochs': epochs,
             'batch_size': batch_size,
             'optim': optim,
             'hidden_dims': hidden_dims,
             'reginfos': reginfos}

X_train = train[wanted_features].values
X_test = test[wanted_features].values
Y_train = train['Survived'].values

n_features = X_train.shape[1]


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
    init_b = initializers.Zeros()
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


estimators = [
        ('standardize', StandardScaler()),
        ('mlp', KerasClassifier(build_fn=build_model, verbose=1, **sk_params))
    ]

pipeline = Pipeline(estimators)
kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
results = cross_val_score(pipeline, X_train, Y_train, cv=kfold)

print('\n\n')
print('features:')
print(wanted_features)
print('parameters:')
print(sk_params)
print('Standardized: %.2f%% (%.2f%%)' % (results.mean()*100, results.std()*100))
