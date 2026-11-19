#Step 2 : Data Preprocessing

import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler,OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from data_ingestion import data_ingestion

def data_preprocessing(df):
    ''' 
    Dataset : Supply Chain and Logistics
    
    Target Column (y): Product_Demand
    
    Steps:
    1. Drop duplicate records.
    2. Feature Engineering (Optional).
    3. Separate independent (X) and dependent (y) columns.
    4. Split data into training and testing sets.
    5. Segregate numerical and categorical columns.
    6. Apply imputation and scaling to numerical columns.
    7. Apply imputation and encoding to categorical columns.
    8. Combine pipelines using ColumnTransformer.
    9. Fit and transform training data; transform testing data.
    10. Return preprocessed datasets.
    
    '''

# Step 1 : Drop duplicate records

    # Step 1: Drop duplicate records

    df = df.drop_duplicates()

    print("Shape after removing duplicates:", df.shape)

# Step 2 : Feature Engineering(Optional)
    
# Step 3 : Separate Independent(X) and Dependent(y) columns

    X = df.drop(columns = ['Product_Demand'])
    y = df['Product_Demand']

# Step 4 : Split data into training and testing datasets

    X_train, X_test,y_train, y_test = train_test_split(X,y,
                                                      test_size = 0.2,
                                                      random_state = 1)
    print("X_train before transformation:", X_train.shape)
    print("X_test before transformation:", X_test.shape)
# Step 5 : Segregate numerical and categorial columns
    
    numerical_columns = X.select_dtypes(include ='number').columns
    categorical_columns = X.select_dtypes(exclude = 'number').columns

# Step 6 : Apply imputation and scaling to numerical columns

    numerical_pipeline = Pipeline(
        [ ('Imputation', SimpleImputer(strategy = 'median')),
          ('Scaling', MinMaxScaler())
        ])

# Step 7 : Apply imputation and encoding to categorical columns

    categorical_pipeline = Pipeline(
        [ ('Imputation', SimpleImputer(strategy = 'most_frequent')),
          ('Encoding', OneHotEncoder(handle_unknown = 'ignore'))
        ])
    for column in categorical_columns:
        print(column, ":", df[column].nunique())

# Step 8 : Combine pipelines using ColumnTransformer
 
    transformer = ColumnTransformer([
        ('numerical_pipeline',numerical_pipeline,numerical_columns),
        ('categorical_pipeline',categorical_pipeline,categorical_columns)
        ])
    
# Step 9 : Fit and transform training testing data

    X_train = transformer.fit_transform(X_train)
    X_test = transformer.transform(X_test)

# Step 10 : Return preprocessed data

    print("X_train Shape:", X_train.shape)
    print("X_test Shape:", X_test.shape)
    print("y_train Shape:", y_train.shape)
    print("y_test Shape:", y_test.shape)


    return X_train,X_test,y_train,y_test

df = data_ingestion()
X_train, X_test, y_train, y_test = data_preprocessing(df)
