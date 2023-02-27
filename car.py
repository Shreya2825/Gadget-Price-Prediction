import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import make_pipeline
from sklearn.compose import make_column_transformer
car=pd.read_csv("data.csv")
x=car.drop(columns='Price')
y=car['Price']

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test=train_test_split(x,y,test_size=0.2)

ohe= OneHotEncoder()
print(ohe.fit(x[['name','company','fuel_type']]))
print(OneHotEncoder())
column_trans=make_column_transformer((OneHotEncoder(),['name','company','fuel_type']),remainder='passthrough')
lr=LinearRegression()
pipe=make_pipeline(column_trans,lr)
pipe.fit(x_train,y_train)
print(ohe.categories)




