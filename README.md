## Automate the automated...

Create a competent model, without ML experience.   
Any dataset with a target to predict; continuous, categorical, with features of text, 
numbers, dates.. doesn't matter; `Fresh` will figure it out. 

```python
import pandas as pd
from fresh import Model

df = pd.read_csv('mydata.csv')
target = df['my-target']
del df['my-target']

model = Model()
model.fit(df, target)

X = pd.read_csv('new_data_without_answers.csv')
predictions = model.predict(X)
```


