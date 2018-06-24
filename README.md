## Automate the automated...

Create a competent model, without ML experience. 
Any dataset with a target to predict. Continuous, categorical, features of text 
numbers, dates.. doesn't matter; `Fresh` will figure it out. 

```python
import pandas as pd
from fresh import Model

df = pd.read_csv('mydata.csv')

model = Model(target=df['my-target'])
model.fit(df)

X = pd.read_csv('data_without_answers.csv')
predictions = model.predict(X)
```


