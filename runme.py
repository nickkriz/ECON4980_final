import json
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn import linear_model
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn import metrics

business_name = []
business_stars = []
business_address = []
business_state = []
business_openess = []
business_categories = []
business_hours = []
business_postal = []
business_weekend = []

with open('business_sample.json', encoding="utf8") as f:
	for line in f:
		json_line = json.loads(line)
		business_name.append(json_line['name'])
		business_address.append(json_line['address'])
		business_stars.append(json_line['stars'])
		business_state.append(json_line['state'])
		business_openess.append(json_line['is_open'])
		business_categories.append(json_line['categories'])
		business_hours.append(json_line['hours'])
		business_postal.append(json_line['postal_code'])


df = pd.DataFrame(data={'name': business_name, 'address': business_address, 'postal': business_postal, 'state': business_state, 'is_open': business_openess, 'categories':business_categories, 'hours': business_hours, 'stars': business_stars})

df.drop(df.index[359:159999],0,inplace=True)

# print(df)

# categories = df.iloc[:, 4].values
# print(categories)

df['state'] = pd.Categorical(df['state'])
df['state code'] = df['state'].cat.codes

df['categories'] = pd.Categorical(df['categories'])
df['category type'] = df['categories'].cat.codes

del df['state']
del df['categories']
del df['name']
del df['address']

# df['hours'] = df['hours'].map(lambda x: str(x)[:-150])

for key in list(df['hours'].keys()):
	if key == 'Saturday':
		business_weekend.append(1)
	else:
		business_weekend.append(0)

df['weekend'] = business_weekend

del df['hours']

df['stars'] = df['stars'].astype(int)

target = df.iloc[:,2].values
# print(target)
data = df.iloc[:,[1,3,4,5]].values
# print(data)

# machine = linear_model.LinearRegression()
# machine = linear_model.LogisticRegression()
# machine.fit(data, target)

# for column in df['hours']:
# 	df['weekend'] = df['hours'].str.find('Saturday')		

# df['weekend'] = df.hours.str.contains("")

# for x in business_hours:
# 	business_weekend = x.find_all("Saturday")

# weekend = df['hours'].isin(['Saturday']).any().any()

# print(df)

# labelencoder_categories = LabelEncoder()
# categories[0] = labelencoder_categories.fit_transform(categories[0])
# categories[1] = labelencoder_categories.fit_transform(categories[1])

# print(categories)

# Y = df.iloc[:, -1].values
# print(Y)

business_name2 = []
business_stars2 = []
business_address2 = []
business_state2 = []
business_openess2 = []
business_categories2 = []
business_hours2 = []
business_postal2 = []
business_weekend2 = []

with open('business_no_stars_review.json', encoding="utf8") as f:
	for line in f:
		json_line = json.loads(line)
		business_name2.append(json_line['name'])
		business_address2.append(json_line['address'])
		# business_stars.append(json_line['stars'])
		business_state2.append(json_line['state'])
		business_openess2.append(json_line['is_open'])
		business_categories2.append(json_line['categories'])
		business_hours2.append(json_line['hours'])
		business_postal2.append(json_line['postal_code'])


df2 = pd.DataFrame(data={'name': business_name2, 'address': business_address2, 'postal': business_postal2, 'state': business_state2, 'is_open': business_openess2, 'categories':business_categories2, 'hours': business_hours2})

df2['state'] = pd.Categorical(df2['state'])
df2['state code'] = df2['state'].cat.codes

df2['categories'] = pd.Categorical(df2['categories'])
df2['category type'] = df2['categories'].cat.codes

for key in list(df2['hours'].keys()):
	if key == 'Saturday':
		business_weekend2.append(1)
	else:
		business_weekend2.append(0)

df2['weekend'] = business_weekend2

del df2['state']
del df2['categories']
del df2['name']
del df2['hours']
del df2['address']

# print(df2)

new_data = df2.iloc[:,[1,2,3,4]]

# results = machine.predict(new_data)

# print(results)
kfold_object = KFold(n_splits = 4)
kfold_object.get_n_splits(data)

print(kfold_object)

i = 0 
for training_index, test_index in kfold_object.split(data):
	i = i+1
	print("round: ", i)
	print("training: ", training_index)
	print("test: ", test_index)
	data_training, data_test = data[training_index], data[test_index]
	target_training, target_test = target[training_index], target[test_index]
	machine = linear_model.LogisticRegression()
	machine.fit(data_training, target_training)
	results = machine.predict(new_data)
	# print(metrics.r2_score(target_test, results))
	print("With Logistic Model: ")
	print(results)
	print(metrics.accuracy_score(target_test, results))
	print(metrics.confusion_matrix(target_test, results))

for training_index, test_index in kfold_object.split(data):
	i = i+1
	print("round: ", i)
	print("training: ", training_index)
	print("test: ", test_index)
	data_training, data_test = data[training_index], data[test_index]
	target_training, target_test = target[training_index], target[test_index]
	machine = linear_model.LinearRegression()
	machine.fit(data_training, target_training)
	results = machine.predict(new_data)
	# print(metrics.r2_score(target_test, results))
	print("With Linear Model: ")
	print(results)
	print(metrics.accuracy_score(target_test, results))
	print(metrics.confusion_matrix(target_test, results))
