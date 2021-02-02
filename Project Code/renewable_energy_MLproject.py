
# import necessary modules
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


"""## Data

# Renewable energy production data
"""

production = pd.read_csv(r"C:\Users\Administrator\Desktop\Renewable_Energy_Production_Project\time_series_60min_singleindex.csv",
                        usecols=(lambda s: s.startswith('utc') | s.startswith('DE')),
                        parse_dates=[0], index_col=0)


print(production.shape)
production.head(1)


production = production.loc[production.index.year == 2016, :]

"Drop the NA columns"

production=production.dropna(axis=1, how='all')

print(production.shape)
production.head(1)


production.info()

from sklearn.impute import SimpleImputer
imp = SimpleImputer(strategy='mean')
ProductionImputed = imp.fit_transform(production)

production = pd.DataFrame(ProductionImputed,
                              columns= production.columns,
                              index=production.index)
"There 8784 entries, which correspond to the number of hours in 2016."


" Exploratory Data Analysis on Generation data "


# create plot
plt.plot(production.index, production['DE_wind_generation_actual'])
plt.title('Actual wind generation in Germany in MW')
plt.xlim(pd.Timestamp('2016-01-01'), pd.Timestamp('2017-01-01'))
plt.ylim(0, 50000)


# create plot
plt.plot(production.index, production['DE_solar_generation_actual'], c='OrangeRed')
plt.title('Actual solar generation in Germany in MW')
plt.xlim(pd.Timestamp('2016-01-01'), pd.Timestamp('2017-01-01'))
plt.ylim(0, 35000)


production_wind_solar = production[['DE_wind_generation_actual', 'DE_solar_generation_actual']]

"""2. Weather data
"""

weather = pd.read_csv(r"C:\Users\Administrator\Desktop\Renewable_Energy_Production_Project/weather_data_GER_2016.csv",
                     parse_dates=[0], index_col=0)

print(weather.shape)

weather.head()
weather.info()


weather_by_day = weather.groupby(weather.index).mean()

print(weather_by_day.shape)


" Exploratory Data Analysis on weather data"

# create plot
plt.plot(weather_by_day.index, weather_by_day['v1'])
plt.title('Wind velocity 2m above displacement height (m/s)')
plt.xlim(pd.Timestamp('2016-01-01'), pd.Timestamp('2017-01-01'))
plt.ylim(0, 9)

# create plot
plt.plot(weather_by_day.index, weather_by_day['v2'])
plt.title('Wind velocity 10m above displacement height (m/s)')
plt.xlim(pd.Timestamp('2016-01-01'), pd.Timestamp('2017-01-01'))
plt.ylim(0, 12)

plt.plot(weather_by_day.index, weather_by_day['v_50m'])
plt.title('Wind velocity 50m above displacement height (m/s)')
plt.xlim(pd.Timestamp('2016-01-01'), pd.Timestamp('2017-01-01'))
plt.ylim(0, 15)


# create plot
plt.plot(weather_by_day.index, weather_by_day['SWGDN'], c='OrangeRed')
plt.title('Ground horizontal radiation (W/m²)')
plt.xlim(pd.Timestamp('2016-01-01'), pd.Timestamp('2017-01-01'))
plt.ylim(0, 1000)


plt.plot(weather_by_day.index, weather_by_day['SWTDN'], c='OrangeRed')
plt.title('Total top of atmosphere horizontal radiation (W/m²)')
plt.xlim(pd.Timestamp('2016-01-01'), pd.Timestamp('2017-01-01'))
plt.ylim(0, 1000)


"Let's convert the kelvin into Celsius units for the temperature."

weather_by_day['T (C)'] = weather_by_day['T'] - 273.15

"EDA on Temperature Data"

# create plot
plt.plot(weather_by_day.index, weather_by_day['T (C)'], c='OrangeRed')
plt.title('Temperature (ºC)')
plt.xlim(pd.Timestamp('2016-01-01'), pd.Timestamp('2017-01-01'))
plt.ylim(-10, 30)


"EDA on Air Data"

# create plot
plt.plot(weather_by_day.index, weather_by_day['rho'])
plt.title('Air density at the surface (kg/m³)')
plt.xlim(pd.Timestamp('2016-01-01'), pd.Timestamp('2017-01-01'))

# create plot
plt.plot(weather_by_day.index, weather_by_day['p'])
plt.title('Air pressure at the surface (Pa)')
plt.xlim(pd.Timestamp('2016-01-01'), pd.Timestamp('2017-01-01'))

" 2. Merge DataFrames "

# merge production_wind_solar and weather_by_day DataFrames
combined = pd.merge(production_wind_solar, weather_by_day, how='left', left_index=True, right_index=True)


print(combined.shape)
combined.head()

"""Finding out correlation between certain quantities"""

sns.pairplot(combined, 
             x_vars=['v1', 'v2', 'v_50m', 'z0'], 
             y_vars=['DE_wind_generation_actual'])

sns.pairplot(combined,
             x_vars=['SWTDN', 'SWGDN', 'T', 'rho', 'p'],
             y_vars=['DE_wind_generation_actual'])

sns.pairplot(combined,
             x_vars=['v1', 'v2', 'v_50m', 'z0'],
             y_vars=['DE_solar_generation_actual'])

sns.pairplot(combined,
             x_vars=['SWTDN', 'SWGDN', 'T', 'rho', 'p'],
             y_vars=['DE_solar_generation_actual'])


"""There seems to be a linear relation between the wind generation and
 the wind velocities v1, v2 and v_50m, but not the other quantities."""

sns.jointplot(x='v1',
              y='DE_wind_generation_actual',
              data=combined, kind='reg')

sns.jointplot(x='v2',
              y='DE_wind_generation_actual',
              data=combined, kind='reg')

sns.jointplot(x='v_50m',
              y='DE_wind_generation_actual',
              data=combined, kind='reg')

"""Similarly, there seems to be a linear relation between the solar generation
 and the top-of-the-atmosphere and ground radiation."""

sns.jointplot(x='SWTDN',
              y='DE_solar_generation_actual',
              data=combined, kind='reg')

sns.jointplot(x='SWGDN',
              y='DE_solar_generation_actual',
              data=combined, kind='reg')




## Wind Generation

"""To predict the wind generation, we construct the features matrix `X_wind`
 with the features v1, v2, v_50m and z0, and the target `Y_wind` with actual wind generation.
"""

X_wind = combined[['v1', 'v2', 'v_50m', 'z0']]
y_wind = combined['DE_wind_generation_actual']

from sklearn.model_selection import train_test_split 

X_wind_train, X_wind_test, y_wind_train, y_wind_test = train_test_split(X_wind, y_wind,
                                                    test_size = 0.2, random_state=2020)


## Solar Generation

"""To predict the solar generation, We again construct the features matrix `X_solar`,
 but now with the features SWTDN, SWGDN and T, and the target `Y_solar` with actual solar generation.
"""

X_solar = combined[['SWTDN', 'SWGDN', 'T']]
y_solar = combined['DE_solar_generation_actual']


from sklearn.model_selection import train_test_split 

X_solar_train, X_solar_test, y_solar_train, y_solar_test = train_test_split(X_solar, y_solar,
                                                    test_size = 0.2,random_state=2020)



                ######----MODEL BUILDING----#########

#-----------------------------------------------------------------------------#
#     #----------------Linear Regression-------------------#
    
#   ### FOR WIND GENERATION ###

# from sklearn.linear_model import LinearRegression
# lin_reg = LinearRegression()

# lin_reg.fit(X_wind_train, y_wind_train)

# y_wind_pred =lin_reg.predict(X_wind_test)

# from sklearn.metrics import r2_score
# print(r2_score(y_wind_test, y_wind_pred))


# ### FOR SOLAR GENERATION ###

# from sklearn.linear_model import LinearRegression
# lin_reg = LinearRegression()
# lin_reg.fit(X_solar_train, y_solar_train)


# y_solar_pred = lin_reg.predict(X_solar_test)

# from sklearn.metrics import r2_score

# print(r2_score(y_solar_test, y_solar_pred))


# #-----------------------------------------------------------------------------#
#   #----------------KNN Regression-------------------#

#   ### FOR WIND GENERATION ###
  
# from sklearn.neighbors import KNeighborsRegressor
# for i in  range(1,21):
#     print("for k = ",i)
#     knn = KNeighborsRegressor(n_neighbors=i)
#     knn.fit( X_wind_train , y_wind_train )
#     y_wind_pred = knn.predict(X_wind_test)
    
#     from sklearn.metrics import r2_score
#     print(r2_score(y_wind_test, y_wind_pred))

#   ### FOR SOLAR GENERATION ###
  
# from sklearn.neighbors import KNeighborsRegressor
# for i in  range(1,21):
#     print("for k = ",i)
#     knn = KNeighborsRegressor(n_neighbors=i)
#     knn.fit( X_solar_train , y_solar_train )
#     y_solar_pred = knn.predict(X_solar_test)
    
#     from sklearn.metrics import r2_score
#     print(r2_score(y_solar_test, y_solar_pred))
    
# #-----------------------------------------------------------------------------#
#   #----------------Ridge Regression-------------------#

#   ### FOR WIND GENERATION ###
# from sklearn.linear_model import Ridge

# ridge = Ridge(alpha=2)
# ridge.fit(X_wind_train, y_wind_train) 
# y_wind_pred = ridge.predict(X_wind_test)

# from sklearn.metrics import r2_score
# print(r2_score(y_wind_test, y_wind_pred))


#   ### FOR SOLAR GENERATION ###
# from sklearn.linear_model import Ridge

# ridge = Ridge(alpha=2)
# ridge.fit(X_solar_train, y_solar_train) 
# y_pred = ridge.predict(X_solar_test)

# from sklearn.metrics import r2_score
# print(r2_score(y_solar_test, y_solar_pred))

# #-----------------------------------------------------------------------------#
#   #----------------ElasticNet Regression-------------------#

#   ### FOR WIND GENERATION ###
  
# from sklearn.linear_model import ElasticNet

# elastic = ElasticNet(alpha=2, l1_ratio=0.6)
# elastic.fit(X_wind_train, y_wind_train) 
# y_solar_pred = elastic.predict(X_wind_test)

# from sklearn.metrics import r2_score
# print(r2_score(y_wind_test, y_wind_pred))

#   ### FOR SOLAR GENERATION ###
  
# from sklearn.linear_model import ElasticNet

# elastic = ElasticNet(alpha=2, l1_ratio=0.6)
# elastic.fit(X_solar_train, y_solar_train) 
# y_solar_pred = elastic.predict(X_solar_test)

# from sklearn.metrics import r2_score
# print(r2_score(y_solar_test, y_solar_pred))

#-----------------------------------------------------------------------------#
 #----------------Bagging Regression-------------------#
 
 ### FOR WIND GENERATION ###
# Default: Tree Regressor
from sklearn.ensemble import BaggingRegressor

model_wind = BaggingRegressor(random_state=2020,oob_score=True,
                            max_features = X_wind_train.shape[1],
                            max_samples=X_wind_train.shape[0])


model_wind.fit( X_wind_train , y_wind_train )

print("Out of Bag Score = " + "{:.4f}".format(model_wind.oob_score_))

y_wind_pred = model_wind.predict(X_wind_test)

from sklearn.metrics import r2_score
print("Wind r2 score : ",r2_score(y_wind_test, y_wind_pred))


### FOR SOLAR GENERATION ###
# Default: Tree Regressor
from sklearn.ensemble import BaggingRegressor

model_slr = BaggingRegressor(random_state=2020,oob_score=True,
                            max_features = X_solar_train.shape[1],
                            max_samples=X_solar_train.shape[0])


model_slr.fit( X_solar_train , y_solar_train )

print("Out of Bag Score = " + "{:.4f}".format(model_slr.oob_score_))

y_solar_pred = model_slr.predict(X_solar_test)

from sklearn.metrics import r2_score
print("Solar r2 score : ",r2_score(y_solar_test, y_solar_pred))


#-----------------------------------------------------------------------------#

import pickle

#To serialize the object --dump()
pickle.dump(model_wind,open('model_wind.pkl','wb'))

# Loading model to compare the results
#To deserialize the object --load()
#model = pickle.load(open('model_wind.pkl','rb'))


#model.predict([[239.858,118.792,285.388]])

#-----------------------------------------------------------------------------#


import pickle
pickle.dump(model_slr,open('model_slr.pkl','wb'))

# Loading model to compare the results
#model = pickle.load(open('model_slr.pkl','rb'))


#model.predict([[239.858,118.792,285.388]])



