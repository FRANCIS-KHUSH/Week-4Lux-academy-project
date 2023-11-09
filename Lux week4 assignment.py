# %I importeed the necessary packages and handled the missing values %
import pytz
import warnings
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from statsmodels.tsa.seasonal import seasonal_decompose
warnings.filterwarnings('ignore', category=FutureWarning)

# %Loaded the dataset%
data_path = "C:/Users/Francis Mwangi/Desktop/SNAP/craigslist_vehicles.csv"
data = pd.read_csv(data_path)

# %Previewed the first 5 rows or the data%
data.head() 

# %list of the columns in the dataframe%
data.columns


# %Checking if the columns exist in the dataframe before dropping them%
columns_to_drop = ['Unnamed: 0', 'url', 'region_url', 'VIN', 'image_url', 'description', 'county', 'lat', 'long']


# %filter the columns to drop only those that exist in the DataFrame%
columns_to_drop_existing = [col for col in columns_to_drop if col in data.columns]


# %drop the existing columns%
data = data.drop(columns=columns_to_drop_existing)


# %Checking how the dataframe looks like now%
data.head() 


# %I fill the numeric ones with mean and categorical ones with mode%
def handle_missing_values(data):
    # fill missing numerical values with mean
    numerical_columns = ['year', 'odometer']
    data[numerical_columns] = data[numerical_columns].fillna(data[numerical_columns].mean())

    # fill missing categorical values with mode
    categorical_columns = ['manufacturer', 'model', 'condition', 'cylinders', 'fuel', 'title_status',
                           'transmission', 'drive', 'size', 'type', 'paint_color', 'posting_date']
    data[categorical_columns] = data[categorical_columns].apply(lambda x: x.fillna(x.mode().iloc[0]))

    return data



data = handle_missing_values(data)


# %Seeing the dataframe%
data.head() 


# %Covert the colunn "postinf_date" to datetime data type%
data['posting_date'] = pd.to_datetime(data['posting_date'])

# %Aggregating the data to be able to analyze the patterns%
def convert_to_tz_aware(posting_date):
    if not posting_date.tzinfo:
        return posting_date.replace(tzinfo=pytz.utc)
    else:
        return posting_date

data['posting_date'] = data['posting_date'].apply(convert_to_tz_aware)

data_agg = data.groupby(['region', 'type', 'posting_date']).size().reset_index(name='count')

data_agg = data_agg.sort_values(by='posting_date')

print(data_agg.head())


# %Creating the time-series chart%
fig = px.line(data_agg, x='posting_date', y='count', color='region', line_group='type',
              title='Number of Available Vehicles Over Time by Region and Vehicle Type',
              labels={'count': 'Number of Vehicles'})

# %customizing the layout%
fig.update_layout(
    xaxis_title='Posting Date',
    yaxis_title='Number of Vehicles',
    hovermode='x',
    showlegend=True,
)

# %Show the chart%
fig.show()


# %Grouping the data by dat and count the no of listings%
data_freq = data_agg.groupby(pd.Grouper(key='posting_date', freq='D')).sum().reset_index()


# %Create the time-frequency graph%
fig_freq = go.Figure(data=go.Bar(
    x=data_freq['posting_date'],
    y=data_freq['count'],
    marker_color='royalblue',
    opacity=0.8
))

# %Customize the layout%
fig_freq.update_layout(
    title='Time Frequency Graph: Number of Vehicle Listings per Day',
    xaxis_title='Posting Date',
    yaxis_title='Number of Vehicle Listings',
    xaxis_tickangle=-45,
)

#%Showing the time frequency graph%
fig_freq.show()

# %Perfomed seasonal decomposition%
data_agg = data_agg.set_index('posting_date')
result = seasonal_decompose(data_agg['count'], model='additive', period=365)


# %Created a new DataFrame to store the decomposion components%
decomposed_data = pd.DataFrame({
    'trend': result.trend,
    'seasonal': result.seasonal,
    'residual': result.resid,
})

# %reset the index for plotting%
decomposed_data = decomposed_data.reset_index()


# %Plot the seasonal decomposition%
fig_decompose = go.Figure()

fig_decompose.add_trace(go.Scatter(x=decomposed_data['posting_date'], y=decomposed_data['trend'],
                                   mode='lines', name='Trend'))
fig_decompose.add_trace(go.Scatter(x=decomposed_data['posting_date'], y=decomposed_data['seasonal'],
                                   mode='lines', name='Seasonal'))
fig_decompose.add_trace(go.Scatter(x=decomposed_data['posting_date'], y=decomposed_data['residual'],
                                   mode='lines', name='Residual'))

# %Customizing the layout%
fig_decompose.update_layout(title='Seasonal Decomposition of Time Series',
                            xaxis_title='Posting Date',
                            yaxis_title='Counts',
                            showlegend=True)

# %Show the plot%
fig_decompose.show()


# %%



