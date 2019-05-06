import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency, linregress
from datetime import date


filename = 'data/temp_datalab_records_linkedin_company.csv'
df = pd.read_csv(filename)#, parse_dates=True)#, index_col='DATE')
df['as_of_date'] = pd.to_datetime(df['as_of_date'])
df.rename(columns={ 'as_of_date': 'date'}, inplace=True)
print(df.head())

# range of the dates
print(df['date'].min())
print(df['date'].max())


# Increasing Speed of Employee Numbers
def slope(x):
    if len(x)<2:
        return np.nan
    x1 = x['date'].values # remove index, since it may not start from 0
    x1 = (x1-x1[0]).astype('timedelta64[D]').astype(int)
    x2 = x['employees_on_platform'].values
    return linregress(x1,x2)[0]

df_slope = df.groupby('company_name').apply(lambda x: slope(x))
df_slope = pd.DataFrame(df_slope)
df_slope.columns = ['slope']
df_slope = df_slope.sort_values('slope', ascending=False)
print(df_slope.head(10))

plt.figure()
df_slope.hist(bins = 50, color = 'blue', edgecolor='black',log=True,figsize = (10,5))
plt.grid(True)
plt.title('Histogram: Increasing Slope of Company Employees')
plt.xlabel('Slope')
plt.ylabel('Count')
plt.tight_layout()
plt.savefig('Expand_Slope.png')


print(df[df['company_name'] == 'Hewlett-Packard']['date'].min())
print(df[df['company_name'] == 'Hewlett-Packard']['date'].max())


# Number of Increased Employees
def expand(x):
    if len(x)<2:
        return np.nan, np.nan
    x1 = x['employees_on_platform'].values
    if x1[0] == 0:
        return x1[-1], np.nan
    expand_num = x1[-1] - x1[0]
    expand_ratio = expand_num/x1[0]
    return expand_num, expand_ratio

df_expand = df.groupby('company_name').apply(lambda x: expand(x))
df_expand = pd.DataFrame(df_expand)
df_expand.columns = ['expansion']
df_expand['expand_num'], df_expand['expand_ratio'] = list(zip(*df_expand['expansion'].values))
df_expand.drop(columns=['expansion'], inplace= True)

df_expand = df_expand.sort_values('expand_num', ascending=False)
print(df_expand.head(20))

plt.figure()
df_expand[['expand_num']].hist(bins = 50, color = 'blue', edgecolor='black',log=True,figsize = (10,5))
plt.grid(True)
plt.title('Histogram: Increased Number of Company Employees')
plt.xlabel('Increased Number')
plt.ylabel('Count')
plt.tight_layout()
plt.savefig('Expand_Number.png')


print(df[df['company_name'] == 'Walmart']['date'].min())
print(df[df['company_name'] == 'Walmart']['date'].max())


names = np.concatenate((df_expand.index[:10].values, df_expand.index[10-1:20].values))
symbols = np.array(['WMT','AMZN','IBM','ACN','MCD','MAR','CTSH','F','AAPL','GOOG','GOOGL',
                   'JPM','HSBC','VOD','C','BAC','S','SBUX','GE','T','WFC'])

# Company Names to Stock Names
print(np.transpose(np.concatenate(([names], [symbols]), axis=0)))


# For Amazon, Compare the Number of Employees and Stock Price
df1 = df[df['company_name'] == 'Amazon'].copy()
df1.set_index('date', inplace = True)

filename = 'data/stocks/AMZN.csv'
df1_stock = pd.read_csv(filename, parse_dates=True, index_col='Date')
df1_stock.head()

df1['price'] = df1_stock['Adj Close']
print(df1[['employees_on_platform','price']].head())

df_plot = df1[['employees_on_platform','price']].copy()
df_plot.dropna(inplace = True)
df_plot = df_plot/df_plot.iloc[0]

plt.figure()
df_plot.plot(color = ['red','black'], figsize = (10,5))
plt.grid(True)
plt.legend()
plt.title('Number of Employees & Stock Price')
plt.xlabel('Date')
plt.tight_layout()
plt.savefig('Employees_Price_Amazon.png')


# Find an Indicator for Stock Price
days = np.array([])
coefs = np.array([])
for i in range(-100,100+1):
    x = df_plot['employees_on_platform'].values
    y = df_plot['price'].values
    if i > 0: # x lead y
        x = x[:-i]
        y = y[i:]
    elif i < 0: # x lag y
        x = x[-i:]
        y = y[:i]
    c = np.corrcoef(x, y)[0,1]
    days = np.append(days,i)
    coefs = np.append(coefs,c)

plt.figure(figsize = (7,5))
plt.plot(days,coefs, color = 'black')
plt.title('Correlation Coefficients vs Leading Days\n Amazon')
plt.xlabel('Leading Days')
plt.ylabel('Correlation Coefficients')
plt.grid(True)
plt.tight_layout()
plt.savefig('Indicator_Amazon.png')


idx = np.argmax(coefs)
print('Highest correlation coefficients: %.10g' % coefs[idx])
print('With Leading days: %d' % days[idx] )



# Indicator for Stocks of Top 20 Expansion Companies

corrcoefs = np.array([])
lag_days =  np.array([])

for i in range(len(names)):
    name = names[i]
    symbol = symbols[i]
    
    df1 = df[df['company_name'] == name].copy()
    df1.set_index('date', inplace = True)

    filename = 'data/stocks/'+symbol+'.csv'
    df1_stock = pd.read_csv(filename, parse_dates=True, index_col='Date')
    df1['price'] = df1_stock['Adj Close']
            
    df_plot = df1[['employees_on_platform','price']].copy()
    df_plot.dropna(inplace = True)
    df_plot = df_plot/df_plot.iloc[0]
    
    days = np.array([])
    coefs = np.array([])
    for i in range(-500,500+1):
        x = df_plot['employees_on_platform'].values
        y = df_plot['price'].values
        if i > 0: # x lead y
            x = x[:-i]
            y = y[i:]
        elif i < 0: # x lag y
            x = x[-i:]
            y = y[:i]
        
        c = np.corrcoef(x, y)[0,1]
        days = np.append(days,i)
        coefs = np.append(coefs,c)
    
    idx = np.argmax(coefs)
    lag_days = np.append(lag_days,days[idx])
    corrcoefs = np.append(corrcoefs,coefs[idx])

plt.figure(figsize = (9,5.5))
sizes = np.abs(lag_days * corrcoefs)*2
for i in range(len(names)):
    plt.scatter(lag_days[i], corrcoefs[i], sizes[i], label = symbols[i])
plt.title('Indicator Performance: Number Employees Leading Stock Prices')
plt.xlabel('Leading Days')
plt.ylabel('Correlation Coefficients')
plt.ylim([None,1.05])
plt.legend(bbox_to_anchor=(1, 1),ncol=1)
plt.grid(True)
plt.tight_layout()
plt.savefig('Indicator_Performance.png')
