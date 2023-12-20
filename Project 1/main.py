import pandas as pd
from datetime import datetime as dt

def read_data(filename, cols):
    data = pd.DataFrame(pd.read_csv(filename), columns=cols)
    data['Date'] = pd.to_datetime(data['Date']).dt.date
    data['Time'] = pd.to_datetime(data['Time']).dt.time
    return data

cgm = read_data('CGMData.csv',['Date','Time','Sensor Glucose (mg/dL)'])
insulin = read_data('InsulinData.csv',['Date','Time','Alarm'])

automode=insulin.loc[insulin['Alarm']=='AUTO MODE ACTIVE PLGM OFF']
date = automode.iloc[-1,0]
time = automode.iloc[-1,1]
manual = cgm.loc[(cgm['Date']<date) | ((cgm['Date']==date) & (cgm['Time']<time))].dropna()
auto = cgm.loc[(cgm['Date']>date) | ((cgm['Date']==date) & (cgm['Time']>=time))].dropna()

def timeframes(data):
    start=dt.strptime('00:00:00','%H:%M:%S').time()
    mid=dt.strptime('06:00:00','%H:%M:%S').time()
    end=dt.strptime('23:59:00','%H:%M:%S').time()
    return data.loc[(data['Time']>=mid) & (data['Time']<=end)],data.loc[(data['Time']>=start) & (data['Time']<mid)]

auto_day, auto_night = timeframes(auto)
manual_day, manual_night = timeframes(manual)
threshold = [(180, 0), (250, 0), (70, 180), (70, 150), (0, 70), (0, 54)]

def cal_data(data, threshold):
    a,b = threshold
    if a != 0 and b != 0:
        v=data.loc[(data.iloc[:,2]>=a) & (data.iloc[:,2]<=b)]
    elif a == 0:
        v=data.loc[data.iloc[:,2]<b]
    else:
        v=data.loc[data.iloc[:,2]>a]
        
    date=data.groupby('Date')
    c=v.groupby('Date').size().reset_index()
    c=c.set_index('Date')
    c.columns=['Value']
    p=c['Value']/288*100
    p=p.to_frame()
    result=(p['Value'].sum())/len(date)
    return result

def add_(lis):
    y = []
    for i in lis:
        for j in threshold:
            y.append(cal_data(i,j))
    return y

manual_output = add_([manual_night, manual_day, manual])
auto_output = add_([auto_night, auto_day, auto])
df = pd.DataFrame([manual_output, auto_output])

df.to_csv('Result.csv',index = False, header = False)