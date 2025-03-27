from src.data.data_loader import load_all_cvs
import pandas as pd
import matplotlib.pyplot as plt

data_dict = load_all_cvs("C:/Users/danil/Desktop/PROJECTS/formula1/datasets/raw/")


##Main data
constructors = data_dict.get("constructors")
drivers = data_dict.get('drivers')
qualifying = data_dict.get('qualifying')
races = data_dict.get('races')
results = data_dict.get('results')


# #Secondary Data
# circuits = data_dict.get('circuits')
# constructor_results = data_dict.get("constructor_results")
# constructor_standings = data_dict.get("constructor_standings")
# driver_standings = data_dict.get('driver_standings')
# lap_times = data_dict.get('lap_times')
# pit_stops = data_dict.get('lap_times')
# seasons = data_dict.get('seasons')
# sprint_results = data_dict.get('sprint_results')
# status = data_dict.get('status')



def unique_values(column):
    return column.nunique()


def missing_values(column1, column2):
    return set(column1) - set(column2)


def duplicates_values(column):
    return column.duplicated().sum()
    
def plot_bar(df, x_col, y_col, title, xlabel, ylabel, color='skyblue', rotate=45, figsize=(16,8)):
    plt.figure(figsize=figsize)
    df.plot(kind='bar', x=x_col, y=y_col, color=color, legend=False)
    plt.xticks(rotation=rotate, ha='center')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()
    plt.show()

unique_df = pd.DataFrame({
    'Category': ['Constructors', 'Drivers', 'Races'],
    'Unique Count': [
        unique_values(results['constructorId']),
        unique_values(results['driverId']),
        unique_values(results['raceId'])
    ]
})
print(unique_df)
missing_df = pd.DataFrame({
    'Category' : ['Constructors', 'Drivers', 'Races'],
    'Missing values' : [
        missing_values(results.constructorId, constructors.constructorId),
        missing_values(results.driverId, drivers.driverId),
        missing_values(results.raceId, races.raceId)
        ]
    })

duplicates_df = pd.DataFrame({
    'Category': ['Constructors', 'Drivers', 'Races'],
    'Duplicate Count': [
        duplicates_values(constructors.constructorId),
        duplicates_values(drivers.driverId),
        duplicates_values(races.raceId)
        ]
    })

#We haven't any duplicates and missing values

constructors_nationality = constructors.groupby('nationality').size().reset_index(name='Count')


constructors_nationality.plot(kind='bar', x='nationality', y='Count', color='skyblue', legend=False)


plt.xticks(rotation=45)
plt.xlabel('Nationality')
plt.ylabel('Count')
plt.title('Frequency Distribution of Constructors by Nationality')

plt.show()

results.shape
            
# master_data = merge_data([results, races, drivers, constructors])
# master_data.head()
# master_data.shape

#Merge data
datas = [races, drivers, constructors, qualifying]
columns = ['raceId', 'driverId', 'constructorId', ['raceId', 'driverId']]

masterdata = results

for d, c in zip(datas, columns):
    masterdata = pd.merge(masterdata, d, how='left', on = c)
    print(masterdata.shape)

masterdata = masterdata.rename(columns={
    'position_x' : 'race_position',
    'position_y' : 'qualifying_position',
    'name_x' : 'gp_name',
    'name_y' : 'constructor_name',
    'constructorId_x' : 'constructorId',
    'number_x' : 'number',
    'nationality_x' : 'nationality_driver',
    'nationality_y' : 'nationality_constructor'
    })


masterdata = masterdata.drop(['number', 'number_y', 'constructorId_y'], axis=1)
masterdata.columns

masterdata.shape

masterdata.info()

masterdata['race_position'] = pd.to_numeric(masterdata['race_position'], errors='coerce')
masterdata.info()

masterdata['top3'] = masterdata['race_position'].apply(lambda x:1 if x <=3 else 0)

constructors_top3 = masterdata.groupby('constructor_name')['top3'].sum().reset_index().sort_values(by='top3', ascending=False)
constructors_top3 = constructors_top3[constructors_top3['top3'] > 0]
   
plot_bar(constructors_top3, "constructor_name", "top3", "Constructors Top 3", "Constructor", 'Count of Top-3 Finishers')



driver_top3 = masterdata.groupby('driverRef')['top3'].sum().reset_index().sort_values(by='top3', ascending=False)
driver_top3 = driver_top3[driver_top3['top3'] > 0]

masterdata.info()

data2000 = masterdata[masterdata.year >= 2000]
top3drivers2000 = data2000.groupby('driverRef')['top3'].sum().reset_index().sort_values(by="top3", ascending=False)
top3drivers2000 = top3drivers2000[top3drivers2000['top3'] > 0]
top3drivers2000['top3_percent'] = (top3drivers2000['top3'] / top3drivers2000['top3'].sum() * 100).round(2)
plot_bar(top3drivers2000, 'driverRef', 'top3', 'Drivers Top 3 from 2000', 'Drivers', 'Couunt Top3-Finishers')

top3constructors2000 = data2000.groupby('constructor_name')['top3'].sum().reset_index().sort_values(by='top3', ascending=False)
top3constructors2000 = top3constructors2000[top3constructors2000['top3'] > 0]
top3constructors2000['top3_percent'] = (top3constructors2000['top3'] / top3constructors2000['top3'].sum() * 100).round(2)
plot_bar(top3constructors2000, 'constructor_name', 'top3', "Constructors Top 3 from 2000", 'Constructors', 'Count of Top-3 Finishers')


# driver_top3 = masterdata.groupby('driverRef')['top3'].sum().reset_index()
# total_top3 = driver_top3['top3'].sum()  # вычисляем общее количество top3
# driver_top3['top3_percent'] = (driver_top3['top3'] / total_top3 * 100).round(2)  # добавляем столбец с процентами
# driver_top3 = driver_top3.sort_values(by='top3_percent', ascending=False)


