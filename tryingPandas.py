import pandas as pd

# Creating a dictionary
data = {
    'Name': ['Rahul', 'Aman', 'Priya', 'Sneha', 'Karan'],
    'Age': [20, 21, 19, 22, 20],
    'Marks': [85, 90, 78, 92, 88],
    'City': ['Delhi', 'Mumbai', 'Pune', 'Bangalore', 'Chennai']
}

# Creating a DataFrame
df = pd.DataFrame(data)

print("Original DataFrame:")
print(df)


# 1. head() - Displays first 5 rows
print("\nFirst 5 rows:")
print(df.head())


# 2. tail() - Displays last 5 rows
print("\nLast 5 rows:")
print(df.tail())


# 3. info() - Displays information about DataFrame
print("\nDataFrame Information:")
df.info()


# 4. describe() - Displays statistical information
print("\nStatistical Description:")
print(df.describe())


# 5. shape - Returns number of rows and columns
print("\nShape of DataFrame:")
print(df.shape)


# 6. columns - Displays column names
print("\nColumn Names:")
print(df.columns)


# 7. index - Displays row indexes
print("\nDataFrame Index:")
print(df.index)


# 8. Accessing a single column
print("\nNames:")
print(df['Name'])


# 9. Accessing multiple columns
print("\nName and Marks:")
print(df[['Name', 'Marks']])


# 10. loc[] - Accessing rows using labels
print("\nFirst row using loc:")
print(df.loc[0])


# 11. iloc[] - Accessing rows using positions
print("\nSecond row using iloc:")
print(df.iloc[1])


# 12. Filtering data
print("\nStudents with marks greater than 85:")
print(df[df['Marks'] > 85])


# 13. Sorting values
print("\nStudents sorted by marks:")
print(df.sort_values(by='Marks'))


# 14. Adding a new column
df['Result'] = ['Pass', 'Pass', 'Pass', 'Pass', 'Pass']

print("\nAfter adding Result column:")
print(df)


# 15. Dropping a column
df = df.drop('City', axis=1)

print("\nAfter dropping City column:")
print(df)


# 16. Checking for null values
print("\nChecking null values:")
print(df.isnull())


# 17. Checking total null values
print("\nTotal null values:")
print(df.isnull().sum())


# 18. Finding average marks
print("\nAverage Marks:")
print(df['Marks'].mean())


# 19. Finding maximum marks
print("\nMaximum Marks:")
print(df['Marks'].max())


# 20. Finding minimum marks
print("\nMinimum Marks:")
print(df['Marks'].min())


# 21. Counting values
print("\nCounting Results:")
print(df['Result'].value_counts())


# 22. Renaming columns
df = df.rename(columns={'Marks': 'Score'})

print("\nAfter renaming column:")
print(df)


# 23. Saving DataFrame to CSV
df.to_csv('students.csv', index=False)
