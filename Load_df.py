#Loads and inspects the source file in the colab environment
import pandas as pd

# Load the Excel file
df = pd.read_excel('/content/aay7315_Data_file_S1.xlsx')

# View column names and the first few rows
print("Column names:\n", df.columns)
df.head()

# Check column data types
df.dtypes
