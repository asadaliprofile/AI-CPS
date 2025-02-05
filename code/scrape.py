import pandas as pd
import requests
import re

# Define the URL of the Markdown file
url = "https://raw.githubusercontent.com/asadaliprofile/exampleRepository/master/AllCars.md"  # Replace with your actual URL

# Fetch the content from the URL
response = requests.get(url)
if response.status_code == 200:
    markdown_content = response.text.split("\n")  # Split into lines
else:
    print("Failed to fetch the file. Check the URL.")
    exit()

# Extract lines containing table-like structures (Markdown tables use "|")
table_lines = [line.strip() for line in markdown_content if "|" in line]

# Remove separator lines (Markdown tables use "|----|----|" as separators)
cleaned_lines = [line for line in table_lines if not re.match(r"^\|\s*-+\s*\|", line)]

# Extract structured data
structured_data = []
for line in cleaned_lines:
    columns = [col.strip() for col in line.split("|") if col.strip()]
    if len(columns) == 9:  # Ensure it matches the expected number of columns
        structured_data.append(columns)

# Define column names
columns = ["model", "year", "price", "transmission", "mileage", "fuelType", "tax", "mpg", "engineSize"]

# Create a DataFrame
df = pd.DataFrame(structured_data, columns=columns)

# Convert numeric columns
numeric_columns = ["year", "price", "mileage", "tax", "mpg", "engineSize"]
for col in numeric_columns:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# Data Cleaning: 
# Drop rows where more than 50% of the columns have missing values
df = df.dropna(thresh=len(df.columns) * 0.5)
# Fill missing numeric values with the column median
numeric_columns = ["year", "price", "mileage", "tax", "mpg", "engineSize"]
for col in numeric_columns:
    if col in df.columns:
        df[col].fillna(df[col].median(), inplace=True)
# Fill missing categorical values with the most common (mode) value
categorical_columns = ["model", "transmission", "fuelType"]
for col in categorical_columns:
    if col in df.columns:
        df[col].fillna(df[col].mode()[0], inplace=True)


# Save to CSV
output_csv = "/Users/Asad Ali/AI-CPS/data/joint_data_collection.csv"
df.to_csv(output_csv, index=False)

print(f"Data successfully extracted and saved as {output_csv}")
# Display the first few rows of the structured DataFrame
print(df.head())