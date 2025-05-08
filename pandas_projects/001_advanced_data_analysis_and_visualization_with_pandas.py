import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pandas import DataFrame
from scipy.stats import zscore
# Load the dataset
file_path: str = "employee_dataset.xlsx"
df: DataFrame = pd.read_excel(file_path)


# 1. Analyze average performance scores across departments
def department_performance_analysis(df: DataFrame):
    """
    This method help you find out the average performance scores across departments
    :param df:
    :return:
    """
    performance_summary: DataFrame = (
        df.groupby('Department')['Performance_Score']
        .mean()
        .reset_index()
        .rename(columns={'Performance_Score': 'Avg_Performance_Score'})
        .sort_values(by='Avg_Performance_Score', ascending=False)
    )
    print("Average Performance Scores by Department:")
    print(performance_summary)
    return performance_summary


# 2. Generate pivot table for salary distribution by city and department
def salary_pivot_table(df: DataFrame):
    salary_pivot: DataFrame = pd.pivot_table(
        df,
        values='Salary',
        index='City',
        columns='Department',
        aggfunc='mean'
    )
    print("Salary Distribution by City and Department:")
    print(salary_pivot)
    return salary_pivot


# 3. Time-series analysis of hiring trends
def hiring_trend_analysis(df: DataFrame):
    """
    Time-series analysis of hiring trends
    :param df:
    """
    df['Join_Year'] = df['Join_Date'].dt.year
    hiring_trend = df.groupby('Join_Year').size()
    print("Hiring Trend Over the Years:")
    print(hiring_trend)

    # Plotting the trend
    plt.figure(figsize=(10, 6))
    sns.lineplot(x=hiring_trend.index, y=hiring_trend.values, marker='o')
    plt.title("Hiring Trends (2010-2023)")
    plt.xlabel("Year")
    plt.ylabel("Number of Hires")
    plt.grid(True)
    plt.show()

    return hiring_trend


# 4. Analyze the average salary by age group
def average_salary_by_age_group(df):
    bins = [20, 30, 40, 50, 60]  # pins and...
    labels = ['20-29', '30-39', '40-49', '50-59']  # space between the pins (if there are n pins, labels = n - 1)
    df['Age_Group'] = pd.cut(df['Age'], bins=bins, labels=labels, right=False)
    age_salary_summary = (
        df.groupby('Age_Group',  observed=False)['Salary']
        .mean()
        .reset_index()
        .rename(columns={'Salary': 'Avg_Salary'})
    )
    print("Average Salary by Age Group:")
    print(age_salary_summary)
    return age_salary_summary


# 5. Identify the top 5 cities with the highest average performance scores
def top_cities_by_performance(df: DataFrame):
    city_performance = (
        df.groupby('City')['Performance_Score']
        .mean()
        .reset_index()
        .rename(columns={'Performance_Score': 'Avg_Performance_Score'})
        .sort_values(by='Avg_Performance_Score', ascending=False)
        .head(5)
    )
    print("Top 5 Cities by Average Performance Score:")
    print(city_performance)
    return city_performance


# 6. Analyze salary distribution by age using boxplots
def salary_distribution_by_age(df):
    """
    Analyze salary distribution by age using boxplots
    :param df:
    """
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='Age', y='Salary', data=df)
    plt.title("Salary Distribution by Age")
    plt.xlabel("Age")
    plt.ylabel("Salary")
    plt.show()
    return df


# 7. Find the department with the longest average tenure of employees
def longest_tenure_by_department(df):
    df['Tenure'] = (pd.Timestamp.now() - df['Join_Date']).dt.days / 365
    tenure_summary = (
        df.groupby('Department')['Tenure']
        .mean()
        .reset_index()
        .rename(columns={'Tenure': 'Avg_Tenure'})
        .sort_values(by='Avg_Tenure', ascending=False)
    )
    print("Longest Average Tenure by Department:")
    print(tenure_summary)
    return tenure_summary


# 8. Correlate performance scores with salaries to identify trends
def performance_salary_correlation(df):
    correlation = df['Performance_Score'].corr(df['Salary'])
    print(f"Correlation between Performance Score and Salary: {correlation:.2f}")

    # Scatter plot
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x='Performance_Score', y='Salary', data=df)
    plt.title("Correlation Between Performance Score and Salary")
    plt.xlabel("Performance Score")
    plt.ylabel("Salary")
    plt.show()
    return correlation


# 9. Visualize department size as a percentage of total employees
def department_size_visualization(df):
    department_counts = df['Department'].value_counts(normalize=True) * 100
    # department_counts_float = (df['Department'].size()).astype(float)
    # department_counts = (department_counts_float / department_counts_float.sum()) * 100
    department_counts.plot(kind='pie', autopct='%1.1f%%', figsize=(8, 8))
    plt.title("Department Size as Percentage of Total Employees")
    plt.xlabel("")
    plt.ylabel("")
    plt.legend(title="Department", bbox_to_anchor=(0.98, 1), loc='upper left')
    plt.grid(True)
    plt.show()
    print(f"the department counts is {department_counts}")
    return department_counts


# 10. Analyze hiring trends by city over the years
def hiring_trend_by_city(df):
    df['Join_Year'] = df['Join_Date'].dt.year

    hiring_trends = (
        df.groupby(['City', 'Join_Year']).size()
        .reset_index(name='Hires')
        .pivot(index='Join_Year', columns='City', values='Hires')
    )
    print("Hiring Trends by City Over the Years:")
    print(hiring_trends)

    # Plotting
    hiring_trends.plot(figsize=(12, 8))
    plt.title("Hiring Trends by City Over the Years")
    plt.xlabel("Year")
    plt.ylabel("Number of Hires")
    plt.legend(title="City", bbox_to_anchor=(0.98, 1), loc='upper left')
    plt.grid(True)
    plt.show()
    return hiring_trends


# 11. Detect any anomalies in salary distribution using z-score analysis
def detect_salary_anomalies(data) -> DataFrame:
    data['Salary Z-Score'] = zscore(data['Salary'])
    anomaly: DataFrame = pd.DataFrame(data[data['Salary Z-Score'].abs() > 3])  # Employees with z-score > 3 or < -3
    print(anomaly)
    return anomaly


# 12. Identify the top 5 highest-paid employees in each department
def top_5_highest_paid_employees(data):
    result = data.groupby('Department').apply(lambda x: x.nlargest(5, 'Salary')).reset_index(drop=True)
    print(result)
    return result


# 13. Calculate the percentage of employees in each department who have a performance score above a threshold
def high_performance_percentage(data, threshold=4):
    result = data.groupby('Department').apply(lambda x: (x['Performance_Score'] > threshold).mean() * 100)
    print(result)
    return result


def generate_report_fit(df: DataFrame):
    return {
        "department_performance": department_performance_analysis(df),  # 1
        "salary_pivot": salary_pivot_table(df),  # 2
        "hiring_trend": hiring_trend_analysis(df),  # 3
        "average_salary_by_agegroup":  average_salary_by_age_group(df),  # 4
        "top_cities_performance": top_cities_by_performance(df),  # 5
        "salary_distribution_byage": salary_distribution_by_age(df),  # 6
        "longest_tenure_bydepartment": longest_tenure_by_department(df),  # 7
        "performance_salarycorrelation": performance_salary_correlation(df),  # 8
        "department_sizevisualization": department_size_visualization(df),  # 9
        "hiring_trend_bycity": hiring_trend_by_city(df),  # 10
        "detect_salary_anomalies": detect_salary_anomalies(df),  # 11
        "top_5_highest_paid_employees": top_5_highest_paid_employees(df),  # 12
        "high_performance_percentage": high_performance_percentage(df),  # 13
    }


# Combining analyses into a comprehensive report
def generate_report(df: DataFrame):
    print("Generating report...")

    print("1) Department Performance Analysis")
    department_performance = department_performance_analysis(df)

    print("2) Salary Pivot Table")
    salary_pivot = salary_pivot_table(df)

    print("3) Time-series analysis of hiring trends")
    hiring_trend = hiring_trend_analysis(df)

    print("4) Average Salary by Age Group:")
    average_salary_by_agegroup = average_salary_by_age_group(df)

    print("5) Top Cities by Performance:")
    top_cities_performance = top_cities_by_performance(df)

    print("6) Salary Distribution by Age:")
    salary_distribution_byage = salary_distribution_by_age(df)

    print("7) Longest Tenure by Department:")
    longest_tenure_bydepartment = longest_tenure_by_department(df)

    print("8) Performance and Salary Correlation:")
    performance_salarycorrelation = performance_salary_correlation(df)

    print("9) Department Size Visualization:")
    department_sizevisualization = department_size_visualization(df)

    print("10) Hiring Trends by City:")
    hiring_trend_bycity = hiring_trend_by_city(df)

    print("11) Detect any anomalies in salary distribution using z-score analysis:")
    detect_salary_anomalies_ = detect_salary_anomalies(df)

    # return {}
    return {
        "department_performance": department_performance, # 1
        "salary_pivot": salary_pivot,  # 2
        "hiring_trend": hiring_trend,  # 3
        "average_salary_by_agegroup": average_salary_by_agegroup,  # 4
        "top_cities_performance": top_cities_performance,  # 5
        "salary_distribution_byage": salary_distribution_byage,  # 6
        "longest_tenure_bydepartment": longest_tenure_bydepartment,  # 7
        "performance_salarycorrelation": performance_salarycorrelation,  # 8
        "department_sizevisualization": department_sizevisualization,  # 9
        "hiring_trend_bycity": hiring_trend_bycity,  # 10
        "detect_salary_anomalies_": detect_salary_anomalies_,  # 11
    }


# test if it works!
comment = """
# 6. Perform correlation analysis between numerical features
def correlation_analysis(data):
    return data.corr()

# 7. Find the average tenure of employees by department and gender
def average_tenure_by_dept_and_gender(data):
    return data.groupby(['Department', 'Gender'])['Tenure'].mean().reset_index()
"""

# Example Use Cases
if __name__ == "__main__":
    # Run all analyses
    # report = generate_report_fit(df)
    high_performance_percentage(df)

# https://chatgpt.com/c/6758be0f-fdf0-800c-98b6-aebae7632d04 (single-pandas-project: 001)
# https://chatgpt.com/c/674b65b9-fecc-800c-8311-7f681df9b305 (projects: pandas)
