from google.cloud import bigquery

# Initialize a BigQuery client
client = bigquery.Client()

# Reference a public dataset
dataset_ref = client.dataset('usa_names', project='bigquery-public-data')

# Fetch the dataset
dataset = client.get_dataset(dataset_ref)

# List tables in the dataset
tables = list(client.list_tables(dataset))
print(f"Tables in dataset: {[table.table_id for table in tables]}")

# Reference a specific table
table_ref = dataset.table('usa_1910_current')

# Fetch the table
table = client.get_table(table_ref)

# Preview the first 5 rows
rows = client.list_rows(table, max_results=5)
for row in rows:
    print(row)
