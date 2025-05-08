import pandas as pd
from causalml.dataset import make_uplift_classification
from sklearn.model_selection import train_test_split
import numpy as np

# Dictionary specifying the number of features that will have a positive effect on retention for each treatment
n_uplift_increase_dict = {
    "email_campaign": 2,
    "in_app_notification": 3,
    "call_campaign": 3,
    "voucher": 4
}

# Dictionary specifying the number of features that will have a negative effect on retention for each treatment
n_uplift_decrease_dict = {
    "email_campaign": 1,
    "in_app_notification": 1,
    "call_campaign": 2,
    "voucher": 1
}

# Dictionary specifying the magnitude of positive effect on retention for each treatment
delta_uplift_increase_dict = {
    "email_campaign": 0.05,  # Email campaign increases retention by 5 percentage points
    "in_app_notification": 0.03,  # In-app notifications have a smaller but still positive effect
    "call_campaign": 0.08,  # Direct calls have a strong positive effect
    "voucher": 0.10  # Vouchers have the strongest positive effect
}

# Dictionary specifying the magnitude of negative effect on retention for each treatment
delta_uplift_decrease_dict = {
    "email_campaign": 0.02,  # Email campaign might slightly decrease retention for some customers
    "in_app_notification": 0.01,  # In-app notifications have minimal negative effect
    "call_campaign": 0.03,  # Calls might annoy some customers more
    "voucher": 0.02  # Vouchers might make some customers think the product is overpriced
}

# Dictionary specifying the number of mixed features (combination of informative and positive uplift) for each treatment
n_uplift_increase_mix_informative_dict = {
    "email_campaign": 1,
    "in_app_notification": 2,
    "call_campaign": 1,
    "voucher": 2
}

# Dictionary specifying the number of mixed features (combination of informative and negative uplift) for each treatment
n_uplift_decrease_mix_informative_dict = {
    "email_campaign": 1,
    "in_app_notification": 1,
    "call_campaign": 1,
    "voucher": 1
}

positive_class_proportion = 0.7  # Baseline retention rate

y_name = 'retention'
# Generate the dataset
df, feature_names = make_uplift_classification(
    n_samples=20000,  # Increased sample size for more robust results
    treatment_name=['email_campaign', 'in_app_notification', 'call_campaign', 'voucher'],
    y_name=y_name,
    n_classification_features=20,  # Increased number of features
    n_classification_informative=10,
    n_uplift_increase_dict=n_uplift_increase_dict,
    n_uplift_decrease_dict=n_uplift_decrease_dict,
    delta_uplift_increase_dict=delta_uplift_increase_dict,
    delta_uplift_decrease_dict=delta_uplift_decrease_dict,
    n_uplift_increase_mix_informative_dict=n_uplift_increase_mix_informative_dict,
    n_uplift_decrease_mix_informative_dict=n_uplift_decrease_mix_informative_dict,
    positive_class_proportion=positive_class_proportion,
    random_seed=42
)


# Encoding treatments variables
encoding_dict = {
    'call_campaign': 3,
    'email_campaign': 1,
    'voucher': 4,
    'in_app_notification': 2,
    'control': 0
}

# Create a new column with encoded values
df['treatment_group_numeric'] = df['treatment_group_key'].map(encoding_dict)

df: pd.DataFrame = pd.DataFrame(df)
print(f"uplift columns: {df.columns}")
print(f"uplift: {df[['treatment_group_numeric', 'retention', 'treatment_effect']].head(5)}")


def prepare_data(df, feature_names, y_name, test_size=0.3, random_state=42):
    """
    Prepare data for uplift modeling, including splitting into train and test sets,
    and creating mono-treatment subsets.
    """
    # Create binary treatment column
    df['treatment_col'] = np.where(df['treatment_group_key'] == 'control', 0, 1)

    # Split data into train and test sets
    df_train, df_test = train_test_split(df, test_size=test_size, random_state=random_state)

    # Create mono-treatment subsets
    df_train_mono = df_train[df_train['treatment_group_key'].isin(['email_campaign', 'control'])]
    df_test_mono = df_test[df_test['treatment_group_key'].isin(['email_campaign', 'control'])]

    # Prepare features, treatment, and target variables for full dataset
    X_train = df_train[feature_names].values
    X_test = df_test[feature_names].values
    treatment_train = df_train['treatment_group_key'].values
    treatment_test = df_test['treatment_group_key'].values
    y_train = df_train[y_name].values
    y_test = df_test[y_name].values

    # Prepare features, treatment, and target variables for mono-treatment dataset
    X_train_mono = df_train_mono[feature_names].values
    X_test_mono = df_test_mono[feature_names].values
    treatment_train_mono = df_train_mono['treatment_group_key'].values
    treatment_test_mono = df_test_mono['treatment_group_key'].values
    y_train_mono = df_train_mono[y_name].values
    y_test_mono = df_test_mono[y_name].values

    return {
        'df_train': df_train, 'df_test': df_test,
        'df_train_mono': df_train_mono, 'df_test_mono': df_test_mono,
        'X_train': X_train, 'X_test': X_test,
        'X_train_mono': X_train_mono, 'X_test_mono': X_test_mono,
        'treatment_train': treatment_train, 'treatment_test': treatment_test,
        'treatment_train_mono': treatment_train_mono, 'treatment_test_mono': treatment_test_mono,
        'y_train': y_train, 'y_test': y_test,
        'y_train_mono': y_train_mono, 'y_test_mono': y_test_mono
    }


# Usage
data = prepare_data(df, feature_names, y_name)

# Print shapes for verification
print(f"Full test set shape: {data['df_test'].shape}")
print(f"Mono-treatment test set shape: {data['df_test_mono'].shape}")

# Access prepared data
df_train, df_test = data['df_train'], data['df_test']
df_train_mono, df_test_mono = data['df_train_mono'], data['df_test_mono']
X_train, y_train = data['X_train'], data['y_train']
X_test, y_test = data['X_test'], data['y_test']
X_train_mono, y_train_mono = data['X_train_mono'], data['y_train_mono']
X_test_mono, y_test_mono = data['X_test_mono'], data['y_test_mono']
treatment_train, treatment_test = data['treatment_train'], data['treatment_test']
treatment_train_mono, treatment_test_mono = data['treatment_train_mono'], data['treatment_test_mono']
