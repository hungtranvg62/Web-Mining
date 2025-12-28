import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# 1. Load your original dataset
df = pd.read_csv('book_recomm_dataset.csv')
print(f"Original Data: {len(df)} rows")

# 2. Encode Users and Books (Must be done BEFORE splitting)
# This ensures that UserID 276747 maps to the same integer in both files.

# User Encoding
user_enc = LabelEncoder()
df['user_encoded'] = user_enc.fit_transform(df['User_ID'])

# Book Encoding (Using Title to match your NCF model)
book_enc = LabelEncoder()
df['book_encoded'] = book_enc.fit_transform(df['Book_Title'])

# 3. Perform the Split
# random_state=42 guarantees this matches your NCF and SVD notebooks exactly
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

# 4. Save to CSV
train_df.to_csv('train.csv', index=False)
test_df.to_csv('test.csv', index=False)

print(f"✅ Splits Saved!")
print(f"Train Rows: {len(train_df)}")
print(f"Test Rows:  {len(test_df)}")
print("\nPreview of train.csv:")
print(train_df[['User_ID', 'user_encoded', 'Book_Title', 'book_encoded', 'Book_Rating']].head())