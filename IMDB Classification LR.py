import pandas as pd, re, nltk, pickle
from matplotlib import style
style.use('ggplot')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
# from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt

nltk.download('stopwords')
nltk.download('punkt')

stop_words = set(stopwords.words('english'))

# Load the dataset
df = pd.read_csv('Logistic Regression\IMDB Dataset.csv', encoding='unicode_escape')


def data_processing(text):
    text = text.lower()
    text = re.sub(r'\@#+|\#', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    text_tokens = word_tokenize(text)
    filtered_text = [w for w in text_tokens if not w in stop_words]
    return " ".join(filtered_text)

def stemming(data):
    stemmer = PorterStemmer()
    text = [stemmer.stem(word) for word in data]
    return data

print(df.head())

# Apply preprocessing to the dataset
df['review'] = df['review'].apply(data_processing)
df['review'] = df['review'].apply(stemming)

print(df.head())

# Vectorization
vect = TfidfVectorizer(max_features=34083, min_df=7, max_df=0.8).fit(df['review'])
features_names = vect.get_feature_names_out()
print(len(features_names))

x = vect.fit_transform(df['review'])
y = df['sentiment']

# Train-test split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.30)

# Model training and prediction
logreg = LogisticRegression(C=1.0, penalty='l1', solver='liblinear')
logreg_pred_train = logreg.fit(x_train, y_train).predict(x_train)
logreg_pred_test = logreg.predict(x_test)

# Calculate accuracy scores
accuracy_train = accuracy_score(y_train, logreg_pred_train)
accuracy_test = accuracy_score(y_test, logreg_pred_test)

print("Training Accuracy = {:.2f}%".format(accuracy_train * 100))
print("Validation Accuracy: {:.2f}%".format(accuracy_test * 100))

# Create learning curve
train_accuracies = []
test_accuracies = []
train_sizes = [1000, 5000, 10000, 20000, 30000, 40000, 50000]

for train_size in train_sizes:
    # Subset the training set
    x_subset_train = x_train[:train_size]
    y_subset_train = y_train[:train_size]

    # Train the model on the subset
    logreg.fit(x_subset_train, y_subset_train)

    # Predict on training and test sets
    logreg_pred_train_subset = logreg.predict(x_subset_train)
    logreg_pred_test_subset = logreg.predict(x_test)

    # Calculate accuracies
    accuracy_train_subset = accuracy_score(y_subset_train, logreg_pred_train_subset)
    accuracy_test_subset = accuracy_score(y_test, logreg_pred_test_subset)

    # Append accuracies to the lists
    train_accuracies.append(accuracy_train_subset)
    test_accuracies.append(accuracy_test_subset)

# Plot the learning curve
plt.plot(train_sizes, train_accuracies, label='Training Accuracy')
plt.plot(train_sizes, test_accuracies, label='Validation Accuracy')

# --------------- Confusion Matrix ------------------------
from sklearn.metrics._plot.confusion_matrix import ConfusionMatrixDisplay
style.use('classic')
ConMat = confusion_matrix(y_test, logreg_pred_test, labels = logreg.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix = ConMat, display_labels= logreg.classes_)
disp.plot()


# --------------- Save The Vectors ------------------------
with open('Logistic Regression\Vectors.sav', 'wb') as file:
      pickle.dump(vect, file)

# --------------- Save The Model ------------------------
with open('Logistic Regression\Model.sav', 'wb') as file:
    pickle.dump(logreg, file)