# 📊 Practice 01 & 02 - Comparison of Negative Movie Reviews

A simple and educational PySpark project that processes a movie review dataset from IMDB in Portuguese and English. It consists of two scripts:

- `main.py`: Calculates the **sum of IDs** from negative reviews.
- `niam.py`: Counts the number of **words in negative reviews** (Portuguese and English) and compares them.

Both scripts use the [Rich](https://github.com/Textualize/rich) library for elegant terminal outputs.

---

## 🚀 Features

- ✅ Reads a CSV dataset with movie reviews
- 🔎 Filters reviews where the sentiment is **negative** (`sentiment == "neg"`)
- ➕ `main.py`: Sums the `id` field of filtered reviews
- 🔤 `niam.py`: Counts and compares word totals in both languages
- 📋 Displays results in formatted tables
- 🎓 Includes a student identifier (**RU**)

---

## 🛠️ Requirements

Before running the scripts, make sure the following are installed:

- [Python 3.7+](https://www.python.org/downloads/)
- [Apache Spark](https://spark.apache.org/downloads.html)
- [PySpark](https://pypi.org/project/pyspark/)
- [Rich](https://pypi.org/project/rich/)

You can install the Python dependencies via:

```bash
pip install pyspark rich
```

Make sure Apache Spark is properly configured on your machine.

---

## 📁 Dataset

The project uses the file `imdb-reviews-pt-br.csv`, which contains movie reviews scraped from IMDB.

**Expected Columns:**

- `id`: Review identifier
- `sentiment`: Sentiment label (`pos` or `neg`)
- `text_pt`: Review text in Portuguese
- `text_en`: Review text in English

> 📌 **Note**: You may replace this file with a similar one, as long as it includes the columns above.

---

## 🧠 What Each Script Does

### `main.py`
- Initializes a Spark session
- Loads the dataset
- Filters reviews with negative sentiment
- Calculates the **sum of their IDs**
- Displays result with **RU**

### `niam.py`
- Initializes a Spark session
- Loads the dataset
- Filters reviews with negative sentiment
- Counts the **total words** in `text_pt` and `text_en`
- Calculates the **difference** between the two
- Displays the result with **RU**

---

## ▶️ Usage

1. Make sure your dataset is in the same folder as the scripts or adjust the file path.

2. Run either script:

```bash
python main.py   # To calculate sum of negative review IDs
python niam.py   # To count and compare word totals in negative reviews
```

3. You'll see an output like:

```text
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃       Practice 01 Results - Sum of IDs               ┃
┡━━━━━━━━━━━━━━━━━━━━━━┯━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
│ Description           │ Value                         │
├──────────────────────┼───────────────────────────────┤
│ Sum of IDs for negat…│ 178932                        │
│ RU                   │ 1234567                        │
└──────────────────────┴───────────────────────────────┘
```

Or:

```text
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃       Practice 02 Results - Word Count Comparison          ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━┯━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
│ Description                 │ Value                         │
├────────────────────────────┼───────────────────────────────┤
│ Total words (Portuguese)   │ 50893                         │
│ Total words (English)      │ 47921                         │
│ Difference (PT - EN)       │ 2972                          │
│ RU                         │ 1234567                        │
└────────────────────────────┴───────────────────────────────┘
```

---

## 📂 Project Structure

```
📦 project-folder/
├── 📄 main.py          # Sum of IDs from negative reviews
├── 📄 niam.py          # Word count comparison (PT vs EN)
├── 📄 imdb-reviews-pt-br.csv
└── 📄 README.md
```

---

## 💡 Potential Enhancements

- Add unit tests for logic validation
- Use Spark SQL for more complex analysis
- Generate visualizations using Matplotlib or Plotly
- Export results to CSV or JSON

---

## 📜 License

This project is intended for educational use. No license restrictions applied.
