from pyspark.sql import SparkSession
from rich.console import Console
from rich.table import Table

# Initialize the Rich console for formatted output
console = Console()

# Personal identifier (RU)
RU = "1234567"

# Function to initialize the Spark session
def start_spark_session(app_name):
    return SparkSession.builder.appName(app_name).getOrCreate()

# Function to load data from a CSV file
def load_data(spark, file_path):
    console.log("[bold green]Loading data from CSV file...[/bold green]")
    return spark.read.csv(file_path, header=True, inferSchema=True, encoding="UTF-8")

# Filters negative texts based on the 'sentiment' column
def filter_negative_texts(df):
    console.log("[bold cyan]Filtering negative texts...[/bold cyan]")
    return df.filter(df.sentiment == "neg")

# Counts the total number of words in a specific column ('text_pt' or 'text_en')
def count_words(df, text_column):
    console.log(f"[bold yellow]Counting words in column '{text_column}'...[/bold yellow]")
    word_rdd = df.rdd.flatMap(lambda row: row[text_column].split())
    return word_rdd.count()

# Displays the results in a formatted table
def display_results(total_pt, total_en, difference, ru):
    table = Table(title="Practice 02 Results - Word Count", style="bold magenta", show_lines=True)
    table.add_column("Description", style="cyan", justify="left")
    table.add_column("Value", style="green", justify="center")
    
    # Add rows with results
    table.add_row("Total words (Portuguese)", str(total_pt), end_section=True)
    table.add_row("Total words (English)", str(total_en), end_section=True)
    table.add_row("Difference (Portuguese - English)", str(difference), end_section=True)
    table.add_row("RU", ru)
    
    console.print(table)

# Main function
def main():
    # Initialize Spark session
    spark = start_spark_session("Negative Word Count")

    try:
        # Dataset path (adjust as needed)
        dataset_path = "imdb-reviews-pt-br.csv"

        # Load data into Spark DataFrame
        df = load_data(spark, dataset_path)

        # Display first records for validation
        console.print("[bold blue]Displaying the first records of the dataset:[/bold blue]")
        console.print(df.show(5), style="dim")

        # Filter only negative texts
        negative_texts = filter_negative_texts(df)

        # Count words in 'text_pt' and 'text_en' columns
        total_words_pt = count_words(negative_texts, "text_pt")
        total_words_en = count_words(negative_texts, "text_en")

        # Calculate the difference between counts
        word_difference = total_words_pt - total_words_en

        # Display final results in table format
        display_results(total_words_pt, total_words_en, word_difference, RU)

    except Exception as e:
        console.print(f"[bold red]Error during execution:[/bold red] {e}")

    finally:
        # Stop Spark session
        spark.stop()
        console.log("[bold green]Spark session terminated.[/bold green]")

# Run the main script
if __name__ == "__main__":
    main()