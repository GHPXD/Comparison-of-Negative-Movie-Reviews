from pyspark.sql import SparkSession
from rich.console import Console
from rich.table import Table

# Initialize the Rich console for nicer output
console = Console()

# My RU
RU = "1234567"

# Starts the Spark session
def start_spark_session(app_name: str):
    return SparkSession.builder \
        .appName(app_name) \
        .getOrCreate()

# Loads the CSV dataset into a Spark DataFrame.
def load_data(spark, file_path: str):
    console.log("[bold green]Loading data from CSV file...[/bold green]")
    return spark.read.csv(
        file_path,
        header=True,
        inferSchema=True,
        encoding="UTF-8"
    )

# Filters the data where the 'sentiment' column equals 'neg'.
def filter_negative_data(df):
    console.log("[bold cyan]Filtering records with negative sentiment...[/bold cyan]")
    return df.filter(df.sentiment == "neg")

# Calculates the sum of IDs in the filtered DataFrame.
def calculate_id_sum(df):
    console.log("[bold yellow]Calculating sum of IDs...[/bold yellow]")
    return df.selectExpr("sum(id) as id_sum").collect()[0]["id_sum"]

# Displays the results in a formatted table, including the RU and a separating line.
def display_results(id_sum, ru):
    table = Table(title="Practice 01 Results - Sum of IDs", style="bold magenta", show_lines=True)

    table.add_column("Description", style="cyan", justify="left")
    table.add_column("Value", style="green", justify="center")

    # Add rows with the results
    table.add_row("Sum of IDs for negative movies", str(id_sum), end_section=True)
    table.add_row("RU", ru)

    console.print(table)

def main():
    # Initialize the Spark session
    spark = start_spark_session("Sum of Negative IDs")

    try:
        # Dataset path (adjust as needed)
        dataset_path = "imdb-reviews-pt-br.csv"

        # Load the data
        df = load_data(spark, dataset_path)

        # Display first records for validation
        console.print("[bold blue]Displaying the first records of the dataset:[/bold blue]")
        console.print(df.show(5), style="dim")

        # Filter the negative data
        negative_data = filter_negative_data(df)

        # Calculate the sum of IDs
        id_sum = calculate_id_sum(negative_data)

        # Display the final result with RU and visual separation
        display_results(id_sum, RU)

    except Exception as e:
        console.print(f"[bold red]Error during execution:[/bold red] {e}")

    finally:
        # Stop the Spark session
        spark.stop()
        console.log("[bold green]Spark session terminated.[/bold green]")

if __name__ == "__main__":
    main()