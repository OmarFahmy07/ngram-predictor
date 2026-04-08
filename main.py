import argparse
from dotenv import load_dotenv

def main():
    # Read configuration from .env file
    load_dotenv("config/.env")

    # Set up command-line argument parsing
    parser = argparse.ArgumentParser(
        description="N-Gram Next-Word Predictor (Capstone Project)")
    
    # Define the command-line argument for selecting the pipeline step
    parser.add_argument(
        "--step",
        choices=["dataprep", "model", "inference", "all"],
        help="Which pipeline step to run.")
    
    # Parse the command-line arguments
    parser.parse_args()

# This is a guard to ensure that the main function is called only when this 
# script is executed directly, and not when imported as a module.
if __name__ == "__main__":
    main()