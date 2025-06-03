import argparse
import sys

def calculate_mae(filepath):
    """
    Calculates the Mean Absolute Error (MAE) for a set of values from a file.
    The MAE is computed as the average absolute deviation from the mean of the values.

    Args:
        filepath (str): The path to the file containing numeric values, one per line.

    Returns:
        float: The calculated MAE value.

    Raises:
        FileNotFoundError: If the specified file does not exist.
        ValueError: If the file contains non-numeric data.
    """
    values = []
    try:
        with open(filepath, 'r') as f:
            for line in f:
                values.append(float(line.strip()))
    except FileNotFoundError:
        raise FileNotFoundError(f"Error: File not found at '{filepath}'")
    except ValueError:
        raise ValueError(f"Error: Non-numeric data found in file '{filepath}'")

    if not values:
        # Handle empty file case: MAE is 0 for an empty set of values
        return 0.0

    # Calculate the mean of the values
    mean_value = sum(values) / len(values)

    # Calculate the Mean Absolute Error (MAE)
    absolute_errors = [abs(v - mean_value) for v in values]
    mae = sum(absolute_errors) / len(absolute_errors)

    return mae

def main():
    # Set up the argument parser
    parser = argparse.ArgumentParser(
        description="Compute the Mean Absolute Error (MAE) for distance values in a file."
    )
    parser.add_argument(
        '--distances_filename',
        type=str,
        required=True,
        help="The path to the file containing distance values, one per line."
    )

    # Parse the arguments
    args = parser.parse_args()
    filepath = args.distances_filename
    filename = filepath.split('/')[-1]  # Extract filename from the path

    try:
        mae_result = calculate_mae(filepath)
        # Using "D MAE" as it was in the original request, but "MAE" or "MAD" are also common.
        print(f"D MAE for file {filename}: {mae_result:.4f}")
    except (FileNotFoundError, ValueError) as e:
        print(e)
        sys.exit(1)

if __name__ == "__main__":
    main()