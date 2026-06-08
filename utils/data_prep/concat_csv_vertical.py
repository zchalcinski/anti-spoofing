import csv
import sys

def concatenate_and_sort_csv_native(file1_path, file2_path, output_file_path, sort_column_name='name'):
    """
    Concatenates two CSV files and sorts the combined data by a specified column.

    Args:
        file1_path (str): Path to the first CSV file.
        file2_path (str): Path to the second CSV file.
        output_file_path (str): Path to save the combined and sorted CSV file.
        sort_column_name (str): The name of the column to sort by.
    """
    all_data = []
    header = None
    sort_column_index = -1

    # Read the first file
    try:
        with open(file1_path, 'r', newline='') as f1:
            reader = csv.reader(f1, delimiter=';')
            header = next(reader)  # Read the header
            try:
                sort_column_index = header.index(sort_column_name)
            except ValueError:
                print(f"Error: Sort column '{sort_column_name}' not found in header of {file1_path}.")
                return

            for row in reader:
                all_data.append(row)
    except FileNotFoundError:
        print(f"Error: File not found - {file1_path}")
        return
    except Exception as e:
        print(f"Error reading {file1_path}: {e}")
        return

    # Read the second file
    try:
        with open(file2_path, 'r', newline='') as f2:
            reader = csv.reader(f2, delimiter=';')
            header2 = next(reader) # Read and skip header of the second file
            # Optional: Check if headers are compatible (same columns, same order)
            if header2 != header:
                print(f"Warning: Headers of {file1_path} and {file2_path} do not match. Using header from the first file.")
                # You might want more sophisticated header merging logic here if needed.

            for row in reader:
                all_data.append(row)
    except FileNotFoundError:
        print(f"Error: File not found - {file2_path}")
        return
    except Exception as e:
        print(f"Error reading {file2_path}: {e}")
        return

    if not all_data:
        print("No data to process.")
        return

    # Sort the data
    # The key lambda function extracts the value from the sort_column_index for comparison
    all_data.sort(key=lambda row: row[sort_column_index])

    # Write the combined and sorted data to the output file
    try:
        with open(output_file_path, 'w', newline='') as outfile:
            writer = csv.writer(outfile,delimiter=';')
            writer.writerow(header)    # Write the header
            writer.writerows(all_data) # Write the sorted data rows
        print(f"Successfully concatenated and sorted files into {output_file_path}")
    except Exception as e:
        print(f"Error writing to {output_file_path}: {e}")

# --- Example Usage ---
if __name__ == "__main__":
    # Create dummy CSV files for testing
    with open('file1.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['id', 'name', 'value'])
        writer.writerow(['1', 'T_0000000005', 'Alpha'])
        writer.writerow(['2', 'T_0000000001', 'Beta'])
        writer.writerow(['3', 'T_0000100000', 'Gamma'])

    with open('file2.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['id', 'name', 'value']) # Ensure header matches for simplicity
        writer.writerow(['4', 'T_0000000002', 'Delta'])
        writer.writerow(['5', 'T_0000000000', 'Epsilon'])
        writer.writerow(['6', 'T_0000000100', 'Zeta'])

    concatenate_and_sort_csv_native(sys.argv[1], sys.argv[2], './csv/concat.csv', sort_column_name='name')

    # Expected output in combined_sorted_native.csv:
    # id,name,value
    # 5,T_0000000000,Epsilon
    # 2,T_0000000001,Beta
    # 4,T_0000000002,Delta
    # 1,T_0000000005,Alpha
    # 6,T_0000000100,Zeta
    # 3,T_0000100000,Gamma