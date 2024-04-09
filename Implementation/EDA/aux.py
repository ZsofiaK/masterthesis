import csv

def write_dict_to_csv(dictionary, csv_file, key_header, value_header):
    '''
    Writes a Python dictionary to a csv file.
    '''
    with open(csv_file, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)

        # Write the header with column names
        csv_writer.writerow([key_header, value_header])

        # Write each key-value pair to the CSV file
        for key, value in dictionary.items():
            csv_writer.writerow([key, value])

def read_csv_to_dict(csv_file, header=True):
    result_dict = {}

    with open(csv_file, 'r') as csvfile:
        csv_reader = csv.reader(csvfile)
        
        # Skip the header row if there is one.
        if header:
            header = next(csv_reader, None)

        # Iterate through each row and populate the dictionary
        for row in csv_reader:
            if len(row) >= 2:  # Check if there are at least two columns
                key = row[0]
                value = row[1]
                
                result_dict[key] = value

    return result_dict