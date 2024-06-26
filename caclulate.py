import csv

input_file = 'test.csv'


with open(input_file, 'r') as file:
    reader = csv.reader(file)
    rows = [row[1] for row in reader]
    print(rows)
    sum = 0
    for i in range(1, len(rows)):
        sum += float(rows[i])
    print(sum)
    average = sum / (len(rows) - 1)
    print(average)