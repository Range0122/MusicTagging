import csv

path = '/home/range/Data/TagATune/raw/annotations_final.csv'
# path = '/home/range/Data/TagATune/raw/clip_info_final.csv'
# path = '/home/range/Data/TagATune/raw/comparisons_final.csv'

with open(path) as csv_file:
    csv_reader = csv.reader(csv_file)
    # header = csv_reader.next()
    # print(header)
    i = 0
    for item in csv_reader:
        print(item)
        i += 1
        if i == 2:
            exit()

