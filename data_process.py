import os
import pandas as pd

CSV_LABELS = ['reference_distance', 'mesured_distance', 'signal_strength', "delta"]

files = os.listdir(os.getcwd() + "\\pomiary_txt")

csv_lines = []
csv_batch = []
csv_batches = []

for file in files:
    if "txt" in file:
        with open(os.path.join(os.getcwd() + "\\pomiary_txt", file), 'r') as f:
            for index, line in enumerate(f.readlines()):
                if(line != "Timed out!|7000\n"):
                    csv_arr = line.split('|')
                    csv_arr = csv_arr[1:3]
                    csv_line = [float(file.replace(".txt", "")), float(csv_arr[0]), float(csv_arr[1]), float(csv_arr[0]) - float(file.replace(".txt", ""))]
                    print("Appending line: " + str(csv_line))
                    csv_lines.append(csv_line)
                    csv_batch.append(csv_line)
            csv_batches.append((file.replace(".txt", ""), pd.DataFrame(columns=CSV_LABELS, data=csv_batch)))
            


df = pd.DataFrame(columns=CSV_LABELS, data=csv_lines)

df.to_csv('./pomiary_csv/data_full.csv')

for batch in csv_batches:
    batch[1].to_csv('./pomiary_csv/' + batch[0] + '.csv')