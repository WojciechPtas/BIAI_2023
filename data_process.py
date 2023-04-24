import os
import pandas as pd

files = os.listdir(os.getcwd() + "\\pomiary_txt")



csv_lines = []

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

df = pd.DataFrame(columns=['reference_distance', 'real_distance', 'signal_strength', "delta"], data=csv_lines)

df.to_csv('data.csv')