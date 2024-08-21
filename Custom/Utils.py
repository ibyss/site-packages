import time
import json
from tkinter import filedialog
from matplotlib import pyplot as plt
from matplotlib import dates

def requestpath(path_name=None):
    request_path = filedialog.askdirectory(initialdir="/", title=f"Please select {path_name} Path")
    return request_path

def requestfile(file_name=None):
    request_file = filedialog.askopenfilename(initialdir="/", title=f"Please select {file_name} File")
    return request_file

class visualization:
    def __init__(self):
        self.jsonfile = f"C:/Users/chanj/Code/torch/computervision/vgg16/vgg16_test4.json"#requestpath(path_name="Json file")
        self()
    def __call__(self):
        with open(self.jsonfile) as f:
            datas = json.load(f)
        name = datas["Train name"]
        total_epoch = datas["Total epoch"]
        start_time = datas["Date"]
        end_time = list(datas["Details"][-1].values())[0]["End time"]
        
        data_list = datas["Details"]
        x_data = []
        elapsed_time_data = []
        cost_data = []
        i = 1
        for epoch in data_list:
            x_data.append(*(epoch.keys()))
            elapsed_time_data.append(float(epoch[f"Epoch{i}"]["Elapsed time"]))
            cost_data.append(float(epoch[f"Epoch{i}"]["Cost"]))
            i += 1
        min_cost = min(cost_data)
        min_cost_epoch = x_data[cost_data.index(min_cost)]
        avg_time = (sum(elapsed_time_data))/len(x_data)
        plt.style.use("default")
        plt.rcParams["figure.figsize"] = (len(x_data) + 4, len(x_data))
        plt.rcParams["font.size"] = 10

        fig, ax1 = plt.subplots()

        ax1.plot(x_data, elapsed_time_data, label="Elapsed time", color="blue")
        ax1.set_ylabel("Elapsed time")
        ax1.text(x=0, y=avg_time, s=f"Avg.time: {avg_time:.3f}", color="purple", verticalalignment="bottom", horizontalalignment="left")
        ax1.axhline(y=avg_time, xmin=0, xmax=1, label="Avg.time", color="purple", linestyle="dotted")

        ax2 = ax1.twinx()
        ax2.plot(x_data, cost_data, label="Cost", color="red")
        ax2.set_ylabel("Costs")
        ax2.text(x=min_cost_epoch, y=min_cost, s=f"Min.cost: {min_cost:.3f} at {min_cost_epoch}", color="orange", verticalalignment="top", horizontalalignment="center")
        ax2.axhline(y=min_cost, xmin=0, xmax=1, label="Min cost", color="orange", linestyle="dotted")

        lines_1, labels_1 = ax1.get_legend_handles_labels()
        lines_2, labels_2 = ax2.get_legend_handles_labels()

        ax1.legend(lines_1 + lines_2, labels_1 + labels_2, fontsize=10)
        
        plt.title(f"{name} at {start_time}~{end_time}(Total epoch: {total_epoch})")
        #plt.subplots_adjust(right=0.8)
        plt.show()