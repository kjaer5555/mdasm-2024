import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import os
import csv
import math

class ImageAnnotator:
    def __init__(self, master, features=["1","2","3"],skipReasons=["1","2","3"]):
        self.master = master
        self.master.title("Image Annotator")

        self.image_name = None
        self.image_label = tk.Label(master)
        self.sliders = {}
        self.current_directory = None
        self.file_list = []
        self.current_file_index = -1
        self.export_path = None
        self.annotations = dict()
        
        self.features = features
        
        #self.skip=0
        self.skipReasons=skipReasons
        self.skipReason=tk.StringVar()
        self.skipReason.set(" ")
        self.checkboxes = {}
        
        self.setup_ui()

    def setup_ui(self):
        for i, label in enumerate(self.features):
            row=3+math.floor(i/9)
            column=i%9
            tk.Label(self.master, text=label.capitalize()).grid(row=row-1, column=column)
            slider = tk.Scale(self.master, from_=1, to=5, orient=tk.HORIZONTAL)
            slider.grid(row=row, column=column)
            self.sliders[label] = slider

        column=int(len(self.features)/2)
        
        for i,reason in enumerate(self.skipReasons):
            checkbox = tk.Checkbutton(self.master, text=reason,variable=self.skipReason, onvalue=reason, offvalue=" ")
            checkbox.grid(row=6,column=column-int(len(self.skipReasons)/2)+i)
            self.checkboxes[reason]=checkbox
        
        self.skipLabel = tk.Label(self.master)
        self.skipLabel.grid(row=7,column=column)


        self.previous_button = tk.Button(self.master, text="Previous", command=self.previous_image)
        self.previous_button.grid(row=8, column=column-1, columnspan=1)

        self.next_button = tk.Button(self.master, text="Next", command=self.next_image)
        self.next_button.grid(row=8, column=column+1, columnspan=1)

        self.annotate_button = tk.Button(self.master, text="Annotate", command=self.annotate_file)
        self.annotate_button.grid(row=8, column=column, columnspan=1)
        
        self.save_button = tk.Button(self.master, text="Export annotations", command=self.save_annotations)
        self.save_button.grid(row=9, column=column, columnspan=1)

    def isAnnotated(self,filename):
        if sum(self.annotations[filename][:-1])!=-len(self.annotations[filename][:-1]):
            return "   Annotated   "
        else:
            return "Not Annotated"

    def display_file(self, filename):
        self.image_name = filename
        filepath=self.current_directory + "/" + filename
        column=int(len(self.features)/2)-1
        
        self.annotate_label = tk.Label(self.master, text=self.isAnnotated(filename))
        self.annotate_label.grid(row=1,column=column+1)
        
        img = Image.open(filepath)
        img = img.resize((500,500))
        img = ImageTk.PhotoImage(image=img)
        self.image_label.configure(image=img)
        self.image_label.image = img
        self.image_label.grid(row=0, column=column, columnspan=3)
        
        for i,slider in enumerate(self.sliders.values()):
            slider.set(self.annotations[filename][i])
            
        self.skipLabel.config(text=self.annotations[filename][-1])
        
        self.skipReason.set(" ")
                
    def annotate_file(self):
        if self.skipReason.get()==' ':
            self.annotations[self.image_name]=[]
            for slider in self.sliders.values():
                    self.annotations[self.image_name].append(slider.get())
            self.annotations[self.image_name].append(" ") #there was no skip so it's empty
        else:
            self.annotations[self.image_name]=[0]*len(self.features)
            self.annotations[self.image_name].append(self.skipReason.get())
        self.display_file(self.image_name)
    
    def save_annotations(self):
        with open(self.export_path, 'w+', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(["filename"]+self.features+["reason for skip"])
                for file in self.file_list:
                    row=[]
                    row.append(file)
                    for k in range(len(self.annotations[file])): 
                        row.append(self.annotations[file][k])
                    writer.writerow(row)
    
    def annotate_directory(self, export_path):
        self.current_directory = filedialog.askdirectory()
        self.export_path = export_path
        self.file_list = [f for f in os.listdir(self.current_directory) if f.endswith('png')]
        
        if os.path.isfile(self.export_path):
            with open(export_path) as f:
                for row in f:
                    split_row = row.split(",")
                    filename=split_row[0]
                    if filename=="filename":continue
                    self.annotations[filename]=[]
                    for k in split_row[1:-1]:
                        self.annotations[filename].append(int(k))
                    split_row[-1] = split_row[-1].strip(' "\'\t\r\n')
                    self.annotations[filename].append(split_row[-1]) #for the skip
        else:
            for i in self.file_list:
                self.annotations[i]=[-1]*(len(self.features)) #the +1 is for the skip
                self.annotations[i].append(" ")
        
        self.current_file_index = -1
        self.next_image()

    def display_current_file(self):
        if self.current_file_index < len(self.file_list):
            self.display_file(self.file_list[self.current_file_index])
        else:
            tk.messagebox.showinfo("End", "No more images to annotate in the directory.")

    def next_image(self):
        self.current_file_index += 1
        self.display_current_file()
    
    def previous_image(self):
        self.current_file_index -= 1
        self.display_current_file()

if __name__ == "__main__":
    root = tk.Tk()
    features=["Color", "Asymmetry", "Atypical pigment network",
    "Blue-white veil","Atypical vascular pattern","Irregular streaks",
    "Irregular dots/globules","Irregular blotches","Regression structures"]
    skipReasons=["low quality","unclear view","no visible leason"]
    annotator = ImageAnnotator(root,features=features,skipReasons=skipReasons)
    annotator.annotate_directory("annotations.csv")
    root.mainloop()
