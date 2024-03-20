import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import os
import csv

class ImageAnnotator:
    def __init__(self, master):
        self.master = master
        self.master.title("Image Annotator")

        self.image_path = None
        self.image_label = tk.Label(master)
        self.sliders = {}
        self.current_directory = None
        self.file_list = []
        self.current_file_index = -1
        self.export_path = None
        self.annotations = dict()

        self.setup_ui()

    def setup_ui(self):
        for i, label in enumerate(["color", "asymmetry", "label"]):
            tk.Label(self.master, text=label.capitalize()).grid(row=2, column=i)
            slider = tk.Scale(self.master, from_=1, to=5, orient=tk.HORIZONTAL)
            slider.grid(row=3, column=i)
            self.sliders[label] = slider

        self.previous_button = tk.Button(self.master, text="Previous", command=self.previous_image)
        self.previous_button.grid(row=4, column=0, columnspan=1)

        self.next_button = tk.Button(self.master, text="Next", command=self.next_image)
        self.next_button.grid(row=4, column=2, columnspan=1)

        self.annotate_button = tk.Button(self.master, text="Annotate", command=self.annotate_file)
        self.annotate_button.grid(row=4, column=1, columnspan=1)
        
        self.save_button = tk.Button(self.master, text="Save annotations", command=self.save_annotations)
        self.save_button.grid(row=5, column=1, columnspan=1)

    def isAnnotated(self,filepath):
        if sum(self.annotations[filepath])!=0:
            return "   Annotated   "
        else:
            return "Not Annotated"

    def display_file(self, filepath):
        self.image_path = filepath
        self.annotate_label = tk.Label(self.master, text=self.isAnnotated(filepath))
        self.annotate_label.grid(row=1,column=1)
        img = Image.open(filepath)
        img.thumbnail((250, 250))
        img = ImageTk.PhotoImage(img)
        self.image_label.configure(image=img)
        self.image_label.image = img
        self.image_label.grid(row=0, column=0, columnspan=3)
        for i,slider in enumerate(self.sliders.values()):
            slider.set(self.annotations[filepath][i])
        

    def annotate_file(self):
        self.annotations[self.image_path]=[]
        for slider in self.sliders.values():
                self.annotations[self.image_path].append(slider.get())
        self.display_file(self.image_path)
    
    def save_annotations(self):
        with open(self.export_path, 'w+', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(["filename", "color", "asymmetry", "label"])
                for file in self.file_list:
                    row=[]
                    row.append(file)
                    for k in range(len(self.annotations[file])): 
                        row.append(self.annotations[file][k])
                    writer.writerow(row)
    
    def annotate_directory(self, directory_path, export_path):
        self.current_directory = directory_path
        self.export_path = export_path
        self.file_list = [os.path.join(directory_path, f) for f in os.listdir(directory_path) if f.endswith('png')]
        for i in self.file_list:
            self.annotations[i]=[0,0,0]
        self.current_file_index = -1
        self.next_image()
        if os.path.isfile(self.export_path):
            with open(export_path) as f:
                for row in f:
                    split_row = row.split(",")
                    filename=split_row[0]
                    if filename=="filename":continue
                    v1=int(split_row[1])
                    v2=int(split_row[2])
                    v3=int(split_row[3])
                    self.annotations[filename]=[v1,v2,v3]
        self.display_file(self.image_path)

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
    annotator = ImageAnnotator(root)
    annotator.annotate_directory("images", "annotations.csv")
    root.mainloop()
