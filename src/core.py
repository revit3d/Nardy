import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk

from image_processing import ImageProcessor
from tiles_segmentation import TileExtractor


class ImageSegmentationApp:
    def __init__(self, root):
        self.root = root
        self.root.geometry(f'+{1400}+{900}')
        self.root.title("Triomino Tiles Classification App")
        self.button_font = ("Helvetica", 12, "bold")

        self.image = None
        self.processed_image = None

        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()

        self.canvas = tk.Canvas(height=screen_height // 2 - 10,
                                width=screen_width // 2 - 10,
                                bg="lightblue")
        self.canvas.pack(fill="both", expand=True)

        self.canvas_text = tk.Canvas(height=screen_height // 4 - 10,
                                     width=screen_width // 2 - 10,
                                     bg="lightblue")
        self.canvas_text.pack(fill="both", expand=True)

        self.frame = tk.Frame(root)

        self.load_button = tk.Button(self.frame,
                                     text="Load Image\n",
                                     command=self.load_image,
                                     font=self.button_font)
        self.load_button.pack(side=tk.LEFT, expand=True, fill=tk.X)

        self.extract_button = tk.Button(self.frame,
                                     text="Extract tile contours\n",
                                     command=self.run_segmentation,
                                     font=self.button_font)
        self.extract_button.pack(side=tk.LEFT, expand=True, fill=tk.X)

        self.process_button = tk.Button(self.frame,
                                     text="Find and classify tiles\n",
                                     command=self.run_classification,
                                     font=self.button_font)
        self.process_button.pack(side=tk.LEFT, expand=True, fill=tk.X)

        self.frame.pack(side=tk.TOP, fill=tk.X, pady=10, padx=10)

        self.path_label = tk.Label(self.root,
                                   text="Output file path:",
                                   font=self.button_font)
        self.path_label.pack(side=tk.LEFT, fill=tk.X)

        self.write_path = tk.Entry(self.root, width=30)
        self.write_path.pack(side=tk.LEFT, pady=10)

    def load_image(self):
        file_path = filedialog.askopenfilename(initialdir="./images", title="Select Image",
                                                filetypes=(("BMP files", "*.bmp"),
                                                           ("All files", "*.*")))
        if file_path:
            self.image = cv2.imread(file_path)
            self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
            self.display_image(self.image)

    def update_message(self, new_message):
        self.canvas_text.delete("all")

        canvas_width = self.canvas_text.winfo_width()
        canvas_height = self.canvas_text.winfo_height()

        center_x = canvas_width / 2
        center_y = canvas_height / 2
        self.canvas_text.create_text(center_x, center_y,
                                     text=new_message,
                                     fill="black",
                                     font=("Helvetica", 14, "bold"),
                                     justify=tk.CENTER,
                                     anchor=tk.CENTER)

    def resize_image_to_canvas(self, img, canvas_size):
        canvas_width, canvas_height = canvas_size

        img_width, img_height = img.size
        width_ratio = canvas_width / img_width
        height_ratio = canvas_height / img_height
        scaling_factor = min(width_ratio, height_ratio)

        new_width = int(img_width * scaling_factor)
        new_height = int(img_height * scaling_factor)

        return img.resize((new_width, new_height))

    def display_image(self, image):
        if image is None:
            return None
        
        self.canvas_text.delete("all")

        canvas_width = self.canvas_text.winfo_width()
        canvas_height = self.canvas_text.winfo_height()

        center_x = canvas_width / 2
        center_y = canvas_height / 2

        img_pil = Image.fromarray(image).convert("RGB")
        img_pil = self.resize_image_to_canvas(img_pil, (canvas_width, canvas_height))

        self.image_tk = ImageTk.PhotoImage(img_pil)
        self.canvas.create_image(center_x,
                                 center_y,
                                 image=self.image_tk,
                                 anchor=tk.CENTER)
    
    def run_segmentation(self):
        if self.image is None:
            return None

        extractor = TileExtractor()
        unique_contours = extractor.find_objects(self.image)

        self.processed_image = self.image.copy()
        contour_clr = (0, 255, 0)
        cv2.drawContours(self.processed_image, [c.vertices for c in unique_contours], -1, contour_clr, 2)
        self.display_image(self.processed_image)
        self.update_message(f'Number of found tiles\non the image: {len(unique_contours)}')
    
    def run_classification(self):
        if self.image is None:
            return None

        proc = ImageProcessor()
        path = self.write_path.get()
        if path is None or path == '':
            self.update_message('Please pass path to the file\nfor saving processing info')
            return None
        proc.process_image(self.image, path)
        self.update_message(f'Results saved in file\n\'{path}\'')


def main():
    root = tk.Tk()
    app = ImageSegmentationApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()
