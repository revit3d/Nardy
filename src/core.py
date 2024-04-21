import tkinter as tk
from tkinter import filedialog
import cv2
from PIL import Image, ImageTk
import numpy as np


class ImageSegmentationApp:
    def __init__(self, root):
        self.root = root
        self.root.geometry(f'+{1400}+{900}')
        self.root.title("Plane Segmentation App")
        self.button_font = ("Helvetica", 12, "bold")

        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()

        self.canvas = tk.Canvas(height=screen_height // 2 - 10, width=screen_width // 2 - 10, bg="lightblue")
        self.canvas.pack(fill="both", expand=True)

        self.canvas_text = tk.Canvas(height=screen_height // 4 - 10, width=screen_width // 2 - 10, bg="lightblue")
        self.canvas_text.pack(fill="both", expand=True)

        self.image = None
        self.processed_image = None

        self.load_button = tk.Button(self.root,
                                     text="Load Image\n",
                                     command=self.load_image,
                                     font=self.button_font)
        self.load_button.pack(side=tk.LEFT, expand=True, fill=tk.X)

        self.load_button = tk.Button(self.root,
                                     text="Process Image using Canny\n",
                                     command=self.process_image_canny,
                                     font=self.button_font)
        self.load_button.pack(side=tk.LEFT, expand=True, fill=tk.X)

        self.load_button = tk.Button(self.root,
                                     text="Process Image using\npixel-by-pixel transformations",
                                     command=self.process_image_pixel,
                                     font=self.button_font)
        self.load_button.pack(side=tk.LEFT, expand=True, fill=tk.X)

        self.load_button = tk.Button(self.root,
                                     text="Process Image using\nOtsu thresholding",
                                     command=self.process_image_otsu,
                                     font=self.button_font)
        self.load_button.pack(side=tk.LEFT, expand=True, fill=tk.X)

    def load_image(self):
        file_path = filedialog.askopenfilename(initialdir="./images", title="Select Image",
                                                filetypes=(("JPEG files", "*.jpg"), ("PNG files", "*.png"),
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

    def process_image_canny(self):
        if self.image is None:
            return None

        image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)

        # reduce noise
        image = cv2.GaussianBlur(image, (5, 5), 0)

        kernel = np.ones((5, 5), np.uint8)
        image = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)

        edges = cv2.Canny(image, 120, 350)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # create a blank mask to store the segmented objects
        mask = np.zeros_like(self.image)
        fill_color = (0, 255, 0)

        cv2.drawContours(mask, contours, -1, fill_color, 2)

        self.processed_image = mask
        self.display_image(self.processed_image)
        self.update_message(f'Approximate number of planes\non the image: {len(contours)}')

    def process_image_pixel(self):
        if self.image is None:
            return None

        image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)

        # reduce noise
        image = cv2.GaussianBlur(image, (31, 31), 0)

        blurred = cv2.GaussianBlur(image, (9, 9), 0)

        # find robust contours
        gauss_contours = blurred - image

        # opening
        erosion = cv2.erode(gauss_contours,
                            kernel=np.ones((3, 3), np.uint8),
                            iterations=2)
        dilation = cv2.dilate(erosion,
                              kernel=np.ones((2, 2), np.uint8),
                              iterations=6)
        
        # apply binary thresholding
        _, thresh = cv2.threshold(dilation, 127, 255, cv2.THRESH_BINARY)

        # find components
        _, labels, stats, _ = cv2.connectedComponentsWithStats(thresh, 8)
        labels = np.unique(labels)

        # draw rectangles for segmented objects, omitting too small objects
        mask = cv2.cvtColor(thresh, cv2.COLOR_GRAY2RGB)
        threshold_S = 10
        for i, label in enumerate(labels):
            x, y, w, h, area = stats[i]
            if area < threshold_S:
                labels[labels == label] = 0
            else:
                cv2.rectangle(mask, (x, y), (x+w, y+h), (0, 255, 0), 2)
        labels = np.unique(labels)

        self.processed_image = mask
        self.display_image(self.processed_image)
        self.update_message(f'Approximate number of planes\non the image: {len(labels) - 1}')
    
    def process_image_otsu(self):
        if self.image is None:
            return None

        image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)

        # reduce noise
        image = cv2.GaussianBlur(image, (3, 3), 0)

        se = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 40))

        bg = cv2.morphologyEx(image, cv2.MORPH_DILATE, kernel=se)

        out_gray = cv2.divide(image, bg, scale=255)

        # use otsu thresholding
        _, out_binary = cv2.threshold(out_gray, 0, 255, cv2.THRESH_OTSU)

        # erode for contours completing
        erosion = cv2.erode(out_binary, np.ones((2, 2), np.uint8), iterations=2)

        _, labels, stats, _ = cv2.connectedComponentsWithStats(erosion, 8)
        labels = np.unique(labels)

        # draw rectangles for segmented objects, omitting too small objects
        mask = cv2.cvtColor(erosion, cv2.COLOR_GRAY2RGB)
        threshold_S = 10
        for i, label in enumerate(labels):
            x, y, w, h, area = stats[i]
            if area < threshold_S:
                labels[labels == label] = 0
            else:
                cv2.rectangle(mask, (x, y), (x+w, y+h), (0, 255, 0), 2)
        labels = np.unique(labels)

        self.processed_image = mask
        self.display_image(self.processed_image)
        self.update_message(f'Approximate number of planes\non the image: {len(labels) - 1}')


def main():
    root = tk.Tk()
    app = ImageSegmentationApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()
