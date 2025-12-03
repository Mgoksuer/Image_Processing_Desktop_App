import cv2
import numpy as np
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, Toplevel
from PIL import Image, ImageTk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import os


class DIPApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Midterm Project")
        self.root.geometry("1100x700")
        self.root.configure(bg="#f0f0f0")

        self.original_img = None
        self.current_img = None
        self.history = []

        self.setup_ui()

    def setup_ui(self):
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True)

        left_panel = ttk.Frame(main_frame, width=300)
        left_panel.pack(side=tk.LEFT, fill=tk.Y, padx=5, pady=5)

        self.notebook = ttk.Notebook(left_panel)
        self.notebook.pack(fill=tk.BOTH, expand=True)

        self.tab_basics = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_basics, text="Basics")
        self.setup_basics_tab()

        self.tab_affine = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_affine, text="Affine")
        self.setup_affine_tab()

        self.tab_intensity = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_intensity, text="Intensity")
        self.setup_intensity_tab()

        self.tab_filters = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_filters, text="Filters")
        self.setup_filters_tab()

        self.tab_hist = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_hist, text="Hist")
        self.setup_hist_tab()

        self.tab_morph = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_morph, text="Morph")
        self.setup_morph_tab()

        control_frame = ttk.LabelFrame(left_panel, text="Global Controls")
        control_frame.pack(fill=tk.X, pady=10)

        ttk.Button(control_frame, text="Open Image", command=self.open_image).pack(fill=tk.X, pady=2)
        ttk.Button(control_frame, text="Save Result", command=self.save_image).pack(fill=tk.X, pady=2)
        ttk.Button(control_frame, text="Undo", command=self.undo).pack(fill=tk.X, pady=2)
        ttk.Button(control_frame, text="Reset to Original", command=self.reset).pack(fill=tk.X, pady=2)

        right_panel = ttk.Frame(main_frame)
        right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5, pady=5)

        self.lbl_original = tk.Label(right_panel, bg="gray", text="Original")
        self.lbl_original.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=2)

        self.lbl_result = tk.Label(right_panel, bg="gray", text="Processed")
        self.lbl_result.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=2)

    def setup_basics_tab(self):
        ttk.Button(self.tab_basics, text="To Grayscale", command=self.convert_grayscale).pack(fill=tk.X, pady=5)
        ttk.Button(self.tab_basics, text="Flip Horizontal", command=lambda: self.apply_flip(1)).pack(fill=tk.X, pady=5)
        ttk.Button(self.tab_basics, text="Flip Vertical", command=lambda: self.apply_flip(0)).pack(fill=tk.X, pady=5)

    def setup_affine_tab(self):
        frame_rot = ttk.LabelFrame(self.tab_affine, text="Rotation")
        frame_rot.pack(fill=tk.X, pady=2)
        self.entry_angle = ttk.Entry(frame_rot)
        self.entry_angle.insert(0, "45")
        self.entry_angle.pack(side=tk.LEFT, expand=True, fill=tk.X)
        ttk.Button(frame_rot, text="Apply", command=self.apply_rotate).pack(side=tk.RIGHT)

        frame_scale = ttk.LabelFrame(self.tab_affine, text="Scale (X, Y)")
        frame_scale.pack(fill=tk.X, pady=2)
        self.entry_sx = ttk.Entry(frame_scale, width=5)
        self.entry_sx.insert(0, "0.5")
        self.entry_sx.pack(side=tk.LEFT, padx=2)
        self.entry_sy = ttk.Entry(frame_scale, width=5)
        self.entry_sy.insert(0, "0.5")
        self.entry_sy.pack(side=tk.LEFT, padx=2)
        ttk.Button(frame_scale, text="Apply", command=self.apply_scale).pack(side=tk.RIGHT)

        frame_trans = ttk.LabelFrame(self.tab_affine, text="Translate (dx, dy)")
        frame_trans.pack(fill=tk.X, pady=2)
        self.entry_dx = ttk.Entry(frame_trans, width=5)
        self.entry_dx.insert(0, "50")
        self.entry_dx.pack(side=tk.LEFT, padx=2)
        self.entry_dy = ttk.Entry(frame_trans, width=5)
        self.entry_dy.insert(0, "30")
        self.entry_dy.pack(side=tk.LEFT, padx=2)
        ttk.Button(frame_trans, text="Apply", command=self.apply_translate).pack(side=tk.RIGHT)

        frame_shear = ttk.LabelFrame(self.tab_affine, text="Shear (X, Y)")
        frame_shear.pack(fill=tk.X, pady=2)
        self.entry_shx = ttk.Entry(frame_shear, width=5)
        self.entry_shx.insert(0, "0.2")
        self.entry_shx.pack(side=tk.LEFT, padx=2)
        self.entry_shy = ttk.Entry(frame_shear, width=5)
        self.entry_shy.insert(0, "0.0")
        self.entry_shy.pack(side=tk.LEFT, padx=2)
        ttk.Button(frame_shear, text="Apply", command=self.apply_shear).pack(side=tk.RIGHT)

    def setup_intensity_tab(self):
        ttk.Button(self.tab_intensity, text="Contrast Stretching", command=self.apply_contrast).pack(fill=tk.X, pady=5)
        ttk.Button(self.tab_intensity, text="Negative", command=self.apply_negative).pack(fill=tk.X, pady=5)

        lbl_gamma = ttk.LabelFrame(self.tab_intensity, text="Gamma Correction")
        lbl_gamma.pack(fill=tk.X, pady=5)
        self.scale_gamma = tk.Scale(lbl_gamma, from_=1, to=50, orient=tk.HORIZONTAL)
        self.scale_gamma.set(10)
        self.scale_gamma.pack(fill=tk.X)
        ttk.Button(lbl_gamma, text="Apply Gamma", command=self.apply_gamma).pack(fill=tk.X)

    def setup_filters_tab(self):
        self.combo_filters = ttk.Combobox(self.tab_filters, values=[
            "Mean/Box", "Gaussian", "Median", "Laplacian", "Sobel X", "Sobel Y"
        ])
        self.combo_filters.current(0)
        self.combo_filters.pack(fill=tk.X, pady=5)

        tk.Label(self.tab_filters, text="Kernel Size:").pack(anchor="w")
        self.spin_kernel = ttk.Spinbox(self.tab_filters, from_=1, to=31, increment=2)
        self.spin_kernel.set(3)
        self.spin_kernel.pack(fill=tk.X, pady=5)

        ttk.Button(self.tab_filters, text="Apply Filter", command=self.apply_filter).pack(fill=tk.X, pady=5)

    def setup_hist_tab(self):
        ttk.Button(self.tab_hist, text="Show Histogram", command=self.show_histogram).pack(fill=tk.X, pady=5)
        ttk.Button(self.tab_hist, text="Histogram Equalization", command=self.apply_equalization).pack(fill=tk.X,
                                                                                                       pady=5)

    def setup_morph_tab(self):
        frame_thresh = ttk.LabelFrame(self.tab_morph, text="Thresholding")
        frame_thresh.pack(fill=tk.X, pady=5)

        self.combo_thresh = ttk.Combobox(frame_thresh, values=["Otsu", "Manual Global"])
        self.combo_thresh.current(0)
        self.combo_thresh.pack(fill=tk.X)

        self.scale_thresh = tk.Scale(frame_thresh, from_=0, to=255, orient=tk.HORIZONTAL)
        self.scale_thresh.set(127)
        self.scale_thresh.pack(fill=tk.X)
        ttk.Button(frame_thresh, text="Apply Threshold", command=self.apply_threshold).pack(fill=tk.X)

        frame_ops = ttk.LabelFrame(self.tab_morph, text="Operations")
        frame_ops.pack(fill=tk.X, pady=5)
        self.combo_morph_op = ttk.Combobox(frame_ops, values=["Erosion", "Dilation", "Opening", "Closing"])
        self.combo_morph_op.current(0)
        self.combo_morph_op.pack(fill=tk.X)

        tk.Label(frame_ops, text="Kernel Size:").pack(anchor="w")
        self.spin_morph_k = ttk.Spinbox(frame_ops, from_=1, to=31, increment=2)
        self.spin_morph_k.set(3)
        self.spin_morph_k.pack(fill=tk.X)

        ttk.Button(frame_ops, text="Apply Morphology", command=self.apply_morphology).pack(fill=tk.X)

    def save_state(self):
        if self.current_img is not None:
            self.history.append(self.current_img.copy())

    def undo(self):
        if self.history:
            self.current_img = self.history.pop()
            self.update_display()

    def reset(self):
        if self.original_img is not None:
            self.save_state()
            self.current_img = self.original_img.copy()
            self.update_display()

    def open_image(self):
        path = filedialog.askopenfilename(filetypes=[("Images", "*.png;*.jpg;*.jpeg;*.bmp;*.tif")])
        if path:
            try:
                data = np.fromfile(path, dtype=np.uint8)
                img = cv2.imdecode(data, cv2.IMREAD_COLOR)
                if img is None: raise Exception("Decode Error")
                self.original_img = img
                self.current_img = img.copy()
                self.history = []
                self.update_display()
            except Exception:
                messagebox.showerror("Error", "Could not load image")

    def save_image(self):
        if self.current_img is None: return
        path = filedialog.asksaveasfilename(defaultextension=".png", filetypes=[("PNG", "*.png"), ("JPG", "*.jpg")])
        if path:
            ext = os.path.splitext(path)[1]
            is_success, buf = cv2.imencode(ext, self.current_img)
            if is_success:
                buf.tofile(path)

    def update_display(self):
        if self.original_img is not None:
            self.display_image(self.original_img, self.lbl_original)
        if self.current_img is not None:
            self.display_image(self.current_img, self.lbl_result)

    def display_image(self, img, label):
        parent = label.master
        h = parent.winfo_height()
        w = parent.winfo_width()

        if h < 10 or w < 10:
            h, w = 400, 800

        w = (w // 2) - 10
        h = h - 10

        if w < 1: w = 1
        if h < 1: h = 1

        if len(img.shape) == 2:
            rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        else:
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        ih, iw = rgb.shape[:2]
        scale = min(w / iw, h / ih)
        nw, nh = int(iw * scale), int(ih * scale)

        if nw < 1: nw = 1
        if nh < 1: nh = 1

        resized = cv2.resize(rgb, (nw, nh))

        im_pil = Image.fromarray(resized)
        im_tk = ImageTk.PhotoImage(im_pil)
        label.config(image=im_tk)
        label.image = im_tk

    def get_gray(self):
        if len(self.current_img.shape) == 3:
            return cv2.cvtColor(self.current_img, cv2.COLOR_BGR2GRAY)
        return self.current_img.copy()

    def convert_grayscale(self):
        if self.current_img is None: return
        self.save_state()
        self.current_img = self.get_gray()
        self.update_display()

    def apply_flip(self, code):
        if self.current_img is None: return
        self.save_state()
        self.current_img = cv2.flip(self.current_img, code)
        self.update_display()

    def apply_rotate(self):
        if self.current_img is None: return
        try:
            angle = float(self.entry_angle.get())
            self.save_state()
            h, w = self.current_img.shape[:2]
            M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
            self.current_img = cv2.warpAffine(self.current_img, M, (w, h))
            self.update_display()
        except ValueError:
            pass

    def apply_scale(self):
        if self.current_img is None: return
        try:
            sx = float(self.entry_sx.get())
            sy = float(self.entry_sy.get())
            self.save_state()
            self.current_img = cv2.resize(self.current_img, None, fx=sx, fy=sy)
            self.update_display()
        except ValueError:
            pass

    def apply_translate(self):
        if self.current_img is None: return
        try:
            dx = float(self.entry_dx.get())
            dy = float(self.entry_dy.get())
            self.save_state()
            h, w = self.current_img.shape[:2]
            M = np.float32([[1, 0, dx], [0, 1, dy]])
            self.current_img = cv2.warpAffine(self.current_img, M, (w, h))
            self.update_display()
        except ValueError:
            pass

    def apply_shear(self):
        if self.current_img is None: return
        try:
            shx = float(self.entry_shx.get())
            shy = float(self.entry_shy.get())
            self.save_state()
            h, w = self.current_img.shape[:2]
            M = np.float32([[1, shx, 0], [shy, 1, 0]])
            self.current_img = cv2.warpAffine(self.current_img, M, (int(w + h * shx), int(h + w * shy)))
            self.update_display()
        except ValueError:
            pass

    def apply_contrast(self):
        if self.current_img is None: return
        self.save_state()
        img = self.current_img
        min_v, max_v = img.min(), img.max()
        if max_v > min_v:
            self.current_img = ((img - min_v) * (255.0 / (max_v - min_v))).astype(np.uint8)
        self.update_display()

    def apply_negative(self):
        if self.current_img is None: return
        self.save_state()
        self.current_img = 255 - self.current_img
        self.update_display()

    def apply_gamma(self):
        if self.current_img is None: return
        self.save_state()
        gamma = self.scale_gamma.get() / 10.0
        inv_gamma = 1.0 / (gamma + 1e-6)
        table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
        self.current_img = cv2.LUT(self.current_img, table)
        self.update_display()

    def apply_filter(self):
        if self.current_img is None: return
        self.save_state()
        ftype = self.combo_filters.get()
        k = int(self.spin_kernel.get())

        if ftype == "Mean/Box":
            self.current_img = cv2.blur(self.current_img, (k, k))
        elif ftype == "Gaussian":
            self.current_img = cv2.GaussianBlur(self.current_img, (k, k), 0)
        elif ftype == "Median":
            self.current_img = cv2.medianBlur(self.current_img, k)
        elif ftype == "Laplacian":
            gray = self.get_gray()
            self.current_img = cv2.convertScaleAbs(cv2.Laplacian(gray, cv2.CV_64F))
        elif ftype == "Sobel X":
            gray = self.get_gray()
            self.current_img = cv2.convertScaleAbs(cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=k))
        elif ftype == "Sobel Y":
            gray = self.get_gray()
            self.current_img = cv2.convertScaleAbs(cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=k))
        self.update_display()

    def show_histogram(self):
        if self.current_img is None: return
        win = Toplevel(self.root)
        win.title("Histogram")
        fig, ax = plt.subplots(figsize=(5, 4))

        if len(self.current_img.shape) == 2:
            ax.hist(self.current_img.ravel(), 256, [0, 256], color='k')
        else:
            colors = ('b', 'g', 'r')
            for i, col in enumerate(colors):
                hist = cv2.calcHist([self.current_img], [i], None, [256], [0, 256])
                ax.plot(hist, color=col)
        ax.set_xlim([0, 256])

        canvas = FigureCanvasTkAgg(fig, master=win)
        canvas.draw()
        canvas.get_tk_widget().pack()

    def apply_equalization(self):
        if self.current_img is None: return
        self.save_state()
        if len(self.current_img.shape) == 2:
            self.current_img = cv2.equalizeHist(self.current_img)
        else:
            ycrcb = cv2.cvtColor(self.current_img, cv2.COLOR_BGR2YCrCb)
            ycrcb[:, :, 0] = cv2.equalizeHist(ycrcb[:, :, 0])
            self.current_img = cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)
        self.update_display()

    def apply_threshold(self):
        if self.current_img is None: return
        self.save_state()
        gray = self.get_gray()
        mode = self.combo_thresh.get()
        if mode == "Otsu":
            _, self.current_img = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        else:
            val = self.scale_thresh.get()
            _, self.current_img = cv2.threshold(gray, val, 255, cv2.THRESH_BINARY)
        self.update_display()

    def apply_morphology(self):
        if self.current_img is None: return
        self.save_state()
        op = self.combo_morph_op.get()
        k = int(self.spin_morph_k.get())
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k, k))

        if len(self.current_img.shape) == 3:
            img_to_proc = self.get_gray()
        else:
            img_to_proc = self.current_img

        if op == "Erosion":
            self.current_img = cv2.erode(img_to_proc, kernel, iterations=1)
        elif op == "Dilation":
            self.current_img = cv2.dilate(img_to_proc, kernel, iterations=1)
        elif op == "Opening":
            self.current_img = cv2.morphologyEx(img_to_proc, cv2.MORPH_OPEN, kernel)
        elif op == "Closing":
            self.current_img = cv2.morphologyEx(img_to_proc, cv2.MORPH_CLOSE, kernel)
        self.update_display()

if __name__ == "__main__":
    root = tk.Tk()
    app = DIPApp(root)
    root.mainloop()