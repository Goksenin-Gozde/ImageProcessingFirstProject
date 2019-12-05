import tkinter as tk
from tkinter import *
from tkinter import filedialog

import cv2
from skimage import data

import Filters as filter
import Transforms as transforms
import colorFilteredVideo as video
import intensityOperations as intensity
import morphologyOperations as morphology


class Root(Tk):

    def __init__(self):
        super(Root, self).__init__()
        self.title("Image Processor")
        self.minsize(640, 400)
        self.img = data.coins()
        self.img_gray = data.coins()
        self.retina = data.retina()
        self.moon = data.moon()
        self.horse = data.horse()
        self.camera = data.camera()
        self.width = 50

        self.labelFrame = tk.LabelFrame(self, text="Open File")
        self.labelFrame.grid(column=0, row=1, padx=20, pady=20)

        self.filterFrame = tk.LabelFrame(self, text="Filters")
        self.filterFrame.grid(column=1, row=1, padx=20, pady=20)

        self.histogramFrame = tk.LabelFrame(self, text="Histogram Matching")
        self.histogramFrame.grid(column=2, row=1, padx=20, pady=20)

        self.transformFrame = tk.LabelFrame(self, text="Transform Operations")
        self.transformFrame.grid(column=0, row=2, padx=20, pady=20)

        self.videoFrame = tk.LabelFrame(self, text="Green Filtered Video from your cam")
        self.videoFrame.grid(column=1, row=2, padx=20, pady=20)

        self.intensityFrame = tk.LabelFrame(self, text="Intensity Operations")
        self.intensityFrame.grid(column=2, row=2, padx=20, pady=20)

        self.morphologyFrame = tk.LabelFrame(self, text="Morphological Operations")
        self.morphologyFrame.grid(column=0, row=3, padx=20, pady=20)

        self.upload_image_button()
        self.filterButton()
        self.histogramButton()
        self.transformButton()
        self.videoButton()
        self.intensityButton()
        self.morphologyButton()

    def morphologyButton(self):
        self.morphology_button = tk.Button(self.morphologyFrame, text="Open Morphology Window", width=self.width,
                                           command=self.open_morphology_window)
        self.morphology_button.grid(column=0, row=3)

    def intensityButton(self):
        self.intensity_button = tk.Button(self.intensityFrame, text="Open Intensity Window", width=self.width,
                                          command=self.open_intensity_window)
        self.intensity_button.grid(column=2, row=2)

    def videoButton(self):
        self.video_button = tk.Button(self.videoFrame, text="Open your cam (Press esc to exit) ", width=self.width,
                                      command=lambda: video.main())
        self.video_button.grid(column=1, row=2)

    def transformButton(self):
        self.transform_button = tk.Button(self.transformFrame, text="Open transform operations window",
                                          width=self.width,
                                          command=self.open_transform_operations)
        self.transform_button.grid(column=0, row=2)

    def histogramButton(self):
        self.filter_button = tk.Button(self.histogramFrame, text='An Example for histogram matching', width=self.width,
                                       command=lambda: filter.histogram_matching())
        self.filter_button.grid(column=2, row=2)

    def filterButton(self):
        self.filter_button = tk.Button(self.filterFrame, text='Open filter options', width=self.width,
                                       command=self.open_filter_window)
        self.filter_button.grid(column=1, row=1)

    def upload_image_button(self):
        self.button = tk.Button(self.labelFrame, text="Browse An Image", width=self.width, command=self.fileDialog)
        self.button.grid(column=1, row=1)

    def fileDialog(self):
        self.filename = filedialog.askopenfilename(initialdir="/", title="Select An Image", filetype=
        (("jpeg files", "*.jpg"), ("all files", "*.*")))
        self.label = tk.Label(self.labelFrame, text="")
        self.label.grid(column=1, row=2)
        if not (self.filename is NONE):
            self.label.configure(text=self.filename)
            self.img = cv2.imread(self.filename)
            self.img = cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB)
            self.img_gray = cv2.imread(self.filename, 0)

    def open_transform_operations(self):
        transform_window = tk.Tk(screenName=None, baseName=None, className='Tk', useTk=1)
        transform_window.title('Transform Operations')
        transform_window.geometry("400x400")

        swirl_button = tk.Button(transform_window, text='Swirl Operation with custom image', width=self.width,
                                 command=lambda: transforms.swirled(self.img))
        swirl_button.pack()

        swirl_with_checker_board_button = tk.Button(transform_window, text='Swirl Operation with checker board',
                                                    width=self.width,
                                                    command=lambda: transforms.swirled_with_checkerboard())
        swirl_with_checker_board_button.pack()

        rescale_button = tk.Button(transform_window, text="Rescale with anti anti aliasing", width=self.width,
                                   command=lambda: transforms.rescale(self.img))
        rescale_button.pack()

        resize_button = tk.Button(transform_window, text="Resize Operation with anti aliasing", width=self.width,
                                  command=lambda: transforms.resize(self.img))
        resize_button.pack()

        downscale_button = tk.Button(transform_window, text='Downscale Operation', width=self.width,
                                     command=lambda: transforms.downscale(self.img_gray))
        downscale_button.pack()

        rotation_button = tk.Button(transform_window, text="Rotation Operation", width=self.width,
                                    command=lambda: transforms.rotation(self.img))
        rotation_button.pack()

    def open_intensity_window(self):
        intensity_window = tk.Tk(screenName=None, baseName=None, className='Tk', useTk=1)
        intensity_window.title('Intensity Operations')
        width = 50
        tk.Label(intensity_window, width=int(width / 2), text="Out_Range Value").grid(row=0)

        label_value = tk.Entry(intensity_window)
        label_value.grid(row=0, column=1)
        out_range = label_value.get()

        if (out_range is not float and out_range is not int):
            out_range = 0.4

        wrapping_button = tk.Button(intensity_window, text="Wrapping Operation", width=int(width / 2),
                                    command=lambda: intensity.main(out_range))
        wrapping_button.grid(row=1)

    def open_morphology_window(self):
        morphology_window = tk.Tk(screenName=None, baseName=None, className='Tk', useTk=1)
        morphology_window.title('Morphology Operations')
        morphology_window.geometry("400x400")
        width = 50

        erosion_button = tk.Button(morphology_window, text="Erosion Operation", width=width,
                                   command=lambda: morphology.erosion(self.moon))
        erosion_button.pack()

        dilation_button = tk.Button(morphology_window, text="Dilation Operation", width=width,
                                    command=lambda: morphology.dilation(self.moon))
        dilation_button.pack()

        opening_button = tk.Button(morphology_window, text="Opening Operation", width=width,
                                   command=lambda: morphology.opening(self.moon))
        opening_button.pack()

        closing_button = tk.Button(morphology_window, text="Closing Operation", width=width,
                                   command=lambda: morphology.closing(self.moon))
        closing_button.pack()

        white_tophat_button = tk.Button(morphology_window, text="White Tophat Operation", width=width,
                                        command=lambda: morphology.white_tophat(self.moon))
        white_tophat_button.pack()

        black_tophat_button = tk.Button(morphology_window, text="Black Tophat Operation", width=width,
                                        command=lambda: morphology.black_tophat(self.moon))
        black_tophat_button.pack()

        skeletonize_button = tk.Button(morphology_window, text="Skeletonize Operation", width=width,
                                       command=lambda: morphology.skeletonize(self.horse))
        skeletonize_button.pack()

        convex_hull_image_button = tk.Button(morphology_window, text="Convex Hull Operation", width=width,
                                             command=lambda: morphology.convex_hull_image(self.horse))
        convex_hull_image_button.pack()

        watershed_button = tk.Button(morphology_window, text="Watershed Operation", width=width,
                                     command=lambda: morphology.watershed(self.camera))
        watershed_button.pack()

        entropy_button = tk.Button(morphology_window, text="Entropy Operation", width=width,
                                   command=lambda: morphology.entropy(self.camera))
        entropy_button.pack()

    def open_filter_window(self):

        filter_window = tk.Tk(screenName=None, baseName=None, className='Tk', useTk=1)
        filter_window.title('Filters')
        filter_window.geometry("400x400")
        width = 50

        sobel_h_button = tk.Button(filter_window, text='Sobel_h Filter', width=width,
                                   command=lambda: filter.sobel_h_filter(self.img_gray))
        sobel_h_button.pack()

        region_boundry_button = tk.Button(filter_window, text='Region Boundry Filter', width=width,
                                          command=lambda: filter.region_boundry(self.img_gray))
        region_boundry_button.pack()

        try_all_threshold_button = tk.Button(filter_window, text='Try all thresholds', width=width,
                                             command=lambda: filter.try_all_threshold_filter(self.img_gray))
        try_all_threshold_button.pack()

        roberts_button = tk.Button(filter_window, text='Roberts Filter', width=width,
                                   command=lambda: filter.roberts_filter(self.img_gray))
        roberts_button.pack()

        ridge_operation_button = tk.Button(filter_window,
                                           text='Ridge Operations \n ( uses retina photo to show filter clearly)',
                                           width=width, command=lambda: filter.ridge_operations(self.retina))
        ridge_operation_button.pack()

        hysteresis_threshold_button = tk.Button(filter_window, text='Hysteresis Threshold', width=width,
                                                command=lambda: filter.hysteresis_threshold(self.img_gray))
        hysteresis_threshold_button.pack()

        median_filter_button = tk.Button(filter_window, text='Median Filter', width=width,
                                         command=lambda: filter.median_filter(self.img_gray))
        median_filter_button.pack()

        scharr_filter_button = tk.Button(filter_window, text='Scharr Filter', width=width,
                                         command=lambda: filter.scharr_filter(self.img_gray))
        scharr_filter_button.pack()

        segmentation_and_superpixel_algorithms_button = tk.Button(filter_window, text = "Comparison of segmentation and superpixel algorithms", width= width ,
                                                                  command = lambda: filter.segmentation_and_superpixel_algorithms())
        segmentation_and_superpixel_algorithms_button.pack()

        find_regular_segments_button = tk.Button(filter_window, text='Find Regular Segments', width=width,
                                                 command=lambda: filter.find_regular_segments(self.img_gray))
        find_regular_segments_button.pack()


root = Root()
root.mainloop()
