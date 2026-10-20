<h1 align="center">Highway Lane Detection Pipeline</h1>

<p align="center">
  <em>A robust Computer Vision pipeline implemented in MATLAB for detecting and estimating highway driving lanes using classical image processing techniques.</em>
</p>

<hr>

## Table of Contents
- [About The Project](#about-the-project)
- [Pipeline Architecture](#pipeline-architecture)
- [Visual Results](#visual-results)
- [Execution & Usage](#execution--usage)
- [Author](#author)

---

## About The Project

Autonomous driving systems heavily rely on accurate lane detection. This project implements a complete, step-by-step image processing pipeline in MATLAB to identify road lanes from raw images. Instead of using black-box neural networks, this approach utilizes core Computer Vision mathematics, ensuring high interpretability and efficiency.

---

## Pipeline Architecture

The detection algorithm follows a strict 5-step methodology:

1. **Color Segmentation & Smoothing:** Conversion to HSV color space to isolate white and yellow lane markings under varying lighting conditions. Noise reduction is applied using customizable filters (Gaussian or Median).
2. **Region of Interest (ROI):** Dynamic generation of a trapezoidal polygonal mask to isolate the road surface and ignore background elements (sky, trees, other lanes).
3. **Edge Detection:** Application of the **Canny Edge Detector** with optimized hysteresis thresholds (50/150) to extract strong geometric boundaries.
4. **Hough Transform:** Implementation of the Standard Hough Transform to find geometric lines. Irrelevant horizontal lines (slope < 0.3) are filtered out.
5. **Lane Estimation (Regression):** Grouping candidate lines into left and right boundaries based on their slope, and applying Linear Regression (`polyfit`) to estimate the final continuous lane borders.

---

## Visual Results

*(Note: Create a `plots` folder and place your output images there to visualize the pipeline steps)*

### Original vs Detected Lanes
A side-by-side comparison of the raw input and the final estimated driving path.

![Final Lane Detection Output](images/lane_output.png)

### Edge Detection & Hough Space
Visualization of the Canny edges inside the defined ROI and the extracted Hough lines before regression.

![Canny Edges and Hough Lines](images/canny_hough.png)

---

## Execution & Usage

### Prerequisites
* MATLAB installed with the **Image Processing Toolbox**.

### Running the Code
The main function `lane_detection_pipeline(img, filter_type)` takes the raw image and an optional smoothing filter as inputs. 

```matlab
% 1. Read your input image
test_img = imread('dataset/test_road.jpg');

% 2. Run the detection pipeline (Default filter: Gaussian)
% Options for filter: 'gaussian', 'median', 'none'
result_img = lane_detection_pipeline(test_img, 'gaussian');

% 3. The function automatically displays the final plotted lanes.
