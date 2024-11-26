# Real-Time-Human-Pose-Detection-with-YOLOv8-and-Python-Amazing-Keypoint-Visualizations-

This project demonstrates real-time human pose detection using the YOLOv8 pose model, OpenCV, and Python. The program identifies human keypoints in a video, connects them with labeled lines, and displays a visually appealing output for better understanding and analysis.  

## **Features**
- Detects 17 keypoints on the human body.
- Draws labeled keypoints with confidence thresholds.
- Connects keypoints with lines to form a skeleton-like structure.
- Real-time visualization of detected poses.
- Outputs a side-by-side comparison of the original frame and keypoint visualization.



## **Demo**
![Pose Detection Example]([https://github.com/your-username/pose-detection-project/assets/demo-image.gif](https://github.com/Tech-Watt/Real-Time-Human-Pose-Detection-with-YOLOv8-and-Python-Amazing-Keypoint-Visualizations))  


## **Technologies Used**
- **YOLOv8 Pose Model**: For state-of-the-art human pose detection.  
- **OpenCV**: For image processing and visualization.  
- **cvzone**: To stack and display images side by side.  
- **NumPy**: For numerical calculations.  


   
Code Explanation
Keypoint Detection: Extracts coordinates and confidence levels for 17 keypoints on the human body.
Visualization:
Circles are drawn on keypoints with confidence > 0.7.
Lines connect specific keypoints to create a skeletal structure.
Side-by-Side Display: Combines the original video frame with the pose-detection overlay using cvzone.stackImages().

Connections Between Keypoints
The program draws the following connections:

Head: Ear, eye, and nose points.
Arms: Shoulder, elbow, and wrist points.
Legs: Hip, knee, and ankle points.
Body: Shoulder and hip connections.


Future Improvements
Support for live webcam feed.
Pose classification (e.g., identifying actions like walking, running, etc.).
Export results as video or images.
Improved user interface for better interaction.


Contributing
Contributions are welcome! Feel free to submit a pull request or open an issue to improve this project.

License
This project is licensed under the MIT License. See the LICENSE file for more details.

Acknowledgments
Ultralytics YOLOv8
OpenCV
cvzone
