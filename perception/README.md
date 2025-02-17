# Wisconsin Autonomous Perception Coding Challenge Submission

## Answer Image Output
Original Image|Processed Image
:-------------------------:|:-------------------------:
![](red.png)|![](answer.png)

## Submission Specification
- Your code should be written in any language, we recommend Python or C++.
- Please upload your code to a public github repository for us to review
- Please document your code. The more readable your code is the better you can show your coding skills.
- Please include a README that contains the following:
    - answer.png
    - Methodolgy 
    - What did you try and why do you think it did not work.
    - What libraries are used

## Methodolgy
I first load the image, and convert it to HSV format for better color segmentation. I then filter out the image for red using the two HSV red color ranges. I then remove random red pixels using morphological operations, and extract contours of red images to locate the cones. I then removed further removed small areas of red. After obtaining the coordinates of all cone points, I split the image into two vertical halves (to calculate the least-squares line for each line of cones), and drew the lines onto the image. Finally, I saved the image to 'answer.png'.

## Challenges
I initially tried to capture all the red colors, thinking it would only capture the cones. However, it ended up capturing parts of the door on the side, which heavily skewed the least-squares line of the left line of cones. To combat this, I ended up tweaking the HSV segments to capture only the red in the cones. While this vastly decreased the amount of non-cone reds detected, there were still splotches of red (particularely with the exit signs and its reflections on the floor) that I was unable to eliminate with just HSV thresholding. Thus, I decided to group all the red points together, and eliminate the smaller red areas (which were the reflections), which would leave the large red areas represented by cones intact. This ended up working, and I was able to isolate a mask that contained only the red points of the cones. 

## Libraries Used
- **OpenCV** (`cv2`) - To process the image and isolate the red pixels representing cones
- **NumPy** (`numpy`) - To manipulate arrays and compute aggregate functions (notable mean())
- **OS** (`os`) - To join files between different directories (to account for different operating systems)

## Repository and Execution
To run this code, clone the repository and execute:

```bash
python perception/main.py
```

This will generate `answer.png` containing the detected path boundaries.