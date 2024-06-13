# HEIG_ARN_Projet



### Data Collection Process

Because we couldn't simply use  `bing-image-downloader` for our model application, we had to manually collect code samples. This data collection process presented some challenges that needed to be solved to ensure the quality of our dataset for training our CNN model.

The first step was to gather 100 code samples for each of the three programming languages: C++, Python, and Haskell. We sourced these samples from GitHub repositories, which was extremely time-consuming. Each sample needed to be representative of the language's syntax and structure to ensure a comprehensive dataset.

To standardize the visual representation of the code snippets, we developed a script to generate Carbon configurations. Carbon is an online tool that allows to create images of source code. The script was designed to vary syntax highlighting themes, font families, and other parameters, reducing the risk of bias in the model's feature extraction process. This was important to be sure that the model learns to recognize the "essence" of the code rather than visual styles. Applying the configurations and generating images was a manual process. Each of the 300 code samples had to be pasted into Carbon with the specified configurations and saved individually. This manual effort was also very time-consuming.

Initially, we encountered challenges related to the image dimensions required by MobileNet, the CNN architecture chosen for transfer learning. MobileNet mandates input images of 224x224 pixels, whereas our initial images were significantly larger, necessitating resizing. Our first attempt was to crop the images to 672x672 pixels and then resizing them to 224x224 pixels. However, the accuracy of our model was very low, indicating that this method was not effective.

Then, we attempted to apply data augmentation to our image set, because we though that the model's low accuracy was due to a lack of data samples. Despite several attempts to fine-tune the model, we concluded that the primary issue was the low quality of the resized images rather than the quantity of data samples.

To address the quality issues, we found a solution that involved dividing the original images into 448x448 pixel tiles. This helped us to maintain higher resolution within smaller sections of the images. We used OpenCV to calculate the density of code (with canny edge detection) within each tile, selecting the most representative tiles based on this metric. This method not only preserved image quality but also ensured that the most important parts of the code snippets were used for training the model.