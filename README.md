# Cell Segmentation using Transfer Learning and Feed Forward Networks

## Description
Accurate segmentation of cells plays a crucial role in understanding cellular structures and functions, particularly in biological research. Leveraging computer vision techniques can extract valuable insights from images, aiding in the comprehension of cellular properties and interactions. The retinal pigment cells, vital for nutrient transport to the retina and aging studies, offer a prime example of the significance of precise cell segmentation.

This repository presents a novel approach to cell segmentation, emphasizing efficiency and accuracy through the fusion of transfer learning and feed forward neural networks. Unlike traditional encoder-decoder architectures, this method employs a unique design comprising a combination of U-Net architecture and convolutional neural networks (CNNs).

### Key Features
1. **Transfer Learning with Instance Segmentation:** The initial layer utilizes transfer learning from an instance segmentation model to generate masks, laying the groundwork for subsequent segmentation.
2. **Feed Forward Network Architecture:** Instead of the conventional encoder-decoder setup, a feed forward network architecture is adopted, optimizing computational efficiency while preserving accuracy.
3. **Improved Precision and Intersection over Union (IoU):** Comparative analysis against the S4 model showcases superior performance, achieving precision and IoU scores of 86.6 and 85.1, respectively.

## Dataset
The dataset utilized for training and evaluation is available [here](https://drive.google.com/drive/u/0/folders/1a6Hb9demcZpm3Bp1Pbs5ef7-nosRWNcv). It includes a comprehensive collection of images necessary for robust model training and validation.

## Usage
1. **Data Preprocessing:** Ensure the dataset is appropriately preprocessed, including resizing, normalization, and augmentation if necessary.
2. **Model Training:** Execute the provided code to train the feed forward network architecture, utilizing the instance segmentation masks generated through transfer learning.
3. **Evaluation:** Assess the model's performance metrics, including precision and IoU, to validate segmentation accuracy.

## Results
![img](https://github.com/codersid19/Biomed_ML_Final-Project/assets/67604975/6ee7e091-773e-4f03-b892-03ea02df4bab)
![img2](https://github.com/codersid19/Biomed_ML_Final-Project/assets/67604975/a9e2db92-7ad4-426f-a99f-869c7e74b2a7)
![imgb](https://github.com/codersid19/Biomed_ML_Final-Project/assets/67604975/dfed1dad-374f-4fa4-a2b6-20c77eae63e2)

## Contributing
Contributions to this project are welcome. Whether it's enhancing model performance, optimizing code efficiency, or refining documentation, your input is valued.



## Acknowledgments
We extend our gratitude to the creators of the instance segmentation model and the contributors to the open-source libraries utilized in this project. Their efforts have significantly contributed to the advancement of computer vision techniques in biological research. This is based on paper Self-supervised semantic segmentation of retinal pigment epithelium cells in flat-mount fluorescent microscopy images

## Questions and Support
For any queries or assistance regarding this project, please feel free to open an issue or reach out to the maintainers directly.
