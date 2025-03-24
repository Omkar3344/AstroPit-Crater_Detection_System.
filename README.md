# Detection of Martian/Lunar Craters

The "Detection of Martian/Lunar Craters" project marks a significant advancement at the intersection of computer vision and planetary science. This project's primary objective is deep learning-based crater detection, complemented by streamlined deployment through FastAPI and Docker. This approach establishes a robust foundation for an efficient and accurate methodology for identifying craters on planetary surfaces.

## Key Components

- **Model Selection:** The choice of YOLOv8 as the primary model was pivotal. YOLOv8 was selected for its advanced architecture, anchor-free approach, balanced accuracy, and speed. Its adaptability to various crater configurations and efficiency in processing extensive datasets make it an excellent choice for accurate crater detection.

  ![image](https://github.com/Paciferousprerana/Martian_Lunar_Crater_Detection/assets/39224404/9b26bde3-1e52-4f56-b531-7a632c1a0493)

- **Exploratory Data Analysis (EDA):** EDA played a crucial role in understanding dataset intricacies, ensuring data quality, and guiding model development. It uncovered valuable insights and contributed to refining the model.

- **Data Augmentation:** Addressing the imbalance between small and large craters in the training dataset, data augmentation played a key role in enhancing model performance.

- **Model Training:** The model underwent training across multiple epochs, optimizing performance through precision, recall, mAP@50, and IOU metrics, resulting in significant improvements.

  ![image](https://github.com/Paciferousprerana/Martian_Lunar_Crater_Detection/assets/39224404/2277d3de-774d-426d-822d-0ef708a3fb2d)

  ![image](https://github.com/Paciferousprerana/Martian_Lunar_Crater_Detection/assets/39224404/ec7797f2-6671-44b9-8b2d-c48908cd7c59)
                                      |
  ![image](https://github.com/Paciferousprerana/Martian_Lunar_Crater_Detection/assets/39224404/9c34ca2e-8cef-48c4-93e9-eded0e8e9e52)


## Deployment

- The deployment phase was a critical milestone. FastAPI enabled real-time predictions, allowing rapid crater identification during space missions.

- Docker containerization improved scalability, streamlining the deployment process and expanding the model's accessibility to a broader audience.

  ![image](https://github.com/Paciferousprerana/Martian_Lunar_Crater_Detection/assets/39224404/6dba5e1a-ee54-47d4-a269-ab3f95d00a9c)
  ![image](https://github.com/Paciferousprerana/Martian_Lunar_Crater_Detection/assets/39224404/5d0cb7bd-8b5d-4ffc-b327-39d3979ffbad)


## Conclusion

In summary, the "Detection of Martian/Lunar Craters" project exemplifies the seamless integration of technology and planetary research. Through the careful selection of the YOLOv8 model and the implementation of data augmentation techniques, this project has paved the way for precise and effective crater detection. Its practical applications, especially in space research missions, hold promise for enhancing our understanding of the geological history of celestial bodies. This initiative underscores the strong connection between planetary science and machine learning, opening exciting new avenues for exploration.

This project represents a significant step forward in advancing our knowledge of planetary surfaces, demonstrating the profound impact of advanced technology in scientific research.

