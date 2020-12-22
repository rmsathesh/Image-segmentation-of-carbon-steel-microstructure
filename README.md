# Image-segmentation-of-carbon-steel-microstructure

This project is developed for the image segmentation of high carbon steel which consists of pearlite and spheroidite as the primary micro constituents. 'U-Net' model is developed to automatically annotate the micrograph images.

Micrograph images were obtained from UHCS database. Mentioned below
'https://materialsdata.nist.gov/handle/11256/940'

Image augmentation was used to increase the number of training samples and enhancing the model's accuracy. IoU and Dice coefficient were used as the metric to evaluate the model's performance.

The above model after training gave IoU of 81.5% and Dice-coefficient of 
