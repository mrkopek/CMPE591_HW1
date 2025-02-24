1. I used MLP model with turning image into a vector and trained the model.
![loss_mlp](https://github.com/user-attachments/assets/827ce1bd-0cc1-4213-a021-b171d06ddc65)

2. I used 2 conv layer for image and concatanated the output of the conv with action vector and plug it to fc layer.
![loss_cnn](https://github.com/user-attachments/assets/6c6aec37-45c0-41d9-ab67-23a4a6809cd6)

3. For the last one I used 2 conv layer and concatane the action tensor with appropriate shape. And used 2 deconv layers.
![loss_deconv](https://github.com/user-attachments/assets/40f911dc-3eab-4ae8-a44f-f41873fd018d)

Outputs of the model

![trial1](https://github.com/user-attachments/assets/0516559d-18c1-4258-b4c4-2870d0539256)
![trial2](https://github.com/user-attachments/assets/8c3afc4e-6e75-4b5e-aedd-e5864f6cb557)
![trial3](https://github.com/user-attachments/assets/1dcfe58f-ea2b-4e65-9750-49d4c84a422b)


