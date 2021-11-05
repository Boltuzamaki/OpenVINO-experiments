### Steps 
- First follow [this](https://docs.openvino.ai/latest/openvino_docs_install_guides_installing_openvino_windows.html)


- Then open cmd as administrator 

- Create a virtual env 
```
python -m venv openvinotf2
```
- Activate env 

```
./openvinotf2/Scripts/activate.bat
````
- Set variables/PATH 
```
C:\openvino\openvino_2021\deployment_tools\model_optimizer\install_prerequisites\openvinotf2\Scripts\activate.bat
```
- Then install and setup environment for openvino tf2
```
C:\openvino\openvino_2021\deployment_tools\model_optimizer\install_prerequisites\install_prerequisites_tf2.bat
```
- Install dependencies 
```
pip install tensorflow==2.4.1
pip install networkx==2.5
pip install defusedxml==0.7.1
pip install requests==2.25.1
```
- Then donwload the .h5 trained model and conevert it into save model format 
```
import tensorflow as tf
model = tf.keras.models.load_model('vgg16_naive_model.h5')
tf.saved_model.save(model,'/content/model')
``` 
- Run the model optimizer script
```
python mo_tf.py --saved_model_dir "./experiments/model" --output_dir "./experiments/new"  --input_shape=(1,224,224,3)
```