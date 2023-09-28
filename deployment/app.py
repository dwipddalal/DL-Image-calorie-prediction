from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS  
import torch
from PIL import Image
import torch.nn as nn
from torchvision import transforms
from torchvision import transforms, models

app = Flask(__name__, static_url_path='/static')
CORS(app)

# Load Model
resnet_model = models.resnet18(pretrained=True)
num_ftrs = resnet_model.fc.in_features
resnet_model.fc = nn.Linear(num_ftrs, 3)

best_model_path = 'best_model.pth'
resnet_model.load_state_dict(torch.load(best_model_path))
resnet_model.eval()

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        file = request.files['file']
        img = Image.open(file.stream).convert("RGB")
        
        # Perform transformations and inference
        transform = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor()
        ])
        
        img = transform(img).unsqueeze(0)
        output = resnet_model(img)
        
        _, predicted = torch.max(output.data, 1)
        
        print(predicted.item())
        if predicted.item() == 0:
            print('FrenchFries')
            return jsonify({'result': 'FrenchFries. It has 142.5 kcal.'})
        elif predicted.item() == 1:
            print('Pizza')
            return jsonify({'result': "Pizza. It has 2298.75 kcal, that's too much!"})
        else:
            print('VegBurger')
            return jsonify({'result': 'VegBurger. It has 277.0 kcal.'})

@app.route('/static/<path:path>')
def send_static(path):
    return send_from_directory('static', path)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
