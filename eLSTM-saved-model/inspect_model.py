import torch

model_path = "eLSTM-saved-model/best_model.pt"
checkpoint = torch.load(model_path, map_location=torch.device('cpu'))

print("Model Keys:", checkpoint.keys())
print("Model State Dict Keys:", checkpoint['model_state_dict'].keys())

embedding_weight = checkpoint['model_state_dict'].get('embedding.weight')
if embedding_weight is not None:
    print("Embedding Weight Shape:", embedding_weight.shape)
