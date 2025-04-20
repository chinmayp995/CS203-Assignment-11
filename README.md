# CS203-Assignment-11
## Team 20
### Dakshata Bhamare (23210027)
### Chinmay Pendse (23110245)

## Introduction

**Github Link:** https://github.com/chinmayp995/CS203-Assignment-11

In this assignment; we Quantised the weights such that they are optimised for the devices with less memory. In this course of assignment; we have constructed Multilayer Percepttron (MLP) model and we also exploered  to  calculate the number of parameters in models and also tried to optimise them by methods from torch library like dynamic and half method. 

## 1. Dataset Preparation
![image](https://github.com/user-attachments/assets/12b38cf7-61d9-4ac3-9711-cb86416af759)

## 2. Construct MLP Model
```python
#defining MLP Model
class MLP(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size):
        super(MLP, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, hidden_sizes[0]), nn.ReLU(),
            nn.Linear(hidden_sizes[0], hidden_sizes[1]), nn.ReLU(),
            nn.Linear(hidden_sizes[1], hidden_sizes[2]), nn.ReLU(),
            nn.Linear(hidden_sizes[2], hidden_sizes[3]), nn.ReLU(),
            nn.Linear(hidden_sizes[3], output_size)
        )

    def forward(self, x):
        return self.model(x)
```

![image](https://github.com/user-attachments/assets/76de3c1e-6024-4cda-a05c-2fef336223db)

### Trainable Parameters
![image](https://github.com/user-attachments/assets/4204d4bc-404c-43de-9edb-beb839ad0999)

## 3. Training Model

```python
# Training the model
best_val_acc = 0.0
train_losses = []
val_losses = []
val_accuracies = []

for epoch in range(10):
    model.train()
    running_loss = 0.0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    train_losses.append(running_loss / len(train_loader))

    # Validation
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    val_losses.append(val_loss / len(val_loader))
    val_acc = 100 * correct / total
    val_accuracies.append(val_acc)
    print(f"Epoch {epoch+1}: Train Loss={train_losses[-1]:.4f}, Val Loss={val_losses[-1]:.4f}, Val Acc={val_acc:.2f}%")

    # this saves the best checkpoint
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), 'checkpoint.pt')
```
![image](https://github.com/user-attachments/assets/a0f7287a-3e6b-49e5-af36-5f1885c339da)

![image](https://github.com/user-attachments/assets/c7e94758-230d-4e7f-8987-ae86ffddd30c)

## 4. Using Dynamic and Half Precision Model

```python
# defining a dynamic model
model_dynamic = torch.quantization.quantize_dynamic(
    model, {nn.Linear}, dtype=torch.qint8
)

# defining a Half Precision model
model_half = MLP(input_size=5000, hidden_sizes=[512, 256, 128, 64], output_size=2).to(device)
model_half.load_state_dict(torch.load("checkpoint.pt"))
model_half = model_half.half()  #casting float16
```
## 5. Final Chart
![image](https://github.com/user-attachments/assets/96e1e136-eba7-440e-9c09-8053b53ae8f6)

## Inferences

We see that the original consumed heavy data ~10 MB is consumed for the original model which gets highly reduced  to 2Mb is case of dynamic and becomes halved for Half case.  The accuracy stays almost similar in all cases; since our data is not that big. Significant difference are seen in the inference time; where it is the lowest for the original case (.01 ms); 8 times higher for the Dynamic case and 10 times higher for the Half case.




