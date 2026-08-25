#Plot Learning Curves

import matplotlib.pyplot as plt

plt.figure(figsize=(10,5))

plt.plot(
    train_losses,
    label="Train"
)

plt.plot(
    val_losses,
    label="Validation"
)

plt.xlabel("Epoch")
plt.ylabel("Loss")

plt.title(
    "Training Curves"
)

plt.legend()

plt.grid(True)

plt.show()

#Load Best Model

model.load_state_dict(
    torch.load(
        "RainVisionV1_best.pth"
    )
)

model.eval()

#Rain Occurrence Accuracy

from sklearn.metrics import (
    accuracy_score,
    classification_report
)

all_true = []
all_pred = []

with torch.no_grad():

    for (
        x_batch,
        y_occ,
        y_int
    ) in val_loader:

        x_batch = x_batch.to(device)

        pred_occ, _ = model(
            x_batch
        )

        probs = torch.sigmoid(
            pred_occ
        )

        preds = (
            probs > 0.5
        ).cpu().numpy()

        all_pred.extend(
            preds.flatten()
        )

        all_true.extend(
            y_occ.numpy()
        )

print(
    "Accuracy:",
    accuracy_score(
        all_true,
        all_pred
    )
)

print(
    classification_report(
        all_true,
        all_pred
    )
)

#Rainfall Intensity Evaluation

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

true_values = []
pred_values = []

with torch.no_grad():

    for (
        x_batch,
        y_occ,
        y_int
    ) in val_loader:

        x_batch = x_batch.to(device)

        _, pred_int = model(
            x_batch
        )

        true_values.extend(
            y_int.numpy()
        )

        pred_values.extend(
            pred_int
            .cpu()
            .numpy()
            .flatten()
        )

mae = mean_absolute_error(
    true_values,
    pred_values
)

rmse = np.sqrt(
    mean_squared_error(
        true_values,
        pred_values
    )
)

r2 = r2_score(
    true_values,
    pred_values
)

print("MAE :", mae)
print("RMSE:", rmse)
print("R2  :", r2)

#Predicted vs Actual Rainfall

plt.figure(figsize=(8,8))

plt.scatter(
    true_values,
    pred_values,
    alpha=0.3
)

plt.xlabel(
    "Observed Rainfall"
)

plt.ylabel(
    "Predicted Rainfall"
)

plt.title(
    "Prediction Skill"
)

plt.grid(True)

plt.show()




