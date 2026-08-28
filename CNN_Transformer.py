import numpy as np

# =====================================================
# CONFIGURATION
# =====================================================

HISTORY = 12
LEAD = 6

TARGET_LAT_MIN = 18
TARGET_LAT_MAX = 18.5

TARGET_LON_MIN = 73.5
TARGET_LON_MAX = 74.5

# =====================================================
# LOAD DATA
# =====================================================

data = np.load("/kaggle/input/datasets/aratrikroy01/rainfall/rainvision_dataset_merged_2016_2025.npz", allow_pickle=True)

X = data["X"]
y = data["y"]

feature_names = list(data["feature_names"])

latitudes = data["latitudes"]
longitudes = data["longitudes"]

print("Original X shape:", X.shape)
print("Original y shape:", y.shape)

print("\nOriginal Features:")
print(feature_names)

# =====================================================
# REMOVE GEOPOTENTIAL IF PRESENT
# =====================================================

if "z" in feature_names:
    z_idx = feature_names.index("z")

    X = np.delete(X, z_idx, axis=-1)
    feature_names.pop(z_idx)

# =====================================================
# GET FEATURE INDICES
# =====================================================

u_idx = feature_names.index("u10")
v_idx = feature_names.index("v10")

tcwv_idx = feature_names.index("tcwv")

t2m_idx = feature_names.index("t2m")
d2m_idx = feature_names.index("d2m")

# =====================================================
# EXTRACT BASE VARIABLES
# =====================================================

u10 = X[..., u_idx]
v10 = X[..., v_idx]

tcwv = X[..., tcwv_idx]

t2m = X[..., t2m_idx]
d2m = X[..., d2m_idx]

# =====================================================
# 1. WIND SPEED
# =====================================================

wind_speed = np.sqrt(u10**2 + v10**2)

# =====================================================
# 2. WIND DIRECTION radians [-pi, pi]
# =====================================================

wind_direction = np.arctan2(v10, u10)

wind_dir_sin = np.sin(wind_direction)
wind_dir_cos = np.cos(wind_direction)
# =====================================================
# 3. DEW POINT DEPRESSION
# =====================================================

dewpoint_depression = t2m - d2m

# =====================================================
# 4. MOISTURE FLUX
# =====================================================
#
# Proxy:
#
# Flux_u = TCWV * U10
# Flux_v = TCWV * V10
#
# =====================================================

moisture_flux_u = tcwv * u10
moisture_flux_v = tcwv * v10

# =====================================================
# 5. MOISTURE FLUX MAGNITUDE
# =====================================================

moisture_flux_mag = np.sqrt(
    moisture_flux_u**2 +
    moisture_flux_v**2
)

# =====================================================
# 6. IMPROVED MOISTURE FLUX CONVERGENCE / DIVERGENCE
# =====================================================

R = 6371000.0

lat_rad = np.deg2rad(latitudes)

dlat = np.deg2rad(np.mean(np.diff(latitudes)))

dlon = np.deg2rad(np.mean(np.diff(longitudes)))

dy = R * dlat

dx = (R * np.cos(lat_rad) * dlon)

dx = dx.reshape(-1, 1)

raw_dFu_dx = np.gradient(
    moisture_flux_u,
    axis=2
)

raw_dFv_dy = np.gradient(
    moisture_flux_v,
    axis=1
)

dFu_dx = raw_dFu_dx / dx[None,:,:]

dFv_dy = raw_dFv_dy / dy

moisture_flux_divergence = (dFu_dx + dFv_dy)

moisture_flux_convergence = (-moisture_flux_divergence)

# =====================================================
# 6. MOISTURE RESIDENCE PROXY
# =====================================================

moisture_residence_proxy = (tcwv * moisture_flux_mag)

new_features = np.stack(
    [
        wind_speed,
        wind_dir_sin,
        wind_dir_cos,
        dewpoint_depression,
        moisture_flux_u,
        moisture_flux_v,
        moisture_flux_mag,
        moisture_flux_convergence,
        moisture_flux_divergence,
        moisture_residence_proxy
    ],
    axis=-1
)

new_feature_names = [
    "wind_speed",
    "wind_dir_sin",
    "wind_dir_cos",
    "dewpoint_depression",
    "moisture_flux_u",
    "moisture_flux_v",
    "moisture_flux_mag",
    "moisture_flux_convergence",
    "moisture_flux_divergence",
    "moisture_residence_proxy"
]

# =====================================================
# APPEND TO ORIGINAL FEATURE SET
# =====================================================

X = np.concatenate(
    [X, new_features],
    axis=-1
)

feature_names.extend(new_feature_names)

# =====================================================
# FINAL SUMMARY
# =====================================================

print("\nEnhanced X shape:", X.shape)

print("\nFeatures After Engineering:")
for f in feature_names:
    print(f)

print("\nTotal Features =", len(feature_names))

import matplotlib.pyplot as plt

# ==========================================
# TARGET REGION
# ==========================================

lat_mask = (
    (latitudes >= TARGET_LAT_MIN) &
    (latitudes <= TARGET_LAT_MAX)
)

lon_mask = (
    (longitudes >= TARGET_LON_MIN) &
    (longitudes <= TARGET_LON_MAX)
)

y_small = y[:, lat_mask][:, :, lon_mask]

# ==========================================
# REGION AVERAGE RAINFALL
# ==========================================

rain_ts = y_small.mean(axis=(1,2))

# ==========================================
# AUTOCORRELATION
# ==========================================

max_lag = 24 * 7

autocorrs = []

for lag in range(1, max_lag + 1):

    r = np.corrcoef(
        rain_ts[:-lag],
        rain_ts[lag:]
    )[0,1]

    autocorrs.append(r)

# ==========================================
# PLOT
# ==========================================

plt.figure(figsize=(12,5))
plt.plot(range(1,max_lag+1), autocorrs)

plt.xlabel("Lag (Hours)")
plt.ylabel("Autocorrelation")
plt.title("Rainfall Autocorrelation")

plt.grid(True)

plt.show()

import torch
import torch.nn as nn
import torch.nn.functional as F

class SpatialEncoder(nn.Module):

    def __init__(self, in_channels=20):

        super().__init__()

        self.encoder = nn.Sequential(

            nn.Conv2d(
                in_channels,
                32,
                kernel_size=3,
                padding=1
            ),

            nn.BatchNorm2d(32),

            nn.ReLU(),

            nn.MaxPool2d(2),

            nn.Conv2d(
                32,
                64,
                kernel_size=3,
                padding=1
            ),

            nn.BatchNorm2d(64),

            nn.ReLU(),

            nn.MaxPool2d(2),

            nn.Conv2d(
                64,
                128,
                kernel_size=3,
                padding=1
            ),

            nn.BatchNorm2d(128),

            nn.ReLU(),

            nn.AdaptiveAvgPool2d((1,1))
        )

        self.fc = nn.Linear(
            128,
            512
        )

    def forward(self, x):

        x = self.encoder(x)

        x = x.flatten(1)

        x = self.fc(x)

        return x
class TemporalTransformer(nn.Module):

    def __init__(self):

        super().__init__()

        encoder_layer = nn.TransformerEncoderLayer(

            d_model=512,

            nhead=8,

            dim_feedforward=2048,

            dropout=0.1,

            batch_first=True

        )

        self.transformer = nn.TransformerEncoder(

            encoder_layer,

            num_layers=4
        )

    def forward(self, x):

        return self.transformer(x)

class RainVisionV1(nn.Module):

    def __init__(self):

        super().__init__()

        self.spatial_encoder = SpatialEncoder(
            in_channels=20
        )

        self.temporal_transformer = TemporalTransformer()

        self.rain_occurrence = nn.Sequential(

            nn.Linear(512,256),

            nn.ReLU(),

            nn.Linear(256,1)
        )

        self.rain_intensity = nn.Sequential(

            nn.Linear(512,256),

            nn.ReLU(),

            nn.Linear(256,64),

            nn.ReLU(),

            nn.Linear(64,1)
        )

    def forward(self, x):

        B,T,H,W,C = x.shape
        x = x.reshape(
            B*T,
            H,
            W,
            C
        )
        
        x = x.permute(
            0,3,1,2
        )
        
        embeddings = self.spatial_encoder(x)
        
        embeddings = embeddings.reshape( B, T, 512 )
        # embeddings = []

        # for t in range(T):

        #     frame = x[:,t]

        #     frame = frame.permute(
        #         0,3,1,2
        #     )

        #     emb = self.spatial_encoder(
        #         frame
        #     )

        #     embeddings.append(emb)

        # embeddings = torch.stack(
        #     embeddings,
        #     dim=1
        # )

        temporal = self.temporal_transformer( embeddings )

        last_state = temporal[:,-1]

        rain_prob = self.rain_occurrence( last_state )

        rain_intensity = self.rain_intensity( last_state )

        return rain_prob, rain_intensity

y_small = y[:, lat_mask, :]
y_small = y_small[:, :, lon_mask]

print(y_small.shape)

HISTORY = 24
LEAD = 6

from torch.utils.data import Dataset

class RainDataset(Dataset):

    def __init__(
        self,
        X,
        rain_target,
        history=24,
        lead=6
    ):

        self.X = X.astype(np.float32)

        self.rain_target = rain_target.astype(
            np.float32
        )

        self.history = history
        self.lead = lead

        self.length = (
            len(X)
            - history
            - lead
        )

    def __len__(self):
        return self.length

    def __getitem__(self, idx):

        t = idx + self.history

        x = self.X[
            t-self.history:t
        ]

        future_rain = self.rain_target[
            t+self.lead
        ]

        occurrence = (
            1.0 if future_rain > 0.1
            else 0.0
        )

        return (
            torch.tensor(x),
            torch.tensor(occurrence),
            torch.tensor(
                future_rain,
                dtype=torch.float32
            )
        )

# Normalize each meteorological channel
for c in range(X.shape[-1]):
    mean = X[..., c].mean()
    std = X[..., c].std()

    X[..., c] = (
        X[..., c] - mean
    ) / (std + 1e-6)
    
y_small = y_small * 1000
rain_target = np.log1p(
    y_small.mean(axis=(1,2))
)

dataset = RainDataset(
    X,
    rain_target,
    HISTORY,
    LEAD
)

print(
    "Samples:",
    len(dataset)
)

print(X[...,0].mean(), X[...,0].std())
print(X[...,1].mean(), X[...,1].std())

from torch.utils.data import random_split

train_size = int(
    0.8 * len(dataset)
)

val_size = len(dataset) - train_size

train_dataset, val_dataset = random_split(
    dataset,
    [train_size, val_size]
)

from torch.utils.data import DataLoader

train_loader = DataLoader(
    train_dataset,
    batch_size=2,
    shuffle=True,
    num_workers=0
)

val_loader = DataLoader(
    val_dataset,
    batch_size=2,
    shuffle=False,
    num_workers=0
)

x_batch, _, _ = next(iter(train_loader))

print(x_batch.min())
print(x_batch.max())
print(x_batch.mean())

for batch in train_loader:

    x_batch, y_occ, y_int = batch

    print(
        x_batch.shape
    )

    print(
        y_occ.shape
    )

    print(
        y_int.shape
    )

    break

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)
print(device)

model = RainVisionV1().to(device)

x_batch, y_occ, y_int = next(
    iter(train_loader)
)

x_batch = x_batch.to(device)

with torch.no_grad():

    pred_occ, pred_int = model(
        x_batch
    )

print(pred_occ.shape)
print(pred_int.shape)

bce_loss = nn.BCEWithLogitsLoss()
mse_loss = nn.MSELoss()
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=1e-4,
    weight_decay=1e-5
)

def evaluate(
    model,
    loader,
    device
):

    model.eval()

    total_loss = 0

    with torch.no_grad():

        for (
            x_batch,
            y_occ,
            y_int
        ) in loader:

            x_batch = x_batch.to(device)

            y_occ = (
                y_occ
                .unsqueeze(1)
                .to(device, non_blocking=True)
            )

            y_int = (
                y_int
                .unsqueeze(1)
                .to(device, non_blocking=True)
            )

            pred_occ, pred_int = model(
                x_batch
            )

            loss_occ = bce_loss(
                pred_occ,
                y_occ
            )

            rain_mask = (
                y_occ > 0.5
            ).float()

            if rain_mask.sum() > 0:
                loss_int = (
                    (((pred_int - y_int) ** 2) * rain_mask).sum()
                    / rain_mask.sum()
                )
            else:
                loss_int = torch.tensor(
                    0.0,
                    device=device
                )

            loss = (
                loss_occ
                + loss_int
            )

            total_loss += loss.item()

    return (
        total_loss
        / len(loader)
    )

print(x_batch.min(), x_batch.max())
print(y_occ.min(), y_occ.max())
print(y_int.min(), y_int.max())

occ_values = []

for _, y_occ, _ in train_loader:
    occ_values.append(y_occ)

occ_values = torch.cat(occ_values)

print(torch.unique(occ_values))
print(occ_values.float().mean())
print((occ_values == 1).sum())
print((occ_values == 0).sum())

print(y_small.min())
print(y_small.max())
print(y_small.mean())

print(rain_target.min())
print(rain_target.max())
print(rain_target.mean())

occ = (rain_target > 0.1)

print("Rain samples:", occ.sum())
print("Total samples:", len(occ))

for i, (_, y_occ, y_int) in enumerate(train_loader):

    print(
        f"Batch {i}"
    )

    print(
        "occ:",
        y_occ
    )

    print(
        "int:",
        y_int
    )

    if i == 100:
        break

EPOCHS = 15

best_val_loss = np.inf

train_losses = []
val_losses = []

accum_steps = 8

for epoch in range(EPOCHS):

    model.train()

    running_loss = 0

    optimizer.zero_grad()

    for step, (
        x_batch,
        y_occ,
        y_int
    ) in enumerate(train_loader):

        x_batch = x_batch.to(device)

        y_occ = y_occ.unsqueeze(1).to(device)

        y_int = y_int.unsqueeze(1).to(device)

        pred_occ, pred_int = model(x_batch)

        loss_occ = bce_loss(
            pred_occ,
            y_occ
        )

        rain_mask = (
            y_occ > 0.5
        ).float()

        if rain_mask.sum() > 0:
            loss_int = (
                (((pred_int - y_int) ** 2)
                 * rain_mask).sum()
                / rain_mask.sum()
            )
        else:
            loss_int = torch.tensor(
                0.0,
                device=device
            )

        loss = loss_occ + loss_int

        # Gradient accumulation
        loss = loss / accum_steps

        loss.backward()

        if (
            (step + 1) % accum_steps == 0
            or
            (step + 1) == len(train_loader)
        ):
            optimizer.step()
            optimizer.zero_grad()

        running_loss += loss.item() * accum_steps


    train_loss = (
        running_loss
        / len(train_loader)
    )

    val_loss = evaluate(
        model,
        val_loader,
        device
    )

    train_losses.append(
        train_loss
    )

    val_losses.append(
        val_loss
    )

    print(
        f"Epoch {epoch+1:02d} "
        f"| Train={train_loss:.10f} "
        f"| Val={val_loss:.10f}"
    )

    if val_loss < best_val_loss:

        best_val_loss = val_loss

        torch.save(
            model.state_dict(),
            "RainVisionV1_best.pth"
        )

        print(
            "Best model saved."
        )

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

model.load_state_dict(
    torch.load(
        "RainVisionV1_best.pth"
    )
)

model.eval()

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

