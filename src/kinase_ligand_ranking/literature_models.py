"""Literature-style DeepDTA and GraphDTA baselines under the repo protocol."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import rdchem

try:
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, Dataset
except ImportError:  # pragma: no cover - exercised only when torch is absent.
    torch = None
    nn = None
    DataLoader = object
    Dataset = object

try:
    from torch_geometric.data import Data
    from torch_geometric.loader import DataLoader as GeometricDataLoader
    from torch_geometric.nn import GCNConv, global_max_pool
except ImportError:  # pragma: no cover - exercised only when torch-geometric is absent.
    Data = None
    GeometricDataLoader = None
    GCNConv = None
    global_max_pool = None


DEEPDTA_CHARPROTSET = {
    "A": 1,
    "C": 2,
    "B": 3,
    "E": 4,
    "D": 5,
    "G": 6,
    "F": 7,
    "I": 8,
    "H": 9,
    "K": 10,
    "M": 11,
    "L": 12,
    "O": 13,
    "N": 14,
    "Q": 15,
    "P": 16,
    "S": 17,
    "R": 18,
    "U": 19,
    "T": 20,
    "W": 21,
    "V": 22,
    "Y": 23,
    "X": 24,
    "Z": 25,
}
DEEPDTA_CHARISOSMISET = {
    "#": 29,
    "%": 30,
    ")": 31,
    "(": 1,
    "+": 32,
    "-": 33,
    "/": 34,
    ".": 2,
    "1": 35,
    "0": 3,
    "3": 36,
    "2": 4,
    "5": 37,
    "4": 5,
    "7": 38,
    "6": 6,
    "9": 39,
    "8": 7,
    "=": 40,
    "A": 41,
    "@": 8,
    "C": 42,
    "B": 9,
    "E": 43,
    "D": 10,
    "G": 44,
    "F": 11,
    "I": 45,
    "H": 12,
    "K": 46,
    "M": 47,
    "L": 13,
    "O": 48,
    "N": 14,
    "P": 15,
    "S": 49,
    "R": 16,
    "U": 50,
    "T": 17,
    "W": 51,
    "V": 18,
    "Y": 52,
    "[": 53,
    "Z": 19,
    "]": 54,
    "\\": 20,
    "a": 55,
    "c": 56,
    "b": 21,
    "e": 57,
    "d": 22,
    "g": 58,
    "f": 23,
    "i": 59,
    "h": 24,
    "m": 60,
    "l": 25,
    "o": 61,
    "n": 26,
    "s": 62,
    "r": 27,
    "u": 63,
    "t": 28,
    "y": 64,
}

GRAPHDTA_SEQUENCE_VOCAB = {value: index + 1 for index, value in enumerate("ABCDEFGHIKLMNOPQRSTUVWXYZ")}
GRAPHDTA_ATOM_SYMBOLS = [
    "C",
    "N",
    "O",
    "S",
    "F",
    "Si",
    "P",
    "Cl",
    "Br",
    "Mg",
    "Na",
    "Ca",
    "Fe",
    "As",
    "Al",
    "I",
    "B",
    "V",
    "K",
    "Tl",
    "Yb",
    "Sb",
    "Sn",
    "Ag",
    "Pd",
    "Co",
    "Se",
    "Ti",
    "Zn",
    "H",
    "Li",
    "Ge",
    "Cu",
    "Au",
    "Ni",
    "Cd",
    "In",
    "Mn",
    "Zr",
    "Cr",
    "Pt",
    "Hg",
    "Pb",
    "Unknown",
]


@dataclass
class DeepDTAExactConfig:
    max_seq_len: int = 1000
    max_smi_len: int = 100
    embedding_dim: int = 128
    num_windows_options: Tuple[int, ...] = (32,)
    smi_window_lengths: Tuple[int, ...] = (4,)
    seq_window_lengths: Tuple[int, ...] = (8,)
    learning_rate: float = 1e-3
    batch_size: int = 256
    max_epochs: int = 30
    patience: int = 6
    dropout: float = 0.1
    random_seed: int = 42


@dataclass
class GraphDTAGCNConfig:
    max_seq_len: int = 1000
    num_features_xd: int = 78
    num_features_xt: int = 25
    embed_dim: int = 128
    n_filters: int = 32
    output_dim: int = 128
    dropout: float = 0.2
    learning_rate: float = 5e-4
    batch_size: int = 512
    max_epochs: int = 120
    patience: int = 20
    random_seed: int = 42


class _DeepDTADataset(Dataset):
    def __init__(self, smiles_tensor: np.ndarray, sequence_tensor: np.ndarray, y: np.ndarray):
        self.smiles_tensor = torch.from_numpy(smiles_tensor).long()
        self.sequence_tensor = torch.from_numpy(sequence_tensor).long()
        self.y = torch.from_numpy(y.astype(np.float32)).float()

    def __len__(self) -> int:
        return int(self.y.shape[0])

    def __getitem__(self, index: int):
        return (
            self.smiles_tensor[index],
            self.sequence_tensor[index],
            self.y[index],
        )


class DeepDTAExact(nn.Module if nn is not None else object):
    """Exact-architecture DeepDTA encoder with configurable kernel sizes."""

    def __init__(
        self,
        *,
        num_windows: int,
        smi_window_length: int,
        seq_window_length: int,
        max_smi_len: int,
        max_seq_len: int,
        dropout: float,
    ):
        super().__init__()
        self.smiles_embedding = nn.Embedding(len(DEEPDTA_CHARISOSMISET) + 1, 128, padding_idx=0)
        self.sequence_embedding = nn.Embedding(len(DEEPDTA_CHARPROTSET) + 1, 128, padding_idx=0)

        self.smiles_conv = nn.Sequential(
            nn.Conv1d(128, num_windows, kernel_size=smi_window_length),
            nn.ReLU(),
            nn.Conv1d(num_windows, num_windows * 2, kernel_size=smi_window_length),
            nn.ReLU(),
            nn.Conv1d(num_windows * 2, num_windows * 3, kernel_size=smi_window_length),
            nn.ReLU(),
            nn.AdaptiveMaxPool1d(1),
        )
        self.sequence_conv = nn.Sequential(
            nn.Conv1d(128, num_windows, kernel_size=seq_window_length),
            nn.ReLU(),
            nn.Conv1d(num_windows, num_windows * 2, kernel_size=seq_window_length),
            nn.ReLU(),
            nn.Conv1d(num_windows * 2, num_windows * 3, kernel_size=seq_window_length),
            nn.ReLU(),
            nn.AdaptiveMaxPool1d(1),
        )
        self.mlp = nn.Sequential(
            nn.Linear(num_windows * 6, 1024),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 1),
        )
        self.max_smi_len = max_smi_len
        self.max_seq_len = max_seq_len

    def forward(self, smiles_tokens: torch.Tensor, sequence_tokens: torch.Tensor) -> torch.Tensor:
        smiles_embed = self.smiles_embedding(smiles_tokens).transpose(1, 2)
        sequence_embed = self.sequence_embedding(sequence_tokens).transpose(1, 2)
        smiles_encoded = self.smiles_conv(smiles_embed).squeeze(-1)
        sequence_encoded = self.sequence_conv(sequence_embed).squeeze(-1)
        return self.mlp(torch.cat([smiles_encoded, sequence_encoded], dim=1)).squeeze(-1)


class GraphDTAGCNExact(nn.Module if nn is not None else object):
    """Exact GCNNet variant from the GraphDTA release."""

    def __init__(self, config: GraphDTAGCNConfig):
        if GCNConv is None or global_max_pool is None:
            raise ImportError("torch-geometric is required for GraphDTA baselines")
        super().__init__()
        self.conv1 = GCNConv(config.num_features_xd, config.num_features_xd)
        self.conv2 = GCNConv(config.num_features_xd, config.num_features_xd * 2)
        self.conv3 = GCNConv(config.num_features_xd * 2, config.num_features_xd * 4)
        self.fc_g1 = nn.Linear(config.num_features_xd * 4, 1024)
        self.fc_g2 = nn.Linear(1024, config.output_dim)
        self.embedding_xt = nn.Embedding(config.num_features_xt + 1, config.embed_dim, padding_idx=0)
        self.conv_xt_1 = nn.Conv1d(
            in_channels=config.max_seq_len,
            out_channels=config.n_filters,
            kernel_size=8,
        )
        self.fc1_xt = nn.Linear(config.n_filters * 121, config.output_dim)
        self.fc1 = nn.Linear(config.output_dim * 2, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.out = nn.Linear(512, 1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, data: Data) -> torch.Tensor:
        x, edge_index, batch = data.x, data.edge_index, data.batch
        target = data.target

        x = self.relu(self.conv1(x, edge_index))
        x = self.relu(self.conv2(x, edge_index))
        x = self.relu(self.conv3(x, edge_index))
        x = global_max_pool(x, batch)
        x = self.dropout(self.relu(self.fc_g1(x)))
        x = self.dropout(self.fc_g2(x))

        embedded_xt = self.embedding_xt(target)
        conv_xt = self.relu(self.conv_xt_1(embedded_xt))
        xt = self.fc1_xt(conv_xt.view(-1, 32 * 121))

        xc = torch.cat((x, xt), dim=1)
        xc = self.dropout(self.relu(self.fc1(xc)))
        xc = self.dropout(self.relu(self.fc2(xc)))
        return self.out(xc).squeeze(-1)


def run_deepdta_exact(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    *,
    config: Optional[DeepDTAExactConfig] = None,
    device: Optional[str] = None,
) -> Tuple[np.ndarray, Dict[str, object]]:
    _require_torch()
    config = config or DeepDTAExactConfig()
    run_device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    encoded = {
        "train": _encode_deepdta_dataframe(train_df, config),
        "val": _encode_deepdta_dataframe(val_df, config),
        "test": _encode_deepdta_dataframe(test_df, config),
    }

    train_dataset = _DeepDTADataset(
        encoded["train"]["smiles"],
        encoded["train"]["sequence"],
        train_df["p_activity"].to_numpy(),
    )
    val_dataset = _DeepDTADataset(
        encoded["val"]["smiles"],
        encoded["val"]["sequence"],
        val_df["p_activity"].to_numpy(),
    )
    test_dataset = _DeepDTADataset(
        encoded["test"]["smiles"],
        encoded["test"]["sequence"],
        test_df["p_activity"].to_numpy(),
    )

    trial_rows: List[Dict[str, object]] = []
    best_state = None
    best_params: Dict[str, int] = {}
    best_rmse = float("inf")
    best_history: List[Dict[str, float]] = []

    for num_windows in config.num_windows_options:
        for smi_window_length in config.smi_window_lengths:
            for seq_window_length in config.seq_window_lengths:
                set_torch_seed(config.random_seed)
                model = DeepDTAExact(
                    num_windows=num_windows,
                    smi_window_length=smi_window_length,
                    seq_window_length=seq_window_length,
                    max_smi_len=config.max_smi_len,
                    max_seq_len=config.max_seq_len,
                    dropout=config.dropout,
                ).to(run_device)
                history = _fit_torch_regressor(
                    model=model,
                    train_loader=DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True),
                    val_loader=DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False),
                    learning_rate=config.learning_rate,
                    max_epochs=config.max_epochs,
                    patience=config.patience,
                    device=run_device,
                )
                val_predictions = _predict_deepdta(
                    model,
                    DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False),
                    device=run_device,
                )
                val_rmse = float(np.sqrt(np.mean((val_predictions - val_df["p_activity"].to_numpy()) ** 2)))
                trial_rows.append(
                    {
                        "num_windows": int(num_windows),
                        "smi_window_length": int(smi_window_length),
                        "seq_window_length": int(seq_window_length),
                        "val_rmse": val_rmse,
                    }
                )
                if val_rmse < best_rmse:
                    best_rmse = val_rmse
                    best_state = deepcopy(model.state_dict())
                    best_params = {
                        "num_windows": int(num_windows),
                        "smi_window_length": int(smi_window_length),
                        "seq_window_length": int(seq_window_length),
                    }
                    best_history = history

    if best_state is None:
        raise RuntimeError("DeepDTA hyperparameter search produced no model state")

    set_torch_seed(config.random_seed)
    best_model = DeepDTAExact(
        num_windows=best_params["num_windows"],
        smi_window_length=best_params["smi_window_length"],
        seq_window_length=best_params["seq_window_length"],
        max_smi_len=config.max_smi_len,
        max_seq_len=config.max_seq_len,
        dropout=config.dropout,
    ).to(run_device)
    best_model.load_state_dict(best_state)
    test_predictions = _predict_deepdta(
        best_model,
        DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False),
        device=run_device,
    )
    metadata = {
        "config": asdict(config),
        "selected_hyperparameters": best_params,
        "validation_trials": trial_rows,
        "training_history": best_history,
        "metrics": {},
    }
    return test_predictions.astype(np.float32), metadata


def run_graphdta_gcn_exact(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    *,
    config: Optional[GraphDTAGCNConfig] = None,
    device: Optional[str] = None,
) -> Tuple[np.ndarray, Dict[str, object]]:
    _require_torch()
    if Data is None or GeometricDataLoader is None:
        raise ImportError("torch-geometric is required for GraphDTA baselines")

    config = config or GraphDTAGCNConfig()
    run_device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    graph_cache = _build_smile_graph_cache(
        pd.concat([train_df["smiles"], val_df["smiles"], test_df["smiles"]]).drop_duplicates().tolist()
    )
    train_data = _graphdta_dataframe_to_data_list(train_df, graph_cache, config.max_seq_len)
    val_data = _graphdta_dataframe_to_data_list(val_df, graph_cache, config.max_seq_len)
    test_data = _graphdta_dataframe_to_data_list(test_df, graph_cache, config.max_seq_len)

    set_torch_seed(config.random_seed)
    model = GraphDTAGCNExact(config).to(run_device)
    history = _fit_geometric_regressor(
        model=model,
        train_loader=GeometricDataLoader(train_data, batch_size=config.batch_size, shuffle=True),
        val_loader=GeometricDataLoader(val_data, batch_size=config.batch_size, shuffle=False),
        learning_rate=config.learning_rate,
        max_epochs=config.max_epochs,
        patience=config.patience,
        device=run_device,
    )
    test_predictions = _predict_graphdta(
        model,
        GeometricDataLoader(test_data, batch_size=config.batch_size, shuffle=False),
        device=run_device,
    )
    metadata = {
        "config": asdict(config),
        "training_history": history,
        "graph_feature_size": int(config.num_features_xd),
        "metrics": {},
    }
    return test_predictions.astype(np.float32), metadata


def set_torch_seed(seed: int) -> None:
    _require_torch()
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _encode_deepdta_dataframe(df: pd.DataFrame, config: DeepDTAExactConfig) -> Dict[str, np.ndarray]:
    smiles_array = np.stack(
        [
            _encode_sequence_like(str(smiles), config.max_smi_len, DEEPDTA_CHARISOSMISET)
            for smiles in df["smiles"].tolist()
        ]
    ).astype(np.int64)
    sequence_array = np.stack(
        [
            _encode_sequence_like(str(sequence), config.max_seq_len, DEEPDTA_CHARPROTSET)
            for sequence in df["target_sequence"].tolist()
        ]
    ).astype(np.int64)
    return {"smiles": smiles_array, "sequence": sequence_array}


def _encode_sequence_like(value: str, max_length: int, alphabet: Dict[str, int]) -> np.ndarray:
    encoded = np.zeros((max_length,), dtype=np.int64)
    for index, char in enumerate(str(value)[:max_length]):
        encoded[index] = alphabet.get(char, 0)
    return encoded


def _fit_torch_regressor(
    *,
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    learning_rate: float,
    max_epochs: int,
    patience: int,
    device: torch.device,
) -> List[Dict[str, float]]:
    _require_torch()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()
    best_state = deepcopy(model.state_dict())
    best_val_rmse = float("inf")
    wait = 0
    history: List[Dict[str, float]] = []

    for epoch in range(1, max_epochs + 1):
        model.train()
        train_losses: List[float] = []
        for smiles_tokens, sequence_tokens, y_batch in train_loader:
            smiles_tokens = smiles_tokens.to(device)
            sequence_tokens = sequence_tokens.to(device)
            y_batch = y_batch.to(device)
            optimizer.zero_grad()
            predictions = model(smiles_tokens, sequence_tokens)
            loss = criterion(predictions, y_batch)
            loss.backward()
            optimizer.step()
            train_losses.append(float(loss.item()))

        val_predictions = _predict_deepdta(model, val_loader, device=device)
        val_targets = np.concatenate([y.numpy() for _, _, y in val_loader], axis=0)
        val_rmse = float(np.sqrt(np.mean((val_predictions - val_targets) ** 2)))
        history.append(
            {
                "epoch": float(epoch),
                "train_loss": float(np.mean(train_losses)),
                "val_rmse": val_rmse,
            }
        )
        if val_rmse + 1e-8 < best_val_rmse:
            best_val_rmse = val_rmse
            best_state = deepcopy(model.state_dict())
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                break

    model.load_state_dict(best_state)
    return history


def _predict_deepdta(model: nn.Module, loader: DataLoader, *, device: torch.device) -> np.ndarray:
    _require_torch()
    model.eval()
    predictions: List[np.ndarray] = []
    with torch.no_grad():
        for smiles_tokens, sequence_tokens, _ in loader:
            batch_prediction = model(
                smiles_tokens.to(device),
                sequence_tokens.to(device),
            )
            predictions.append(batch_prediction.detach().cpu().numpy())
    return np.concatenate(predictions, axis=0)


def _fit_geometric_regressor(
    *,
    model: nn.Module,
    train_loader: GeometricDataLoader,
    val_loader: GeometricDataLoader,
    learning_rate: float,
    max_epochs: int,
    patience: int,
    device: torch.device,
) -> List[Dict[str, float]]:
    _require_torch()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()
    best_state = deepcopy(model.state_dict())
    best_val_rmse = float("inf")
    wait = 0
    history: List[Dict[str, float]] = []

    for epoch in range(1, max_epochs + 1):
        model.train()
        train_losses: List[float] = []
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            predictions = model(batch)
            loss = criterion(predictions, batch.y.view(-1))
            loss.backward()
            optimizer.step()
            train_losses.append(float(loss.item()))

        val_predictions, val_targets = _predict_graphdta_with_targets(model, val_loader, device=device)
        val_rmse = float(np.sqrt(np.mean((val_predictions - val_targets) ** 2)))
        history.append(
            {
                "epoch": float(epoch),
                "train_loss": float(np.mean(train_losses)),
                "val_rmse": val_rmse,
            }
        )
        if val_rmse + 1e-8 < best_val_rmse:
            best_val_rmse = val_rmse
            best_state = deepcopy(model.state_dict())
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                break

    model.load_state_dict(best_state)
    return history


def _predict_graphdta(model: nn.Module, loader: GeometricDataLoader, *, device: torch.device) -> np.ndarray:
    predictions, _ = _predict_graphdta_with_targets(model, loader, device=device)
    return predictions


def _predict_graphdta_with_targets(
    model: nn.Module,
    loader: GeometricDataLoader,
    *,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray]:
    _require_torch()
    model.eval()
    predictions: List[np.ndarray] = []
    labels: List[np.ndarray] = []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            predictions.append(model(batch).detach().cpu().numpy())
            labels.append(batch.y.view(-1).detach().cpu().numpy())
    return np.concatenate(predictions, axis=0), np.concatenate(labels, axis=0)


def _graphdta_dataframe_to_data_list(
    df: pd.DataFrame,
    graph_cache: Dict[str, Tuple[int, np.ndarray, np.ndarray]],
    max_seq_len: int,
) -> List[Data]:
    data_list: List[Data] = []
    sequence_cache: Dict[str, torch.Tensor] = {}
    for row in df.itertuples(index=False):
        c_size, features, edge_index = graph_cache[str(row.smiles)]
        target_sequence = str(row.target_sequence)
        if target_sequence not in sequence_cache:
            sequence_cache[target_sequence] = torch.from_numpy(
                _encode_graphdta_sequence(target_sequence, max_seq_len)
            ).unsqueeze(0)
        graph_data = Data(
            x=torch.tensor(features, dtype=torch.float32),
            edge_index=torch.tensor(edge_index, dtype=torch.long).t().contiguous(),
            y=torch.tensor([float(row.p_activity)], dtype=torch.float32),
        )
        graph_data.target = sequence_cache[target_sequence]
        graph_data.c_size = torch.tensor([c_size], dtype=torch.long)
        data_list.append(graph_data)
    return data_list


def _encode_graphdta_sequence(sequence: str, max_seq_len: int) -> np.ndarray:
    encoded = np.zeros((max_seq_len,), dtype=np.int64)
    for index, char in enumerate(str(sequence).upper()[:max_seq_len]):
        encoded[index] = GRAPHDTA_SEQUENCE_VOCAB.get(char, 0)
    return encoded


def _build_smile_graph_cache(smiles_values: Iterable[str]) -> Dict[str, Tuple[int, np.ndarray, np.ndarray]]:
    graph_cache: Dict[str, Tuple[int, np.ndarray, np.ndarray]] = {}
    for smiles in smiles_values:
        graph_cache[str(smiles)] = _smile_to_graph(str(smiles))
    return graph_cache


def _smile_to_graph(smiles: str) -> Tuple[int, np.ndarray, np.ndarray]:
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        raise ValueError(f"Invalid SMILES for GraphDTA featurization: {smiles}")

    features = []
    for atom in molecule.GetAtoms():
        atom_feature = _graphdta_atom_features(atom)
        features.append(atom_feature / max(atom_feature.sum(), 1.0))

    edges: List[List[int]] = []
    for bond in molecule.GetBonds():
        begin = bond.GetBeginAtomIdx()
        end = bond.GetEndAtomIdx()
        edges.append([begin, end])
        edges.append([end, begin])
    if not edges:
        edges = [[0, 0]]

    return molecule.GetNumAtoms(), np.asarray(features, dtype=np.float32), np.asarray(edges, dtype=np.int64)


def _graphdta_atom_features(atom) -> np.ndarray:
    return np.asarray(
        _one_of_k_encoding_unk(atom.GetSymbol(), GRAPHDTA_ATOM_SYMBOLS)
        + _one_of_k_encoding(atom.GetDegree(), list(range(11)))
        + _one_of_k_encoding_unk(atom.GetTotalNumHs(), list(range(11)))
        + _one_of_k_encoding_unk(atom.GetValence(rdchem.ValenceType.IMPLICIT), list(range(11)))
        + [atom.GetIsAromatic()],
        dtype=np.float32,
    )


def _one_of_k_encoding(value, allowable_set: Sequence[object]) -> List[bool]:
    if value not in allowable_set:
        raise ValueError(f"Input {value!r} not in allowable set")
    return [value == item for item in allowable_set]


def _one_of_k_encoding_unk(value, allowable_set: Sequence[object]) -> List[bool]:
    if value not in allowable_set:
        value = allowable_set[-1]
    return [value == item for item in allowable_set]


def _require_torch() -> None:
    if torch is None or nn is None:
        raise ImportError("PyTorch is required for literature baselines. Install torch and torch-geometric.")
