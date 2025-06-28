"""
GeoClustering module extracted from route_generator.ipynb for production use.
This file exposes the `GeoClustering` class which is consumed by
`services/geoclustering.py`.
"""

from __future__ import annotations

import json
import pickle
from datetime import datetime
from math import atan2, cos, radians, sin, sqrt
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple, Union

import numpy as np
from sklearn.cluster import DBSCAN, KMeans
from sklearn.metrics import silhouette_score

CoordinateTuple = Tuple[float, float]  # (lon, lat)


class GeoClustering:
    """Utility class for clustering geographical coordinates.

    Parameters
    ----------
    cluster_method : str, default="kmeans"
        One of ``{"kmeans", "dbscan", "distance"}``.
    """

    def __init__(self, cluster_method: str = "kmeans") -> None:
        self.cluster_method: str = cluster_method
        self.model: Any = None  # Trained model instance (e.g. KMeans, DBSCAN)
        self.cluster_centers: List[CoordinateTuple] = []
        self.model_metadata: Dict[str, Any] = {}

    # ---------------------------------------------------------------------
    # Helpers
    # ---------------------------------------------------------------------
    @staticmethod
    def haversine_distance(pt1: CoordinateTuple, pt2: CoordinateTuple) -> float:
        """Return Haversine distance in kilometres between two points."""
        R = 6371  # Earth radius in km
        lon1, lat1 = map(radians, pt1)
        lon2, lat2 = map(radians, pt2)
        dlon = lon2 - lon1
        dlat = lat2 - lat1
        a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
        return R * (2 * atan2(sqrt(a), sqrt(1 - a)))

    # ------------------------------------------------------------------
    # Clustering implementations
    # ------------------------------------------------------------------
    def _create_clusters_dbscan(
        self, coordinates: List[CoordinateTuple], eps_km: float = 5.0, min_samples: int = 6
    ) -> Tuple[np.ndarray, List[CoordinateTuple]]:
        if len(coordinates) < min_samples:
            return np.array([-1] * len(coordinates)), []

        coords_rad = np.radians([[lat, lon] for lon, lat in coordinates])
        eps_rad = eps_km / 6371.0

        dbscan = DBSCAN(eps=eps_rad, min_samples=min_samples, metric="haversine")
        cluster_labels = dbscan.fit_predict(coords_rad)

        centers: List[CoordinateTuple] = []
        for label in set(cluster_labels):
            if label == -1:
                continue
            cluster_points = [coordinates[i] for i, l in enumerate(cluster_labels) if l == label]
            center_lon = sum(p[0] for p in cluster_points) / len(cluster_points)
            center_lat = sum(p[1] for p in cluster_points) / len(cluster_points)
            centers.append((center_lon, center_lat))
        return cluster_labels, centers

    def _create_clusters_kmeans(
        self, coordinates: List[CoordinateTuple], n_clusters: int = 3
    ) -> Tuple[np.ndarray, List[CoordinateTuple]]:
        n_clusters = min(max(n_clusters, 1), len(coordinates))
        coords_array = np.array(coordinates)
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(coords_array)
        self.model = kmeans
        self.cluster_centers = [(float(c[0]), float(c[1])) for c in kmeans.cluster_centers_]
        return cluster_labels, self.cluster_centers

    def _create_distance_based_clusters(
        self, coordinates: List[CoordinateTuple], max_distance_km: float = 5.0
    ) -> Tuple[np.ndarray, List[CoordinateTuple]]:
        clusters: List[List[int]] = []
        cluster_labels = [-1] * len(coordinates)
        current_cluster = 0
        for i, coord in enumerate(coordinates):
            if cluster_labels[i] != -1:
                continue
            cluster_points = [i]
            cluster_labels[i] = current_cluster
            for j, other in enumerate(coordinates):
                if i == j or cluster_labels[j] != -1:
                    continue
                if self.haversine_distance(coord, other) <= max_distance_km:
                    cluster_points.append(j)
                    cluster_labels[j] = current_cluster
            clusters.append(cluster_points)
            current_cluster += 1

        centers: List[CoordinateTuple] = []
        for idxs in clusters:
            cluster_coords = [coordinates[i] for i in idxs]
            center_lon = sum(p[0] for p in cluster_coords) / len(cluster_coords)
            center_lat = sum(p[1] for p in cluster_coords) / len(cluster_coords)
            centers.append((center_lon, center_lat))
        return np.array(cluster_labels), centers

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def cluster_coordinates(
        self,
        coordinates: List[CoordinateTuple],
        *,
        auto: bool = False,
        max_k: int = 10,
        **kwargs,
    ) -> Tuple[np.ndarray, List[CoordinateTuple]]:
        if self.cluster_method == "dbscan":
            labels, centers = self._create_clusters_dbscan(coordinates, **kwargs)
        elif self.cluster_method == "kmeans":
            if auto:
                labels, centers = self._auto_kmeans(coordinates, max_k=max_k)
            else:
                labels, centers = self._create_clusters_kmeans(coordinates, **kwargs)
        elif self.cluster_method == "distance":
            labels, centers = self._create_distance_based_clusters(coordinates, **kwargs)
        else:
            raise ValueError("Unknown clustering method")

        # Populate metadata for persistence & info endpoint
        self.model_metadata = {
            "cluster_method": self.cluster_method,
            "n_clusters": len(centers),
            "created": datetime.now().isoformat(),
        }
        self.cluster_centers = centers
        return labels, centers

    # ------------------------------------------------------------------
    def _auto_kmeans(
        self, coordinates: List[CoordinateTuple], *, max_k: int = 10
    ) -> Tuple[np.ndarray, List[CoordinateTuple]]:
        coords_array = np.array(coordinates)
        best_score = -1.0
        best_k = 2
        best_labels: np.ndarray | None = None
        best_centers: np.ndarray | None = None
        best_model: KMeans | None = None

        for k in range(2, min(max_k, len(coordinates)) + 1):
            model = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = model.fit_predict(coords_array)
            try:
                score = silhouette_score(coords_array, labels)
            except Exception:
                score = -1
            if score > best_score:
                best_k = k
                best_score = score
                best_labels = labels
                best_centers = model.cluster_centers_
                best_model = model

        if best_labels is None or best_centers is None or best_model is None:
            # Fallback – this should rarely happen
            return self._create_clusters_kmeans(coordinates, n_clusters=best_k)

        self.model = best_model
        self.cluster_centers = [(float(c[0]), float(c[1])) for c in best_centers]
        return best_labels, self.cluster_centers

    # ------------------------------------------------------------------
    # Persistence helpers
    # ------------------------------------------------------------------
    def save_model(self, model_path: str | Path, metadata_path: str | Path | None = None) -> None:
        """Persist trained model & metadata to disk."""
        if self.model is None:
            raise ValueError("No model to save – train or load a model first.")

        model_path = Path(model_path)
        with model_path.open("wb") as f:
            pickle.dump(self.model, f)

        if metadata_path is None:
            metadata_path = model_path.with_suffix(".json")
        metadata_path = Path(metadata_path)

        # Refresh metadata with latest values
        self.model_metadata.setdefault("n_clusters", len(self.cluster_centers))
        self.model_metadata.setdefault("cluster_method", self.cluster_method)
        self.model_metadata["saved"] = datetime.now().isoformat()
        self.model_metadata["cluster_centers"] = [
            {"lon": lon, "lat": lat} for lon, lat in self.cluster_centers
        ]
        with metadata_path.open("w", encoding="utf-8") as f:
            json.dump(self.model_metadata, f, indent=2)

    def load_model(self, model_path: str | Path, metadata_path: str | Path | None = None) -> None:
        """Load previously saved model & metadata from disk."""
        model_path = Path(model_path)
        with model_path.open("rb") as f:
            self.model = pickle.load(f)

        if metadata_path is None:
            metadata_path = model_path.with_suffix(".json")
        metadata_path = Path(metadata_path)
        if metadata_path.exists():
            with metadata_path.open("r", encoding="utf-8") as f:
                self.model_metadata = json.load(f)
            self.cluster_centers = [
                (c["lon"], c["lat"]) for c in self.model_metadata.get("cluster_centers", [])
            ]
        else:
            # Fallback: derive from model if possible
            if hasattr(self.model, "cluster_centers_"):
                self.cluster_centers = [
                    (float(c[0]), float(c[1])) for c in self.model.cluster_centers_
                ]
            self.model_metadata = {
                "cluster_method": self.cluster_method,
                "n_clusters": len(self.cluster_centers),
                "loaded": datetime.now().isoformat(),
            }

    # ------------------------------------------------------------------
    # Prediction + formatting helpers
    # ------------------------------------------------------------------
    def predict_cluster(self, new_coords: List[CoordinateTuple]) -> np.ndarray:
        """Return cluster label for each coordinate in *new_coords*."""
        if self.model is None:
            raise ValueError("Model not loaded – call `load_model()` or train first.")
        if self.cluster_method != "kmeans":
            raise NotImplementedError(
                "predict_cluster currently supports only KMeans models.")
        X = np.array(new_coords)
        return self.model.predict(X)

    # Key method used by FastAPI service
    def predict_and_format_for_backend(
        self, coordinates: Sequence[Union[Dict[str, float], CoordinateTuple]]
    ) -> Dict[str, Any]:
        """Predict clusters and format the output expected by FastAPI layer."""
        # Normalise input -> list[CoordinateTuple]
        coords_list: List[CoordinateTuple] = []
        for c in coordinates:
            if isinstance(c, (list, tuple)) and len(c) == 2:
                coords_list.append((float(c[0]), float(c[1])))
            elif isinstance(c, dict) and {"lon", "lat"}.issubset(c):
                coords_list.append((float(c["lon"]), float(c["lat"])) )
            else:
                raise ValueError("Invalid coordinate format: expected {'lon','lat'} or (lon,lat)")

        labels = self.predict_cluster(coords_list)

        clusters: Dict[int, List[Dict[str, float]]] = {}
        for (lon, lat), lbl in zip(coords_list, labels):
            lbl_int = int(lbl) if hasattr(lbl, "item") else int(lbl)
            clusters.setdefault(lbl_int, []).append({"lon": lon, "lat": lat})

        cluster_info: List[Dict[str, Any]] = []
        for lbl, pts in clusters.items():
            center_dict: Dict[str, float] | None = None
            if 0 <= lbl < len(self.cluster_centers):
                center_lon, center_lat = self.cluster_centers[lbl]
                center_dict = {"lon": center_lon, "lat": center_lat}
            cluster_info.append(
                {
                    "cluster_id": lbl,
                    "cluster_center": center_dict if center_dict else {},
                    "coordinates": pts,
                }
            )

        response = {
            "success": True,
            "timestamp": datetime.now().isoformat(),
            "total_points": len(coords_list),
            "total_clusters": len(clusters),
            "clusters": cluster_info,
            "model_info": self.model_metadata,
        }
        return response

    # ------------------------------------------------------------------
    # Convenience utility for notebooks / debugging
    # ------------------------------------------------------------------
    def analyze_clusters(self, coordinates: List[CoordinateTuple], labels: Sequence[int]) -> Dict[str, Any]:
        """Return dict describing clusters (counts, radii etc.)."""
        cluster_info: Dict[str, Any] = {}
        for lbl in set(labels):
            if lbl == -1:
                cluster_info["noise"] = {
                    "count": int(sum(l == -1 for l in labels)),
                    "points": [coordinates[i] for i, l in enumerate(labels) if l == -1],
                }
                continue
            cluster_points = [coordinates[i] for i, l in enumerate(labels) if l == lbl]
            center_lon = sum(p[0] for p in cluster_points) / len(cluster_points)
            center_lat = sum(p[1] for p in cluster_points) / len(cluster_points)
            max_dist = max(
                self.haversine_distance((center_lon, center_lat), pt) for pt in cluster_points
            )
            cluster_info[f"cluster_{lbl}"] = {
                "count": len(cluster_points),
                "center": (center_lon, center_lat),
                "radius_km": max_dist,
            }
        return cluster_info
