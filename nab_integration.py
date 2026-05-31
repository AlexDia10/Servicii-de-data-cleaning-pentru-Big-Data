"""
NAB Integration Module for BigData Anomaly Detection
Integrează NAB (Numenta Anomaly Benchmark) cu Spark pentru detecția anomaliilor în date streaming
"""

import os
import sys
import json
from pathlib import Path
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple

# Adaugă calea NAB la sys.path
NAB_PATH = Path(__file__).parent / "NAB"
if str(NAB_PATH) not in sys.path:
    sys.path.insert(0, str(NAB_PATH))


class NABIntegration:
    """Clasă pentru integrarea NAB cu Spark"""
    
    def __init__(self, nab_root: str = None):
        """
        Inițializează integrarea NAB
        
        Args:
            nab_root: Cale către directorul rădăcină NAB (implicit: ./NAB)
        """
        self.nab_root = nab_root or str(NAB_PATH)
        self.data_dir = Path(self.nab_root) / "data"
        self.labels_dir = Path(self.nab_root) / "labels"
        self.results_dir = Path(self.nab_root) / "results"
        
        # Asigură-te că directoarele există
        self.results_dir.mkdir(parents=True, exist_ok=True)
    
    def get_available_datasets(self) -> List[str]:
        """
        Returnează lista datelor disponibile din NAB cu cale relativă
        
        Returns:
            Lista cu cale relativă: dossier/dataset_name (fără .csv)
        """
        # Caută CSV-uri în subdirectoare și în root
        data_files = {}
        
        # Caută în subdirectoare
        for subdir in self.data_dir.iterdir():
            if subdir.is_dir():
                for csv_file in subdir.glob("*.csv"):
                    relative_path = f"{subdir.name}/{csv_file.stem}"
                    data_files[relative_path] = relative_path
        
        # Caută în root
        for csv_file in self.data_dir.glob("*.csv"):
            data_files[csv_file.stem] = csv_file.stem
        
        return sorted(data_files.values())
    
    def load_data(self, dataset_name: str) -> pd.DataFrame:
        """
        Încarcă un set de date din NAB
        
        Args:
            dataset_name: Cale relativă (dossier/dataset_name sau dataset_name)
            
        Returns:
            DataFrame cu coloane: timestamp, value
        """
        file_path = None
        
        # Dacă dataset_name conține "/", încearcă direct
        if "/" in dataset_name:
            file_path = self.data_dir / f"{dataset_name}.csv"
            if file_path.exists():
                df = pd.read_csv(file_path)
                return df
        
        # Caută fișierul în subdirectoare
        for subdir in self.data_dir.iterdir():
            if subdir.is_dir():
                candidate = subdir / f"{dataset_name}.csv"
                if candidate.exists():
                    file_path = candidate
                    df = pd.read_csv(file_path)
                    return df
        
        # Caută în root
        file_path = self.data_dir / f"{dataset_name}.csv"
        if file_path.exists():
            df = pd.read_csv(file_path)
            return df
        
        raise FileNotFoundError(f"Dataset not found: {dataset_name}")
    
    def load_labels(self, label_file: str = "labels/combined_windows.json") -> Dict:
        """
        Încarcă etichetele anomaliilor din NAB
        
        Args:
            label_file: Calea către fișierul de etichete
            
        Returns:
            Dicționar cu anomalii etichetate pe dataset
        """
        labels_path = Path(self.nab_root) / label_file
        if not labels_path.exists():
            raise FileNotFoundError(f"Labels not found: {labels_path}")
        
        with open(labels_path, 'r') as f:
            labels = json.load(f)
        
        return labels
    
    def prepare_for_spark(self, dataset_name: str) -> Tuple[pd.DataFrame, Dict]:
        """
        Pregătește datele NAB pentru procesare cu Spark
        
        Args:
            dataset_name: Numele setului de date
            
        Returns:
            Tuplu (DataFrame, etichete)
        """
        data = self.load_data(dataset_name)
        labels = self.load_labels()
        
        return data, labels.get(dataset_name, {})
    
    def get_dataset_statistics(self, dataset_name: str) -> Dict:
        """
        Obține statistici despre un set de date
        
        Args:
            dataset_name: Numele setului de date
            
        Returns:
            Dicționar cu statistici
        """
        data = self.load_data(dataset_name)
        
        return {
            "name": dataset_name,
            "rows": len(data),
            "columns": list(data.columns),
            "mean": float(data["value"].mean()),
            "std": float(data["value"].std()),
            "min": float(data["value"].min()),
            "max": float(data["value"].max()),
            "timestamp_range": f"{data['timestamp'].min()} to {data['timestamp'].max()}"
        }
    
    def get_anomaly_intervals(self, dataset_name: str, df: pd.DataFrame = None) -> List[Tuple[int, int]]:
        """
        Extrage intervalele de anomalii pentru un dataset din NAB labels.
        
        Suportă ambele formate:
        - Timestamps: "2014-10-30 15:30:00" → convertite la indices
        - Indices directe: [100, 200]
        
        Args:
            dataset_name: Numele setului de date (ex: "nyc_taxi" sau "realKnownCause/nyc_taxi")
            df: Optional DataFrame cu timestamps pentru convertire
            
        Returns:
            Lista de tuple (start_index, end_index) pentru fiecare interval anomalie
        """
        try:
            labels = self.load_labels()
            
            # Încearcă variantele de key
            key_variants = [
                dataset_name,
                f"{dataset_name}.csv",
                # Dacă dataset_name e doar "nyc_taxi", cauta si cu prefixe de subdirectoare
            ]
            
            # Adaugă variante cu toate subdirectoarele
            for subdir in self.data_dir.iterdir():
                if subdir.is_dir():
                    subdir_name = subdir.name
                    key_variants.append(f"{subdir_name}/{dataset_name}")
                    key_variants.append(f"{subdir_name}/{dataset_name}.csv")
            
            dataset_labels = None
            matching_key = None
            
            for key in key_variants:
                if key in labels:
                    dataset_labels = labels[key]
                    matching_key = key
                    print(f"Found labels for key: {key}")
                    break
            
            if dataset_labels is None or len(dataset_labels) == 0:
                print(f"No labels found. Available keys: {list(labels.keys())[:5]}...")
                return []
            
            # Converteste labels la indices
            intervals = []
            
            if isinstance(dataset_labels, list):
                for interval in dataset_labels:
                    if isinstance(interval[0], str) and df is not None:
                        # Format timestamp - converteste la indices
                        try:
                            start_ts = pd.to_datetime(interval[0])
                            end_ts = pd.to_datetime(interval[1])

                            # Primul index cu timestamp >= start_ts
                            start_mask = df['timestamp'] >= start_ts
                            if not start_mask.any():
                                continue
                            start_idx = int(start_mask.idxmax())

                            # Ultimul index cu timestamp <= end_ts
                            # idxmax() returnează PRIMUL True, nu ultimul — de aceea folosim index[-1]
                            end_mask = df['timestamp'] <= end_ts
                            if not end_mask.any():
                                continue
                            end_idx = int(end_mask[end_mask].index[-1])

                            intervals.append((start_idx, end_idx + 1))
                        except Exception as e:
                            print(f"Error converting timestamps: {e}")
                            # Fallback: incearca direct indexing
                            try:
                                start_idx = int(interval[0]) if isinstance(interval[0], (int, str)) else 0
                                end_idx = int(interval[1]) if isinstance(interval[1], (int, str)) else len(df)
                                intervals.append((start_idx, end_idx))
                            except:
                                pass
                    else:
                        # Format index direct
                        try:
                            intervals.append((int(interval[0]), int(interval[1])))
                        except:
                            pass
            
            print(f"Extracted {len(intervals)} anomaly intervals from {matching_key}")
            return intervals
            
        except Exception as e:
            print(f"Error loading anomaly intervals: {e}")
            import traceback
            traceback.print_exc()
            return []
    
    def list_detectors(self) -> List[str]:
        """
        Listează detectori disponibili în NAB
        
        Returns:
            Lista cu nume detectori
        """
        detectors_dir = Path(self.nab_root) / "nab" / "detectors"
        if not detectors_dir.exists():
            return []
        
        detectors = [d.name for d in detectors_dir.iterdir() if d.is_dir()]
        return sorted(detectors)


def print_nab_info():
    """Afișează informații despre integrarea NAB"""
    print("=" * 60)
    print("NAB Integration - Anomaly Detection for BigData")
    print("=" * 60)
    
    try:
        nab = NABIntegration()
        
        print("\n[INFO] Available Databases:")
        datasets = nab.get_available_datasets()
        print(f"   Total: {len(datasets)} datasets")
        for i, dataset in enumerate(datasets[:5], 1):
            print(f"   {i}. {dataset}")
        if len(datasets) > 5:
            print(f"   ... și {len(datasets) - 5} mai multe")
        
        print("\n🔍 Detectori disponibili:")
        detectors = nab.list_detectors()
        for detector in detectors[:5]:
            print(f"   - {detector}")
        if len(detectors) > 5:
            print(f"   ... și {len(detectors) - 5} mai mulți")
        
        print("\n[DATA] Dataset Statistics Example:")
        if datasets:
            stats = nab.get_dataset_statistics(datasets[0])
            for key, value in stats.items():
                print(f"   {key}: {value}")
        
        print("\n" + "=" * 60)
        return nab
    
    except Exception as e:
        print(f"[ERROR] Error: {e}")
        return None


if __name__ == "__main__":
    nab = print_nab_info()
