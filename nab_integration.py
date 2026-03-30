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
        Returnează lista datelor disponibile din NAB
        
        Returns:
            Lista cu numele fișierelor de date (fără extensie)
        """
        # Caută CSV-uri în subdirectoare și în root
        data_files = []
        
        # Caută în subdirectoare
        for subdir in self.data_dir.iterdir():
            if subdir.is_dir():
                data_files.extend(subdir.glob("*.csv"))
        
        # Caută în root
        data_files.extend(self.data_dir.glob("*.csv"))
        
        return [f.stem for f in data_files]
    
    def load_data(self, dataset_name: str) -> pd.DataFrame:
        """
        Încarcă un set de date din NAB
        
        Args:
            dataset_name: Numele setului de date
            
        Returns:
            DataFrame cu coloane: timestamp, value
        """
        # Caută fișierul în subdirectoare și în root
        file_path = None
        
        # Caută în subdirectoare
        for subdir in self.data_dir.iterdir():
            if subdir.is_dir():
                candidate = subdir / f"{dataset_name}.csv"
                if candidate.exists():
                    file_path = candidate
                    break
        
        # Caută în root
        if not file_path:
            file_path = self.data_dir / f"{dataset_name}.csv"
        
        if not file_path.exists():
            raise FileNotFoundError(f"Dataset not found: {dataset_name}")
        
        df = pd.read_csv(file_path)
        return df
    
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
        
        print("\n📊 Disponibile Database:")
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
        
        print("\n📈 Exemplu statistică dataset:")
        if datasets:
            stats = nab.get_dataset_statistics(datasets[0])
            for key, value in stats.items():
                print(f"   {key}: {value}")
        
        print("\n" + "=" * 60)
        return nab
    
    except Exception as e:
        print(f"❌ Error: {e}")
        return None


if __name__ == "__main__":
    nab = print_nab_info()
