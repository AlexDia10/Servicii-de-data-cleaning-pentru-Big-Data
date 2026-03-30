# NAB Integration - Ghid de Utilizare

## 📋 Descriere

Proiectul tău a fost integrat cu **NAB (Numenta Anomaly Benchmark)**, un benchmark de referință pentru detecția anomaliilor în date streaming.

## 🗂️ Structura Proiectului

```
implementare/
├── NAB/                          # Repository NAB clonat
│   ├── data/                     # 58 datasetsuri reale și artificiale
│   ├── labels/                   # Etichetele anomaliilor (JSON)
│   ├── nab/                      # Framework NAB
│   │   ├── detectors/            # Algoritmi de detectare
│   │   ├── scorer.py             # Sistem de scoring
│   │   └── run.py                # Runner NAB
│   └── results/                  # Rezultate detecții
│
├── data/
│   └── src/
│       └── main.py               # Script principal (actualizat)
│
├── nab_integration.py            # Modul de integrare NAB
├── spark_nab_detector.py         # Detector Spark + NAB
└── README.md                      # Acest fișier
```

## 🚀 Cum să Utilizezi

### 1. **Rulează Script-ul Principal**

```bash
cd d:\Anul 4\LICENTA\implementare
python data/src/main.py
```

Aceasta va:
- Testa Spark cu date sintetice
- conecta cu NAB și afișa datasetsuri disponibile
- Rula detector Spark + NAB pe primele datasetsuri

### 2. **Utilizează Direct NAB Integration Module**

```python
from nab_integration import NABIntegration

# Inițializează NAB
nab = NABIntegration()

# Listează datasetsuri disponibile
datasets = nab.get_available_datasets()
print(f"Disponibile: {len(datasets)} datasetsuri")

# Încarcă un dataset
data = nab.load_data("artificial_no_anomaly_test_1")
print(data.head())

# Obține statistici
stats = nab.get_dataset_statistics("aws_ec2_cpu_utilization_3f62")
print(stats)

# Încarcă etichetele
labels = nab.load_labels()
```

### 3. **Utilizează Spark Detector**

```python
from spark_nab_detector import SparkAnomalyDetector

# Inițializează detectorul
detector = SparkAnomalyDetector()

# Încarcă datele NAB în Spark
spark_df = detector.load_nab_data_to_spark("aws_ec2_cpu_utilization_3f62")

# Detectează anomalii cu Z-score
anomalies, mean, std = detector.detect_anomalies_zscore(spark_df, threshold=2.0)
anomalies.show()

# Detectează cu IQR
anomalies_iqr, q1, q3, iqr = detector.detect_anomalies_iqr(spark_df, multiplier=1.5)

# Compară metode
results = detector.compare_methods("aws_ec2_cpu_utilization_3f62")
```

## 📊 Datasetsuri Disponibile

NAB conține 58 datasetsuri cu anomalii reale și artificiale:

- **AWS EC2 metrics** - Metrici de CPU, RAM, disk
- **CloudWatch** - Metrici AWS CloudWatch  
- **Synthetic data** - Date artificiale cu anomalii cunoscute
- **Traffic data** - Date despre trafic de rețea
- **Twitter data** - Volume de tweet-uri
- **Advertisement** - Metrici click-uri anunțuri

## 🔍 Metode de Detecție Disponibile

### 1. **Z-Score**
- Detectează valori care se abat de la medie cu mai mult de N deviații standard
- Parameter: `threshold` (default: 2.0)
- Rapid și ușor de implementat

### 2. **Interquartile Range (IQR)**  
- Detectează outliers bazate pe quartile
- Parameter: `multiplier` (default: 1.5)
- Mai robust la distribuții non-normale

### 3. **Windowed Methods**
- Aplică detecție pe ferestre de timp
- Util pentru date cu pattern sezonier
- Parameter: `window_size` (ex: "5 minutes")

## 🎯 Exemplu Complet

```python
from spark_nab_detector import SparkAnomalyDetector

# 1. Inițializează
detector = SparkAnomalyDetector()

# 2. Alege dataset
dataset = "aws_ec2_cpu_utilization_3f62"

# 3. Obține statistici
detector.get_statistics(dataset)

# 4. Compară metode
results = detector.compare_methods(dataset)

# 5. Folosește metoda dorita
spark_df = detector.load_nab_data_to_spark(dataset)
anomalies, mean, std = detector.detect_anomalies_zscore(spark_df, threshold=2.0)

# 6. Analizează rezultate
print(f"Total anomalii detectate: {anomalies.count()}")
anomalies.show(10)
```

## 📈 Utilizare cu Spark la Scară Mare

Pentru procesare pe cluster Spark:

```python
spark = SparkSession.builder \
    .appName("NAB-Anomaly-Detection") \
    .master("spark://master:7077") \  # URL cluster
    .config("spark.executor.memory", "4g") \
    .config("spark.executor.cores", "4") \
    .getOrCreate()

detector = SparkAnomalyDetector()
# ... rest de cod
```

## 📝 Scoring NAB (Evaluere)

Pentru a compara cu alte algoritmi și a obține score NAB:

```bash
cd NAB
python run.py -d your_detector --detect --score --normalize
```

## 🛠️ Dependențe Instalate

- `pyspark` - Framework de procesare distribuită
- `pandas` - Analiza datelor
- `numpy` - Operații numerice
- `scikit-learn` - Machine Learning
- `plotly` - Vizualizare
- `boto3` - AWS integration

## ⚙️ Configurare

Modifică parametri în `spark_nab_detector.py`:

- `threshold` - Prag Z-score (default: 2.0)
- `multiplier` - Multiplicator IQR (default: 1.5)  
- `window_size` - Mărime fereastră timp (default: "5 minutes")

## 🚨 Troubleshooting

### Problem: "ModuleNotFoundError: No module named 'pyspark'"
```bash
pip install pyspark
```

### Problem: "JAVA_HOME not set"
Scriptul va încerca să detecteze automat, dar asigură-te că ai JDK instalat.

### Problem: Erori la citire dateNAB
Verifică că `NAB` folder este în directorul `implementare/`

## 📚 Referințe

- [NAB GitHub](https://github.com/numenta/NAB)
- [NAB Paper](http://www.sciencedirect.com/science/article/pii/S0925231217309864)
- [Spark Documentation](https://spark.apache.org/docs/latest/)

## 🎓 Următorii Pași

1. **Explorează datasetsurile** - Utilizează `nab_integration.py` pentru a analiza date
2. **Implementează algoritmi personalizați** - Extinde `SparkAnomalyDetector`
3. **Evaluează performanță** - Compară cu NAB scoreboard
4. **Scalează** - Rulează pe Spark cluster pentru date mari

---

**Status**: ✅ Integrat și gata de utilizare  
**Data**: 2026-03-30  
**Actualizare**: Adaugă mai mulți algoritmi și optimizări după testare
