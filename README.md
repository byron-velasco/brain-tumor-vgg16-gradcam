# Brain Tumor Detection — VGG16 + Grad-CAM

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange?logo=tensorflow&logoColor=white)
![Accuracy](https://img.shields.io/badge/Accuracy-97%25-brightgreen)
![Dataset](https://img.shields.io/badge/Dataset-4600%20MRI-lightgrey)
![License](https://img.shields.io/badge/License-MIT-green)

> Transfer Learning con VGG16 para clasificación binaria de tumores cerebrales en imágenes MRI.  
> Grad-CAM y mapas de saliencia confirman atención clínicamente coherente sobre regiones tumorales.

---

## ¿Qué hace este proyecto?

Clasifica imágenes de resonancia magnética cerebral (MRI) en dos categorías — **tumor / sin tumor** — usando Transfer Learning sobre VGG16 preentrenado en ImageNet.

Va más allá de la clasificación: aplica técnicas de explicabilidad (**Grad-CAM** y mapas de saliencia) para visualizar qué regiones de la imagen determinan la predicción del modelo, validando que el razonamiento de la red sea clínicamente coherente antes de confiar en el diagnóstico.

---

## ¿Por qué importa?

Este proyecto demuestra dominio de dos habilidades complementarias: construir un clasificador de alta precisión con Transfer Learning, y validar que ese clasificador **razona de forma interpretable** mediante Grad-CAM — no solo que memoriza patrones del dataset.

Llegar a 97% de accuracy es una cosa. Saber *por qué* el modelo toma cada decisión es otra. Aquí se trabajan las dos:
- Transfer Learning con VGG16 supera al entrenamiento desde cero en datasets pequeños (~4,600 imágenes)
- `class_weight` reduce explícitamente los **falsos negativos** — priorizando el error más costoso
- Grad-CAM confirma que las activaciones corresponden a **estructuras anatómicas reales**, no a artefactos del dataset

**Hallazgo central:** Transfer Learning + class_weight + EarlyStopping eleva la accuracy de 95.11% → **97.17%** con un modelo cuyo razonamiento es verificable.

---

## Dataset

| Fuente | Clases | Total | Distribución |
|--------|--------|-------|--------------|
| [Brain Tumor MRI Dataset — Kaggle](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset) | Tumor / Sin tumor | 4,600 imágenes | 54.6% tumor · 45.4% sano |

**Preprocesamiento:**
- Redimensionamiento a **(224 × 224 × 3)** — entrada estándar de VGG16
- Normalización al rango **[0, 1]**
- Split estratificado: **70% train · 15% val · 15% test**
- Data augmentation: rotaciones, zoom, flips horizontales, ajustes de brillo

---

## Arquitectura

```
Input (224×224×3)
    ↓
VGG16 — capas convolucionales congeladas (pesos ImageNet)
    ↓
GlobalAveragePooling2D
    ↓
Dense(256, relu) → Dropout(0.5)
    ↓
Dense(1, sigmoid)   ← clasificador binario
```

| Parámetro | Valor |
|-----------|-------|
| Optimizer | Adam |
| Learning rate | 1e-5 |
| Loss | Binary Crossentropy |
| class_weight | Frecuencia inversa por clase |
| EarlyStopping | patience=10, monitor=val_loss |
| Épocas máx. | 80 |

---

## Resultados

| Modelo | Accuracy | Loss |
|--------|----------|------|
| CNN desde cero (baseline) | 95.11% | 0.1413 |
| **VGG16 Transfer Learning** ★ | **97.17%** | **0.0882** |

Falsos negativos (tumor predicho como sano): **25 / 690 imágenes de test**

---

## Explicabilidad — Grad-CAM & Saliency Maps

**Grad-CAM** aplicado sobre la última capa convolucional de VGG16:

- **Positivos (tumor):** activación focalizada sobre masas con bordes irregulares — coherente con criterios radiológicos
- **Negativos (sano):** atención distribuida en estructuras subcorticales centrales, sin activar zonas anómalas
- **Mapas de saliencia:** confirman que el modelo no se guía por artefactos ni fondo de imagen

---

## Estructura del repositorio

```
brain-tumor-vgg16-gradcam/
│
├── Deep_learning_v4.ipynb   ← Notebook principal (Colab-ready)
├── README.md
└── requirements.txt
```

> Las imágenes del dataset (~500 MB) no están incluidas.  
> Descargar desde Kaggle: [Brain Tumor MRI Dataset](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset)

---

## Instalación y ejecución

```bash
# Opción recomendada — Google Colab
# Abrir Deep_learning_v4.ipynb directamente en Colab
# El notebook instala dependencias automáticamente

# Local
git clone https://github.com/byron-velasco/brain-tumor-vgg16-gradcam.git
cd brain-tumor-vgg16-gradcam
pip install -r requirements.txt
```

**requirements.txt**
```
tensorflow>=2.10
numpy
matplotlib
scikit-learn
opencv-python
Pillow
```

---

## Decisiones metodológicas

**¿Por qué VGG16 y no una CNN desde cero?**  
Con ~4,600 imágenes, entrenar una red profunda desde cero genera sobreajuste. VGG16 transfiere representaciones de texturas y bordes que generalizan bien a imágenes médicas, reduciendo el tiempo de convergencia y mejorando la robustez.

**¿Por qué class_weight?**  
El dataset tiene desbalance leve (54.6% / 45.4%). En diagnóstico, un falso negativo tiene consecuencias clínicas mucho más graves que un falso positivo. `class_weight` penaliza explícitamente los errores en tumores no detectados.

**¿Por qué Grad-CAM y no solo accuracy?**  
Un modelo con 97% de accuracy podría estar aprendiendo artefactos del dataset. Grad-CAM permite verificar que las activaciones corresponden a estructuras anatómicas reales — haciendo el modelo confiable, no solo preciso.

---

## Licencia

MIT — código libre para uso, modificación y distribución.

El dataset está disponible en Kaggle bajo sus propias condiciones de uso.
