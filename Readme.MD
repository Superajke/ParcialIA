# ParcialIA – Teledetección con SAR (Sentinel‑1)

Proyecto académico que implementa un flujo completo de procesamiento SAR desde el preprocesamiento y filtrado hasta la clasificación temática agua/no‑agua y la creación de un dataset de pares 512×512 para entrenamiento de modelos de despeckle, alineado con los cuatro retos y ponderaciones definidos por la guía del curso.

## Índice
- Propósito y motivación
- Objetivos
- Datos y alcance
- Estructura del repositorio
- Metodología por puntos
- Resultados destacados
- Cómo reproducir
- Dataset final 512×512
- Sostenibilidad y mejoras futuras
- Créditos

---

## Propósito y motivación
Construir un pipeline reproducible de análisis SAR que permita visualizar, filtrar, clasificar y generar datos etiquetados útiles para investigación y entrenamiento de modelos de denoising, aprovechando la capacidad de Sentinel‑1 para observar bajo nubosidad e iluminación variable. 

## Objetivos
- Preparar y filtrar una secuencia de al menos cinco imágenes de una misma escena con una sola polarización, re‑escaladas para visualización, sin recorte ni registro en la etapa inicial.
- Ejecutar clasificación no supervisada con hasta cuatro clases sobre imagen filtrada y no filtrada, con salida en escala de grises 0–255 y análisis interpretativo de diferencias. 
- Derivar máscaras binarias agua/no‑agua a partir del clustering previo y cuantificar la presencia relativa de agua comparando ambas condiciones. 
- Registrar y fusionar multitemporalmente para obtener un ground truth “limpio”, recortar pares 512×512 y documentar el dataset resultante con metadata formal. 

## Datos y alcance
Se utilizó una secuencia de cinco fechas Sentinel‑1 de la misma región, re‑escaladas a 0–255 para hacer visibles los contrastes radiométricos, produciendo evidencias y entregables por cada reto conforme a la evaluación de la guía. 

## Estructura del repositorio
ParcialIA/
├── Punto1/ # Imágenes y filtrado (re‑escalado, speckle)
├── Punto2/ # Clasificación no supervisada (0–255)
├── Punto3/ # Clasificación agua/no‑agua y métricas
└── Punto4/ # Dataset 512×512 (noisy/ground_truth + metadata)

text
Esta organización asegura trazabilidad de insumos, análisis y salidas por cada punto, facilitando la revisión, la reproducción y la sustentación. 

## Metodología por puntos

### Punto 1: Imágenes y filtrado
- Re‑escalado de intensidad a 0–255 para visualización sin recorte ni registro en esta etapa, seguido de reducción de speckle con filtros apropiados para mejorar interpretabilidad. 
- Evidencias: secuencia re‑escalada y comparativas antes/después por regiones de interés, cumpliendo requerimientos de análisis visual. 

### Punto 2: Clasificación no supervisada
- Clustering sobre imagen filtrada y la misma sin filtrar, con mapeo de clases a 0, 85, 170 y 255 para cumplir el formato 0–255 y facilitar la interpretación temática. 
- Evidencias: mapas de clases, análisis de coberturas y contraste de diferencias entre condiciones, guardando resultados en estructura dedicada. 

### Punto 3: Agua/no‑agua
- Selección de la clase de menor intensidad del clustering como agua, generación de máscara binaria agua=255/no‑agua=0 para ambas condiciones. 
- Evidencias: máscaras binarias, visualización comparativa y cuantificación del porcentaje de agua por condición, con soporte gráfico. 

### Punto 4: Dataset 512×512
- Selección de imagen base (noisy), registro de las demás fechas a la base y fusión multitemporal por promediado para construir el ground truth “limpio”. 
- Recorte emparejado en ventanas 512×512 y documentación en `dataset_info.json` (método de registro, índice de base, método de fusión, tamaño de parche y total de pares). 

## Resultados destacados
- Visualizaciones claras del efecto del filtrado en texturas SAR y de la sensibilidad del clustering a la condición filtrada/no filtrada, en línea con los análisis que demanda la guía. 
- Máscaras agua/no‑agua coherentes con la morfología hídrica local, con cuantificación comparativa entre condiciones y gráficos de apoyo para la discusión. 
- Dataset ML‑ready con pares 512×512 en carpetas paralelas `noisy/` y `ground_truth/`, con metadata formal que habilita entrenamiento y evaluación reproducibles. 

## Cómo reproducir
1. Preparación: ubicar las cinco imágenes de la misma escena y re‑escalarlas a 0–255 sin recorte ni registro, aplicando filtrado de speckle para la etapa comparativa posterior. 
2. Clasificación: ejecutar el clustering sobre la imagen filtrada y la misma sin filtrar, generando salidas 0–255 y visualizaciones comparativas.
3. Agua/no‑agua: derivar las máscaras binarias desde la clase agua del clustering y calcular porcentajes por condición con gráficos de soporte. 
4. Dataset: registrar todas las fechas a una base, fusionar por media para ground truth y recortar pares 512×512, verificando contenido y generando metadata. 

## Dataset final 512×512
Estructura estándar para entrenamiento de redes de denoising:  
Punto4/dataset_patches/
├── noisy/ # Parches de la imagen base (speckle natural)
└── ground_truth/ # Parches de la imagen fusionada (denoised)

text
La metadata incluye método de registro, índice de base, método de fusión, tamaño de parche, total de pares y descripción de carpetas para facilitar la integración con frameworks de aprendizaje. 

## Sostenibilidad y mejoras futuras
- Incorporar métricas objetivas (p. ej., SSIM/PSNR) para evaluar registro y fusión, y estudiar fusión por mediana o ponderada para robustecer el ground truth. 
- Ampliar el número de parches y explorar solapes alternativos para incrementar diversidad y generalización de los modelos de despeckle. 

## Créditos
Proyecto académico desarrollado para la asignatura de Visión por Computador e IA, siguiendo los cuatro retos, alcances y condiciones de evaluación de la Guía 3. 