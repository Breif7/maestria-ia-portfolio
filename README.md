# Maestría en IA · Portafolio aplicado

Lo que aprendo en la Maestría en Inteligencia Artificial (UNIR, 2025–) aplicado a mis propios proyectos, con datos reales cuando los hay y siempre medido contra una línea base.

Aquí solo hay trabajo propio. El temario, las presentaciones y los enunciados de la universidad no se publican.

## Aplicaciones

| Carpeta | Problema | Método | Materia · tema | Proyecto de origen |
| --- | --- | --- | --- | --- |
| [anomalias-gastos](aplicaciones/anomalias-gastos/) | Detectar gastos raros, adivinar la categoría de un movimiento, agrupar tipos de día | Isolation Forest con z robusto por categoría · TF-IDF de n-gramas de caracteres + Naive Bayes vs regresión logística (k-fold estratificado, F1 macro) · k-medias | Aprendizaje Automático · 3, 4, 10, 11 | [LifeOS](https://github.com/Breif7/LifeOS) |
| [analisis-guiones](aplicaciones/analisis-guiones/) | ¿Qué tan fácil se entiende un guion en español? ¿Los arquetipos de gancho se distinguen en el texto? | Índice Fernández-Huerta con contador de sílabas en español · Naive Bayes multinomial escrito a mano, leave-one-out · Spearman | Procesamiento de Lenguaje Natural | Pipeline de videos de finanzas personales (privado) |
| [estimador-impresion-3d](aplicaciones/estimador-impresion-3d/) | Estimar gramos y horas de impresión antes de laminar | Volumen de malla 3MF (tetraedros con signo) · regresión ridge escrita a mano, λ elegido por leave-one-out | Aprendizaje Automático · 2, 13 | Herramientas de impresión 3D (privado) |
| [categorizador-recibos](aplicaciones/categorizador-recibos/) | Que la categoría de un recibo aprenda de los hábitos de cada usuario y no solo del conocimiento genérico de un LLM | Naive Bayes multinomial sobre trigramas del comercio, con umbral de confianza y soporte; leave-one-out vs mayoría | Aprendizaje Automático · 4 | [receipt-accountant](https://github.com/Breif7/receipt-accountant) |
| [evaluacion-recuperacion](aplicaciones/evaluacion-recuperacion/) | ¿El asistente RAG encuentra la nota correcta? | recall@k y MRR sobre preguntas etiquetadas · BM25 como línea base léxica vs embeddings densos (coseno) | Procesamiento de Lenguaje Natural | [second-brain-ai](https://github.com/Breif7/second-brain-ai) |

## Resultados medidos (19-nov-2026)

| Aplicación | Datos | Modelo | Línea base |
| --- | --- | --- | --- |
| Estimador de impresión: horas | 27 piezas laminadas reales | MAE 0.16 h | 0.62 h (siempre la media) |
| Estimador de impresión: gramos | 27 piezas laminadas reales | MAE 10.4 g | 25.0 g |
| Arquetipo de gancho desde el texto | 40 guiones reales, 4 clases | 57.5 % de aciertos | 32.5 % (clase mayoritaria) |
| Longitud del gancho vs vistas | 16 videos publicados | ρ = 0.32 | muestra demasiado chica para actuar |
| BM25 en el set de ejemplo | 6 notas sintéticas | recall@8 0.50 | las preguntas parafraseadas no comparten palabras: ahí entran los embeddings |

Los datos personales (finanzas, notas, métricas de redes) no salen de mi PC: las pruebas de este repositorio usan solo datos sintéticos generados en el código.

## Cómo correrlo

```
python -m venv .venv
.venv\Scripts\activate            # Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
pytest -q                          # aplicaciones en Python
npm test                           # aplicaciones en TypeScript (Node 23.6+, sin dependencias)
```

No usa variables de entorno ni claves. CI corre ambas suites en cada push ([.github/workflows/ci.yml](.github/workflows/ci.yml)).

## Criterios

- Toda métrica se reporta junto a una línea base trivial; si el modelo no la supera, se dice.
- Validación honesta para pocos datos: leave-one-out o k-fold estratificado, nunca evaluar sobre lo mismo que se entrenó.
- Lo más simple que funcione: dos de los modelos están escritos a mano (Naive Bayes, ridge) para que cada paso sea visible.
- Los modelos asesoran; las decisiones con consecuencias (publicar, cobrar, guardar) las sigue tomando una persona o una regla determinista.
