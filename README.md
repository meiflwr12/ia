# Clasificador de Desechos con Cámara Web (Python + Flask)

Aplicación web que usa tu cámara para clasificar desechos (vidrio, papel, plástico,
metal, cartón, orgánico) usando un modelo de deep learning entrenado con transfer
learning sobre MobileNetV2.

## Estructura del proyecto

```
clasificador_desechos/
├── app.py                 ← Backend Flask (sirve la web y clasifica imágenes)
├── train_model.py          ← Script para entrenar el modelo con tus imágenes
├── requirements.txt         ← Dependencias de Python
├── dataset/                ← Aquí van tus imágenes de entrenamiento (tú las agregas)
│   ├── vidrio/
│   ├── papel/
│   ├── carton/
│   ├── plastico/
│   ├── metal/
│   └── organico/
├── templates/
│   └── index.html          ← Página web (interfaz con cámara)
└── static/
    ├── style.css
    └── script.js
```

## Paso 1: Instalar Python y VS Code

1. Instala **Python 3.10 u 11** (evita 3.12/13 por compatibilidad con TensorFlow).
2. Instala **VS Code** y la extensión "Python".
3. Abre esta carpeta (`clasificador_desechos`) en VS Code.

## Paso 2: Crear un entorno virtual e instalar dependencias

En la terminal de VS Code (dentro de la carpeta del proyecto):

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Mac / Linux
source venv/bin/activate

pip install -r requirements.txt
```

## Paso 3: Conseguir un dataset de imágenes de desechos

No entrenes con pocas fotos tuyas — necesitas cientos de imágenes por categoría
para que el modelo generalice bien. Te recomiendo:

- **TrashNet** (dataset público, gratuito): buscar "TrashNet dataset github garythung"
  o en Kaggle "trashnet dataset". Trae imágenes ya organizadas en:
  cardboard, glass, metal, paper, plastic, trash.
- Puedes complementarlo tomando tus propias fotos con el celular, en distintas
  posiciones/luces, y agregándolas a las carpetas correspondientes.

Copia (o renombra) las carpetas descargadas dentro de `dataset/`, respetando
que cada subcarpeta = una categoría. Ejemplo:

```
dataset/
├── vidrio/     (fotos de botellas, frascos, etc.)
├── papel/
├── carton/
├── plastico/
├── metal/
└── organico/
```

Mínimo recomendado: 100-150 imágenes por categoría para resultados aceptables;
300+ para resultados buenos.

## Paso 4: Entrenar el modelo

```bash
python train_model.py
```

Esto entrena un modelo usando **MobileNetV2 preentrenado** (transfer learning),
lo cual funciona bien incluso con pocas imágenes y sin GPU potente. Al terminar,
se genera:

- `waste_classifier.h5` → el modelo entrenado
- `class_names.json` → el mapeo de categorías

Esto puede tardar entre 5 y 30 minutos dependiendo de tu computadora y cuántas
imágenes tengas.

## Paso 5: Correr la aplicación web

```bash
python app.py
```

Abre tu navegador en:

```
http://localhost:5000
```

Dale permiso a la página para usar tu cámara, apunta a un objeto y presiona
"Clasificar". El resultado (categoría + confianza) aparecerá en pantalla.

## Notas y ajustes

- Si tu computadora es lenta para entrenar, en `train_model.py` puedes bajar
  `EPOCHS` (por ejemplo a 5) para una prueba rápida, aunque la precisión será menor.
- Puedes agregar o quitar categorías: solo agrega/quita carpetas en `dataset/`
  y vuelve a entrenar.
- Si el modelo se equivoca mucho, la causa más común es tener pocas imágenes
  o imágenes poco variadas (mismo fondo, misma luz). Diversifica las fotos.
