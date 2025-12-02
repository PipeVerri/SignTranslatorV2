[English](Experiments.md), [Español](Experiments.es.md)
# ¿Qué tan posible es hacer traducción en tiempo real de LSA a español?

En este documento voy a documentar mi proceso de pensamiento, qué probé, qué salió mal, y como tengo planeado seguir

---

# Investigación previa y estado del arte

La primera corriente (la que intenté replicar yo al principio) fue:

1. **Entrenar una RNN para identificar glosas individuales** a partir de un video ya segmentado.
2. Hacer **segmentación heurística** en tiempo real: mirar aceleración, ventanas de tamaño fijo, etc. para decidir cuándo alguien “hizo una seña”.
3. Cada vez que las heurísticas dicen “acá hubo una seña”, recorto el pedazo y se lo paso a la RNN.

El problema de este enfoque es que la segmentación heurística **se dejo de lado porque anda muy mal**. Algunos problemas son:
1. Hay coarticulacion: Si hago la seña de yo("me apunto con el dedo indice") y luego la de voy(apunto hacia adelante con el indice):
   - Si las grabara individualmente, el brazo iria desde abajo hacia apuntarme para "yo", o iria desde abajo hacia adelante para "voy"
   - Al hacerlas juntas, primero me apunto, y luego apunto para adelante **sin llegar a bajar el brazo** o a pausar.
2. La velocidad de una misma seña cambia dentro de una frase vs aislada, o hasta entre frases puede haber una gran diferencia(dependiendo de la velocidad del señante)

Después vinieron los **modelos gloss-free**:

- Hacen *todo* en un solo transformer:
  - segmentación de glosas,
  - interpretación textual de las glosas,
  - traducción glosas → español.
- Andan bien cuando tenés muchos datos, pero con datasets chicos (como LSA-T) funcionan peor porque les estás pidiendo aprender **tres tareas distintas a la vez** con poca señal.

Lo más nuevo que está saliendo son cosas **semi-gloss-free**, que se parecen más a lo que quiero hacer con una LLM(explicados posteriormente):

- No dependen tanto de anotaciones perfectas de glosas.
- Se apoyan en modelos de lenguaje grandes para la parte textual.
- Dejan a los modelos de visión/secuencia concentrarse más en la parte de “qué está pasando en el video”.

---

# Experimento 1: RNN simple

## Dataset y setup

Quería empezar con algo manejable, así que usé **LSA64**:
- 64 señas distintas.
- ~50 samples por seña.

La idea era:

- Entrenar una **RNN** que recibe como input cada frame de *landmarks* (pose/manos),
- El output es un **softmax sobre las 64 señas posibles**.
- Para el caso continuo:
  - Le voy pasando frames a medida que se generan.
  - Solo considero que “se hizo una seña” cuando alguna clase pasa cierto **threshold** de probabilidad.

Además, probé una versión más explícita en tiempo real donde:
- Procesaba los videos a **6 fps**.
- Usaba **ventanas de 2 segundos** para el procesamiento en tiempo real.
- Metía un **umbral sobre el softmax** para distinguir “está haciendo la seña X” vs “no está haciendo nada”.

#### Por que use una RNN en vez de una Gated RNN si eran(antes de los transformers) el estado del arte?
Queria probar usando el modelo mas sencillo posible para ver su performance. Las Gated RNNs fueron hechas para resolver el problema de la memoria a largo plazo en RNNs(ya que al hacer muchas recurrencias, se vuelve inestable numericamente). Y como se que no iba a tener problemas de memoria a largo plazo?

En [este](https://ieeexplore.ieee.org/document/279181) paper se puede ver un grafico con la probabilidad de poder llegar a un minimo local util al optimizar vs la distancia temporal de las dependencias
![img.png](../docs/img.png)

A las ~17 dependencias empieza a caer drasticamente la probabilidad, y si los clips de LSA64 duran ~2s, entonces tendria 12 dependencias(si los grabo a 6 FPS)

## Arquitectura y parámetros

- RNN con **5 capas ocultas**.
- Ancho de **144** (el generador de landmarks devuelve 144 parámetros por frame).
- **100 epochs** de entrenamiento.
- Learning rate de \(10^{-4}\).
- Regularización L2 con \(\lambda = 10^{-3}\).

## Prueba 1

Entrenando con videos ya recortados a una sola seña:

- Train accuracy > **99%**.
- Validation accuracy ~ **97%**.

Para usar una red tan sencilla con un conjunto de datos tan pequeño sin hacer ningun tipo de dataset augmentation, fueron muy buenos resultados.  
El problema apareció cuando lo llevé al caso continuo:

- Ahí funcionaba muy mal.
- El modelo **nunca vio secuencias largas con múltiples señas seguidas**, ni momentos de “no se está haciendo nada”.
- Como consecuencia, **nunca aprendió a olvidar cosas del pasado**.

Para intentar arreglar esto, probé:

- En vez de meterle el video completo, se lo fui pasando en **clips de 2 segundos** con **0.1s de separación** entre ventanas.

## Prueba 2

Aunque así mejoraba un poco, seguía siendo muy difícil que el modelo detectara bien la seña correcta en continuo.  
Las razones principales:

1. **Segmentación rota**
   - Si al hacer segmentación recortaba la seña a la mitad, el modelo no la reconocía.
   - Mucho depende de caer justo en la ventana correcta.

2. **El softmax no responde a “¿hay seña o no?”, sino a “¿qué seña es?”**
   - Cuando tenía los brazos relajados, sin moverme, el modelo igual estaba **muy seguro** de que estaba haciendo alguna seña.
   - Para dos señas muy parecidas (ej: *rojo* y *amarillo*), si el modelo predecía “rojo”, la probabilidad puede ser ~60% porque también activa bastante la neurona de “amarillo”.  
     Eso no significa “no hay seña”, sino “no estoy 100% seguro de cuál”.

En ese momento no le di tanta bola al problema de segmentación y me enfoqué en la parte de clasificación “seña vs no-seña”.

### Idea inicial para “no seña”

La solución que intenté fue:

- Incorporar clips de ~2 segundos de **charlas TED** donde se vean las manos del orador pero **no están haciendo señas**.
- Agregar una **clase número 65** que represente “no hay ninguna seña”.
- Entrenar la RNN para que:
  - identifique las 64 señas, y
  - aprenda a mapear esos clips de oradores a la clase “no-seña”.

## Siguientes pasos que salieron de este experimento

Después de ver estos problemas, me quedaron varias ideas sobre qué cambiar:

- **Bajar los FPS** a ~6  
  Quiero que el procesamiento sea lo más de corto plazo posible. Si cada seña dura ~2 segundos, tener ~10 frames por seña es razonable.

- **Cambiar el tipo de RNN**  
  Si las secuencias de interés superan los 10 frames, tiene más sentido usar una **LSTM** o una **GRU** para manejar mejor dependencias temporales largas.

- **Mezclar LSA64 con clips de “no-seña”**  
  No solo entrenar con las señas de LSA64, sino también con videos de oradores de charlas TED moviendo las manos pero sin señar, para que el modelo tenga explícitamente la opción “no-seña”.

### Generalización a otras condiciones

Otro problema potencial:

- En LSA64 todos los señantes están **sentados de la misma manera**, con un fondo y condiciones muy similares.
- Eso probablemente haga que el modelo **generalice mal** a escenarios reales.

Ideas (futuras) para pelear contra eso:

- Tener una red dedicada a predecir si la seña es:
  - de **una mano**, o
  - de **dos manos**.  
  Incluso podría jugar con espejar la imagen para ver si la mano dominante es zurda o derecha, o dejar eso configurable.

- Para la clasificación de señas:
  - Una sola red que clasifique señas de 1 y 2 manos, pero:
    - si la seña es de una mano, poner en 0 la otra y pasarle un flag de “missing”, o  
  - directamente **dos redes**:
    - una enfocada en señas de una mano,
    - otra enfocada en señas de dos manos.

Todavía no tengo claro dónde meter el **logit de “no seña”**:
- no sé si conviene en esa primera etapa (detectar tipo de seña),  
- o recién en la segunda (clasificador de señas específicas).

---

# Experimento 2: RNN más expresiva y más datos

La idea acá fue mantener el enfoque general del experimento 1, pero:

- hacer la RNN más expresiva, y  
- atacar mejor el tema “seña vs no-seña” metiendo más variedad de datos.

## Data processing

- Procesado a **6 fps**.
- Clips de charlas TED de **2 segundos** por clip para la clase “no-seña”.
- Para la escala de las manos, uso los **puntos de mano del pose** para normalizar.

## Modelo

- Misma idea general de RNN, pero:
  - más ancha: capas ocultas de **300 unidades**. Agregarle profundidad resultaba en overfit sin performance notable en el validation set, agregarle anchura fue lo que ayudo.

## Resultados

- Solo cuando ponés las manos como **orador de charla TED** el modelo lo clasifica como “no sign”.
  - Está claramente **sesgado** por ese tipo de datos.
- Cuando repetís muchas veces una seña, la termina tomando bien.
  - Pero el problema grande son las **ventanas intermedias** entre “no-seña” y “seña”:
    - no se parecen a alguien con las manos abajo,
    - tampoco a una seña bien formada,
    - y el modelo las clasifica como “alguna seña”.

Conclusión de este experimento:

> Para trabajar con señas continuas voy a tener que hacer que el modelo también aprenda **segmentación temporal**, porque no hay forma de resolver esto solo con ventanas fijas.  
> La duración de las señas varía demasiado y las transiciones importan un montón.

## Idea: usar LSA-T + segmentación débil

Acá aparece una idea más interesante:

- Tomar datasets continuos tipo **LSA-T**, los cuales son videos de un canal de noticias en LSA  subtitulados **en español**(no tengo acceso directo a las glosas)
- Convertir los textos continuos en **glosas** usando un modelo de NLP.
- Usar esas glosas para hacer **segmentación temporal débilmente supervisada**:
  - no tengo alineaciones perfectas, pero sí una estructura aproximada de qué glosa va en qué parte.
- Con eso, entrenar algo parecido a mi RNN original pero:
  - usando ya una estructura temporal más razonable,
  - sin depender de ventanas fijas arbitrarias.

Un [paper similar](https://arxiv.org/abs/2505.15438) a mi idea:

## Cosas para probar

- Probar la [**técnica basada en PCA**](data_augmentation/02_PCA_augmentation/README.md) para aumento de datos o la de [movimiento kinematico](data_augmentation/01_kinematic_augmentation/README.md).
  - Igual me parece medio innecesaria en este punto, porque el problema principal no es el reconocimiento de glosas sino la segmentación.
  - Aun así podría testearla en una versión chica y bien overfiteable de LSA64.
- Una **RNN de dos etapas** para que si estoy haciendo una seña solo con la mano dominante, no mire la otra mano(reduce el overfitting ya que si los oradores tienden a poner la otra mano de una manera especifica, el modelo generaliza mal):
  - una etapa para detectar “tipo de seña(mano dominante o ambas manos) / no-seña”,
  - otra para la clasificación fina.
- Probar **semi-supervised learning**:
  - etiquetar a mano partes de LSA-T,
  - usar eso para mejorar las alineaciones y ayudar a la segmentación.

---

# Experimento 3: ir a LSA-T con modelos más grandes
Se que voy a tener que usar un modelo gloss-free o semi-gloss free. Quiero probar modificando la arquitectura de [LSA-T](https://sedici.unlp.edu.ar/bitstream/handle/10915/176192/Documento_completo.pdf-PDFA.pdf?sequence=1&isAllowed=y)

En el paper original, alimentan al transformer el resultado de 3 convoluciones a lo largo del tiempo con un kernel de tamaño 1. Eso significa que por cada frame **representan la pose entera usando solo 3 parametros**. Siento que es una reduccion de dimensionalidad demasiado grande

## Qué quiero probar

- Usar **~50** temporales en vez de 3.
- O directamente:
  - **no usar convoluciones** y pasar los landmarks “as is” a un modelo secuencial (RNN/Transformer).

## Cara y expresiones

Otra cosa que no me convence:

- Meter información de la cara de forma **cruda**.
- Me parece mejor idea:
  - tener un transformer o RNN que **interprete la expresión** a partir de los landmarks de la cara(puedo usar uno ya entrenado para sentiment analysis), y
  - pasar esa **representación más limpia** al modelo principal dedicado a la traducción.

En otras palabras:
- un modelo que se encargue de “qué está pasando en la cara”, y
- otro que se enfoque en “qué seña se está haciendo y qué significa”, sin que tenga que lidiar con landmarks faciales raw.

---