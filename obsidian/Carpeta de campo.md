# Performance metrics
- Prototipos como [FAU](https://www.mdpi.com/1424-8220/25/7/2138?utm_source=chatgpt.com) lograron un mAP@0.5=98,2% (con solo el alfabeto)
	- Esto significa que solo procesaban el resultado cuando el bounding box coincidia al menos un 50%
	- Al procesarlo, tenian un 98,2% de accuracy

Mi objetivo es:
- Hacer algo similar a mediapipe para poder entender como funciona
- Con una RNN o un transformer, traducir de señas a gestos
- Con un transformer, traducir de lenguaje de señas "YO MIRAR PERSONA" a español "Yo mire a esa persona"

Algo con lo que la gente pueda tener conversaciones basicas 
# Dataset
Estoy considerando entre usar [LSA-T](https://github.com/midusi/LSA-T) o [LSA-64](https://facundoq.github.io/datasets/lsa64/). 
- LSA-T es el que mas me gusta porque son oraciones y es muy extenso, pero las oraciones ya estan interpretadas y no tengo acceso a las señas posta. Aparte, esta separado en oraciones en vez de señas
- LSA-64 Son las señas separadas, tiene muy pocas(64) y son señas "raras" que no son usadas super comunmente en oraciones

|ID|Name|H|ID|Name|H|ID|Name|H|ID|Name|H|
|---|---|---|---|---|---|---|---|---|---|---|---|
|01|Opaque|R|17|Call|R|33|Hungry|R|49|Yogurt|B|
|02|Red|R|18|Skimmer|R|34|Map|B|50|Accept|B|
|03|Green|R|19|Bitter|R|35|Coin|B|51|Thanks|B|
|04|Yellow|R|20|Sweet milk|R|36|Music|B|52|Shut down|R|
|05|Bright|R|21|Milk|R|37|Ship|R|53|Appear|B|
|06|Light-blue|R|22|Water|R|38|None|R|54|To land|B|
|07|Colors|R|23|Food|R|39|Name|R|55|Catch|B|
|08|Pink|R|24|Argentina|R|40|Patience|R|56|Help|B|
|09|Women|R|25|Uruguay|R|41|Perfume|R|57|Dance|B|
|10|Enemy|R|26|Country|R|42|Deaf|R|58|Bathe|B|
|11|Son|R|27|Last name|R|43|Trap|B|59|Buy|R|
|12|Man|R|28|Where|R|44|Rice|B|60|Copy|B|
|13|Away|R|29|Mock|B|45|Barbecue|B|61|Run|B|
|14|Drawer|R|30|Birthday|R|46|Candy|R|62|Realize|R|
|15|Born|R|31|Breakfast|B|47|Chewing-gum|R|63|Give|B|
|16|Learn|R|32|Photo|B|48|Spaghetti|B|64|Find|R|

Voy a empezar usando LSA-64 para hacer un modelo que reconozca señas individuales estaticas o con movimiento, y voy a intentar hacerlo para que con la menor cantidad de datos posibles siga entrenando bien.

Una vez que haga eso, voy a fijarme si deberia crear mi propio LSA-T pero con las señas mejor separadas o si deberia usar LSA-T con transformers

Otra cosa que se me ocurre es usar el [[Señario 2.pdf]], extraer los videos de los QR, y hacer un gran dataset augmentation.

Una forma que se me ocurre de hacer dataset augmentation aparte de rotacion(escala no sirve ya que lo normalizo) es construyendo una matriz $X$ de **solo una de las señas**, hacer PCA y
1. Fijarme la magnitud de los autovectores para ver las direcciones en las que una unica seña varia
2. Hacer eso para todas las señas y hacer un mapa 3d donde $z=||v_x-v_y||$ donde me fijo si esas direcciones en las que la seña varia es compartida(la norma se aproxima a 0) o no
# Seleccion de modelos
Mediapipe esta hecho para correr en celulares o en web asi que se entiende que ande medio para el orto con el accuracy. Para eso, lo mejor ahora es
- FrankMocap para hacer todo de una
- Para robustez en un gpu limitado, usar 2 etapas. Detector corporal rapido como [MoveNet](https://www.tensorflow.org/hub/tutorials/movenet?utm_source=chatgpt.com), recortar la imagen para que tenga el tamaño de las manos, y luego generar los landmarks de manos con InterHand/MMPose/FrankMocap hand
- 