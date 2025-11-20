from utils.mp_utils.parse import mp_to_arr
import numpy as np
from utils.segment_tree import SegmentTree
from dataclasses import dataclass, field
from typing import List, Dict
from pathlib import Path

np.seterr(divide='raise', invalid='raise')

class Landmarks:
    _neutral_hand = None

    @dataclass
    class Hand:
        lm: List[np.ndarray] = field(default_factory=list)
        empty: SegmentTree = field(default_factory=SegmentTree)
        angles: Dict[int, np.ndarray] = field(default_factory=dict)

    def __init__(self, max_frames_interpolation=48):
        self.pose = []
        self.empty_pose = SegmentTree()
        self.left = self.Hand()
        self.right = self.Hand()
        self.max_frames_interpolation = max_frames_interpolation

        if Landmarks._neutral_hand is None:
            path = Path(__file__).resolve().with_name("neutral_hand.npy")
            Landmarks._neutral_hand = np.load(path)

    def add(self, pose, left, right):
        self._mediapipe_parser(pose, self.pose, self.empty_pose, pose=True)
        self._mediapipe_parser(left, self.left.lm, self.left.empty)
        self._mediapipe_parser(right, self.right.lm, self.right.empty)

    def _mediapipe_parser(self, lm, arr, s, pose=False):
        if lm:
            r_idx = 25 if pose else len(lm.landmark)
            arr.append(mp_to_arr(lm.landmark[:r_idx]))
        else:
            s.add_point(len(arr))
            arr.append(None)

    def _interpolate(self, start, end, arr):
        interpol_diff = arr[end + 1] - arr[start - 1]
        interpol_length = end - start + 1
        interpol_diff = interpol_diff / interpol_length
        for i in range(interpol_length):
            yield arr[start - 1] + interpol_diff * (i + 1)
        yield None

    def get_landmarks(self):
        # Un generador que va retornando los landmarks a procesar
        current_frame = 0
        pose_interpolated = None

        while True:
            # Empezar interpolando los datos y sacando los rangos missing demasiado grandes
            # Empezar con el pose asi tengo algo de lo cual puedo agarrar las manos
            if current_frame == len(self.pose[0]) - 1 or len(self.pose) == 0:
                continue

            if pose_interpolated is not None: # Fijarme si estoy interpolando
                pose_frame = next(pose_interpolated)
                if pose_frame is None:
                    pose_interpolated = None

            # Fijarme si puedo retornar la pose asi nomas
            if (self.pose[current_frame] is not None) and (pose_interpolated is None):
                pose_frame = self.pose[current_frame]
            else:
                # Fijarme si tengo limite por derecha para interpolar
                start, end = self.empty_pose.get_interval(current_frame)
                if end != current_frame: # Si hay limite y puedo interpolar
                    interpol_length = end - start + 1
                    # Fijarme si tomarlo como 2 secuencias diferentes o no
                    if interpol_length > self.max_frames_interpolation:
                        current_frame = end + 1 # Saltar a la siguiente secuencia
                        pose_frame = self.pose[current_frame]
                    else:
                        pose_interpolated = self._interpolate(start, end, self.pose)
                        pose_frame = next(pose_interpolated)
                else: # Si no puedo interpolar, esperar a re-capturar la pose o que haya pasado demasiado tiempo
                    if current_frame - start >= self.max_frames_interpolation:
                        break
                    else:
                        continue

            # Setearlo para saber el pose_frame generado en el i frame. No hay problema porque no lo cuenta dentro del SegmentTree
            self.pose[current_frame] = pose_frame

            # Ahora procesar las manos
            left_frame = self._process_hand(self.left, current_frame, pose_frame, 13, 15)
            right_frame = self._process_hand(self.right, current_frame, pose_frame, 14, 16)

            # Visualizarlas
            yield pose_frame, left_frame, right_frame
            current_frame += 1

    def _rodrigues(self, vec1, vec2):
        v1 = vec1 / np.linalg.norm(vec1)
        v2 = vec2 / np.linalg.norm(vec2)

        # Producto cruzado y producto punto
        cross = np.cross(v1, v2)
        dot = np.dot(v1, v2)

        # Caso: vectores casi iguales → no se necesita rotación
        if dot > 0.999999:
            return np.eye(3)

        # Caso: vectores opuestos → rotación de 180° alrededor de un eje perpendicular arbitrario
        if dot < -0.999999:
            # buscar un eje perpendicular a v1
            axis = np.cross(v1, np.array([1, 0, 0]))
            if np.linalg.norm(axis) < 1e-6:
                axis = np.cross(v1, np.array([0, 1, 0]))
            axis /= np.linalg.norm(axis)

            # 180 degrees
            angle = np.pi
        else:
            # eje normal
            axis = cross / np.linalg.norm(cross)
            angle = np.arccos(dot)

        # Matriz de rotación (Rodrigues)
        K = np.array([
            [0, -axis[2], axis[1]],
            [axis[2], 0, -axis[0]],
            [-axis[1], axis[0], 0]
        ])

        R = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)
        return R

    def _process_hand(self, hand: Hand, current_frame, pose_frame, elbow_num, wrist_num):
        # Ahora fijarme si puedo retornar la mano
        if hand.lm[current_frame] is not None:
            return hand.lm[current_frame]
        else:
            # Ya checkee el limite de longitud de interpolacion antes. No voy a interpolar por ahora
            start, end = hand.empty.get_interval(current_frame)
            # Fijarme si tengo limite por izquierda
            if start != 0:
                hand_frame = hand.lm[start - 1]
            else:  # Usar la mano default
                hand_frame = Landmarks._neutral_hand

            # Empiezo moviendo la muñeca a 0, 0 asi lo roto usandola como eje de coordenadas
            wrist_pos = hand_frame[0].reshape(1, 3)
            hand_frame = hand_frame - wrist_pos  # Restarle wrist_pos a cada una de las columnas de left_frame

            # Rotar la mano para que sufra la misma rotacion que sufrio el antebrazo
            # Ver que vector y que angulo describen la rotacion de v_antebrazo. Esta dado por el codo y la muñeca(de la pose)
            v_forearm_new = pose_frame[elbow_num] - pose_frame[wrist_num]
            if start != 0:
                v_forearm_old = self.pose[start - 1][elbow_num] - self.pose[start - 1][wrist_num]  # Origen de coordenadas en el codo
                # Ahora calcular el eje y el angulo
                R = self._rodrigues(v_forearm_old, v_forearm_new)
            else:
                # No se como estaba la mano originalmente rotada, asi que aplicarle una rotacion no tiene sentido
                # Quiero encontrar la rotacion que haga que v_forearm_norm = v_mano_norm
                # Para eso puedo usar Rodrigues. El eje de rotacion es v_forearm_norm x v_mano_norm, y calculo el angulo entre ellos
                v_forearm_norm = v_forearm_new / np.linalg.norm(v_forearm_new)
                v_wrist_norm = hand_frame[0] - hand_frame[9]
                v_wrist_norm = v_wrist_norm / np.linalg.norm(v_wrist_norm)
                R = self._rodrigues(v_wrist_norm, v_forearm_norm)

            # Aplicar la rotacion
            hand_frame = (R @ hand_frame.T).T
            # Devolver la mano a sus coordenadas originales
            hand_frame = hand_frame + wrist_pos

            # Escalar la mano para que tenga el ratio correcto
            # (||v_mano|| * x) / ||v_antebrazo|| = 6.1 => ||v_mano|| * x = 6.1 * ||v_antebrazo|| => x  = ||v_mano|| * (6.1 * ||v_antebrazo||)
            scale_factor = np.linalg.norm(hand_frame[9] - hand_frame[0]) / (6.1 * np.linalg.norm(pose_frame[wrist_num] - pose_frame[elbow_num]))
            hand_frame = hand_frame * scale_factor

            # Ahora posicionar la mano en el lugar correcto. El wrist de la mano(0) en el wrist del pose(15)
            # mano_wrist_x + x = pose_wrist_x => x = pose_wrist_x - mano_wrist_x
            x_offset = pose_frame[wrist_num][0] - hand_frame[0][0]
            y_offset = pose_frame[wrist_num][1] - hand_frame[0][1]
            z_offset = pose_frame[wrist_num][2] - hand_frame[0][2]
            hand_frame[:, 0] += x_offset
            hand_frame[:, 1] += y_offset
            hand_frame[:, 2] += z_offset

            return hand_frame