"""Servidor de voz de Pancho y Pancha. Lo abre el mod al entrar a un mundo; también se puede abrir a mano:
  D:/Documentos/BuhoIA/herramientas/venv_chatterbox/Scripts/python.exe servidor_voz.py [--padre PID] [--puerto 47823]

Solo escucha en 127.0.0.1 (nada sale de la PC). Motores:
  - kokoro: procesador, rápido (~1 s por frase). Voces elegidas por Gerardo: Pancho = em_santa normal,
    Pancha = ef_dora grave (~2 semitonos abajo). Es el motor principal.
  - chatterbox: GPU, imita Algenib/Despina (voces/referencias). Lento en la RTX 5060 (~9 s por frase): solo se
    carga si se pide (--chatterbox o POST /cargar_chatterbox). "Calidad máxima (lento)".
  - piper: procesador, respaldo (más robótico).

  GET  /estado  -> {"kokoro": "...", "chatterbox": "apagado|cargando|listo|error", "piper": "...", "gpu": "..."}
  POST /hablar  {"motor": "kokoro|chatterbox|piper", "personaje": "pancho|pancha", "texto": "...", "idioma": "es|en"}
                -> PCM 16 bits mono a 24 kHz (cabecera X-Frecuencia). 503 si el motor no está listo.
  POST /cargar_chatterbox -> empieza a cargar Chatterbox (si no estaba).
Se cierra solo si el proceso de Minecraft (--padre) deja de existir: así nunca queda la GPU ocupada.
Para los demás jugadores va como paquete de voz (PanchoVoz.exe, solo Kokoro): ver paquete_voz/LEEME.md.
"""
import argparse
import ctypes
import io
import json
import logging
import os
import sys
import threading
import time
import wave
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

# Dos formas de correr: el script en la PC de Gerardo (Kokoro + Piper + Chatterbox) o el paquete de voz para los
# jugadores (PanchoVoz.exe, armado con paquete_voz/armar_paquete.py): solo Kokoro, con las voces al lado del .exe.
PAQUETE = getattr(sys, "frozen", False)
RAIZ = Path(sys.executable).resolve().parent if PAQUETE else Path(__file__).resolve().parent.parent
REFS = RAIZ / "voces" / "referencias"
VOCES_PIPER = RAIZ / "voces_piper"
VOCES_KOKORO = RAIZ / "voces_kokoro"
HZ = 24000                                         # todo el audio sale a 24 kHz (lo que espera el mod)

# Voces elegidas por Gerardo (voces/LEEME.md)
KOKORO = {"pancho": {"voz": "em_santa", "tono": 1.0}, "pancha": {"voz": "ef_dora", "tono": 0.89}}
# en inglés (opción "Idioma" del mod): el mismo "Santa" para Pancho; Pancha con una voz femenina inglesa, más grave
KOKORO_EN = {"pancho": {"voz": "am_santa", "tono": 1.0}, "pancha": {"voz": "af_heart", "tono": 0.92}}
CHATTERBOX = {
    "pancho": {"ref": REFS / "pancho_ref.wav", "exaggeration": 0.8, "cfg_weight": 0.3},
    "pancha": {"ref": REFS / "pancha_ref.wav", "exaggeration": 0.5, "cfg_weight": 0.5},
}
PIPER = {"pancho": "es_MX-ald-medium", "pancha": "es_AR-daniela-high"}

logging.basicConfig(filename=str((RAIZ if PAQUETE else Path(__file__).parent) / "servidor_voz.log"), level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(message)s", encoding="utf-8")
log = logging.getLogger("voz")

estado = {"kokoro": "cargando", "chatterbox": "apagado", "piper": "cargando", "gpu": ""}
gpu_lock = threading.Lock()
cb_iniciado = threading.Lock()
kokoro = None
cb_modelo = None
cb_conds = {}
piper_voces = {}


# ------------------------------------------------------------------ utilidades de audio
def a_24k(pcm16, hz):
    """Cambia la frecuencia a 24 kHz (interpolación lineal: suficiente para voz)."""
    import numpy as np
    if hz == HZ:
        return pcm16
    x = np.frombuffer(pcm16, dtype=np.int16).astype(np.float32)
    n = int(len(x) * HZ / hz)
    y = np.interp(np.linspace(0, len(x) - 1, n), np.arange(len(x)), x)
    return y.astype(np.int16).tobytes()


def bajar_tono(audio, factor):
    """Estira el audio 1/factor: suena más grave (tono x factor) y más lento. Por eso Kokoro lo genera antes más
    rápido (velocidad 1/factor): queda grave a la velocidad normal. Así no hace falta ffmpeg (medido 18-nov-2026:
    mismo tono que con rubberband, ±5 Hz)."""
    import numpy as np
    if factor == 1.0:
        return audio
    n = int(len(audio) / factor)
    return np.interp(np.linspace(0, len(audio) - 1, n), np.arange(len(audio)), audio)


# ------------------------------------------------------------------ carga de motores
def cargar_kokoro():
    global kokoro
    try:
        from kokoro_onnx import Kokoro
        kokoro = Kokoro(str(VOCES_KOKORO / "kokoro-v1.0.onnx"), str(VOCES_KOKORO / "voices-v1.0.bin"))
        kokoro.create("hola", voice="em_santa", speed=1.0, lang="es")   # calentar
        estado["kokoro"] = "listo"
        log.info("kokoro listo")
    except Exception as e:
        estado["kokoro"] = "error"
        log.exception("kokoro: %s", e)


def cargar_piper():
    if PAQUETE:
        estado["piper"] = "no instalado"
        return
    try:
        from piper import PiperVoice
        for p, nombre in PIPER.items():
            piper_voces[p] = PiperVoice.load(str(VOCES_PIPER / f"{nombre}.onnx"))
        estado["piper"] = "listo"
        log.info("piper listo")
    except Exception as e:
        estado["piper"] = "error"
        log.exception("piper: %s", e)


def iniciar_chatterbox():
    if PAQUETE:                                          # el paquete de voz no trae Chatterbox
        estado["chatterbox"] = "no instalado"
        return
    with cb_iniciado:
        if estado["chatterbox"] in ("cargando", "listo"):
            return
        estado["chatterbox"] = "cargando"
    threading.Thread(target=cargar_chatterbox, daemon=True).start()


def cargar_chatterbox():
    global cb_modelo
    try:
        t0 = time.time()
        import torch
        from chatterbox.mtl_tts import ChatterboxMultilingualTTS
        if not torch.cuda.is_available():
            raise RuntimeError("sin GPU CUDA")
        estado["gpu"] = torch.cuda.get_device_name(0)
        cb_modelo = ChatterboxMultilingualTTS.from_pretrained(device="cuda")
        for p, cfg in CHATTERBOX.items():               # la "huella" de cada voz se calcula una sola vez
            cb_modelo.prepare_conditionals(str(cfg["ref"]), exaggeration=cfg["exaggeration"])
            cb_conds[p] = cb_modelo.conds
        estado["chatterbox"] = "listo"
        log.info("chatterbox listo en %.1f s (%s)", time.time() - t0, estado["gpu"])
    except Exception as e:
        estado["chatterbox"] = "error"
        log.exception("chatterbox: %s", e)


# ------------------------------------------------------------------ hablar
def hablar_kokoro(personaje, texto, idioma):
    import numpy as np
    cfg = (KOKORO_EN if idioma == "en" else KOKORO)[personaje]
    audio, hz = kokoro.create(texto, voice=cfg["voz"], speed=1.0 / cfg["tono"], lang="en-us" if idioma == "en" else "es")
    pcm = (np.clip(bajar_tono(audio, cfg["tono"]), -1, 1) * 32767).astype(np.int16).tobytes()
    return a_24k(pcm, hz)


def hablar_chatterbox(personaje, texto, idioma):
    import numpy as np
    cfg = CHATTERBOX[personaje]
    with gpu_lock:
        cb_modelo.conds = cb_conds[personaje]
        wav = cb_modelo.generate(texto, language_id=idioma, exaggeration=cfg["exaggeration"], cfg_weight=cfg["cfg_weight"])
    audio = wav.squeeze(0).clamp(-1, 1).cpu().numpy()
    return a_24k((audio * 32767).astype(np.int16).tobytes(), cb_modelo.sr)


def hablar_piper(personaje, texto, idioma):
    if idioma != "es":
        raise SinVoz("Piper solo tiene voces en español")
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        piper_voces[personaje].synthesize_wav(texto, w)
    buf.seek(0)
    with wave.open(buf, "rb") as r:
        return a_24k(r.readframes(r.getnframes()), r.getframerate())


class SinVoz(Exception):
    """El motor no tiene voz para ese idioma: el mod prueba el siguiente."""


MOTORES = {"kokoro": hablar_kokoro, "chatterbox": hablar_chatterbox, "piper": hablar_piper}


class Manejador(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        pass

    def _json(self, codigo, datos):
        cuerpo = json.dumps(datos).encode("utf-8")
        self.send_response(codigo)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(cuerpo)))
        self.end_headers()
        self.wfile.write(cuerpo)

    def do_GET(self):
        if self.path == "/estado":
            self._json(200, estado)
        else:
            self._json(404, {"error": "no existe"})

    def do_POST(self):
        if self.path == "/cargar_chatterbox":
            iniciar_chatterbox()
            return self._json(200, estado)
        if self.path != "/hablar":
            return self._json(404, {"error": "no existe"})
        try:
            pedido = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))).decode("utf-8"))
            motor, personaje, texto = pedido["motor"], pedido.get("personaje", "pancho"), pedido["texto"].strip()
            idioma = "en" if pedido.get("idioma") == "en" else "es"
            if personaje not in KOKORO:
                personaje = "pancho"
            if motor not in MOTORES or estado.get(motor) != "listo":
                return self._json(503, {"error": f"{motor} {estado.get(motor, 'desconocido')}"})
            t0 = time.time()
            try:
                pcm = MOTORES[motor](personaje, texto, idioma)
            except SinVoz as e:
                return self._json(503, {"error": str(e)})
            log.info("%s/%s/%s: %d letras -> %.1f s de audio en %.2f s", motor, personaje, idioma, len(texto),
                     len(pcm) / 2 / HZ, time.time() - t0)
            self.send_response(200)
            self.send_header("Content-Type", "application/octet-stream")
            self.send_header("X-Frecuencia", str(HZ))
            self.send_header("Content-Length", str(len(pcm)))
            self.end_headers()
            self.wfile.write(pcm)
        except Exception as e:
            log.exception("error al hablar")
            self._json(500, {"error": str(e)})


def vigilar_padre(pid):
    """Se cierra si el proceso de Minecraft ya no existe."""
    if os.name == "nt":
        k32 = ctypes.windll.kernel32
        h = k32.OpenProcess(0x00100000, False, pid)      # SYNCHRONIZE
        if not h:
            log.info("no encuentro el proceso padre %s: me cierro", pid)
            os._exit(0)
        k32.WaitForSingleObject(h, 0xFFFFFFFF)           # espera hasta que el padre termine
    else:
        while True:                                      # Mac/Linux: pregunta cada 2 s si sigue vivo
            try:
                os.kill(pid, 0)
            except OSError:
                break
            time.sleep(2)
    log.info("Minecraft se cerró: apago el servidor de voz")
    os._exit(0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--puerto", type=int, default=47823)
    ap.add_argument("--padre", type=int, default=0)
    ap.add_argument("--chatterbox", action="store_true", help="cargar Chatterbox al arrancar")
    a = ap.parse_args()
    if a.padre:
        threading.Thread(target=vigilar_padre, args=(a.padre,), daemon=True).start()
    threading.Thread(target=cargar_kokoro, daemon=True).start()
    threading.Thread(target=cargar_piper, daemon=True).start()
    if a.chatterbox:
        iniciar_chatterbox()
    srv = ThreadingHTTPServer(("127.0.0.1", a.puerto), Manejador)
    log.info("servidor de voz en 127.0.0.1:%d (padre %s)", a.puerto, a.padre)
    print(f"listo en 127.0.0.1:{a.puerto}", flush=True)
    srv.serve_forever()


if __name__ == "__main__":
    main()
