"""
vix_inverse_alert.py — Evalúa el último cierre y envía el correo de alerta.

Corre en GitHub Actions después de actualizar la curva (varias veces por día
hábil: la que primero vea el cierre oficial envía; las demás no repiten).

Configuración (secretos del repo, nunca en el código — el repo es público):
  SMTP_USER       cuenta remitente (p. ej. una cuenta de Gmail)
  SMTP_PASSWORD   contraseña de aplicación de esa cuenta
  ALERT_TO        destinatario(s), separados por coma
  SMTP_HOST       opcional, por defecto smtp.gmail.com
  SMTP_PORT       opcional, por defecto 465 (SSL)
  APP_URL         opcional, enlace al panel en el correo

Estado: data/vix_inverse_alerts.json (qué se evaluó, qué se envió). El panel
lo lee para mostrar el historial.

Uso:
  python scripts/vix_inverse_alert.py              # evaluar y enviar
  python scripts/vix_inverse_alert.py --preview    # escribir el HTML del último evento
  python scripts/vix_inverse_alert.py --test       # enviar un correo de prueba
"""
from __future__ import annotations

import argparse
import json
import os
import smtplib
import ssl
import sys
from datetime import datetime, timezone
from email.message import EmailMessage
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from vix_controller import alerts as al  # noqa: E402

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

CURVE = Path("data/vx_curve_history.parquet")
STATE = Path("data/vix_inverse_alerts.json")
ET = ZoneInfo("America/New_York")
MAX_LOG = 300


def cargar_estado() -> dict:
    if STATE.exists():
        try:
            return json.loads(STATE.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            print("AVISO: estado de alertas ilegible — se reinicia")
    return {"evaluado": None, "eventos": [], "correo": "sin configurar"}


def guardar_estado(st: dict) -> None:
    st["eventos"] = st["eventos"][-MAX_LOG:]
    STATE.parent.mkdir(parents=True, exist_ok=True)
    STATE.write_text(json.dumps(st, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def smtp_config() -> dict | None:
    user, pwd, to = (os.environ.get(k, "").strip() for k in ("SMTP_USER", "SMTP_PASSWORD", "ALERT_TO"))
    if not (user and pwd and to):
        return None
    return {"user": user, "pwd": pwd, "to": [t.strip() for t in to.split(",") if t.strip()],
            "host": os.environ.get("SMTP_HOST", "smtp.gmail.com").strip() or "smtp.gmail.com",
            "port": int(os.environ.get("SMTP_PORT", "465") or 465)}


def enviar(cfg: dict, subject: str, text: str, html_body: str) -> None:
    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"] = f"VIX Controller <{cfg['user']}>"
    msg["To"] = ", ".join(cfg["to"])
    msg.set_content(text)
    msg.add_alternative(html_body, subtype="html")
    ctx = ssl.create_default_context()
    if cfg["port"] == 465:
        with smtplib.SMTP_SSL(cfg["host"], cfg["port"], context=ctx, timeout=30) as s:
            s.login(cfg["user"], cfg["pwd"])
            s.send_message(msg)
    else:
        with smtplib.SMTP(cfg["host"], cfg["port"], timeout=30) as s:
            s.starttls(context=ctx)
            s.login(cfg["user"], cfg["pwd"])
            s.send_message(msg)


def leer_curva() -> pd.DataFrame:
    c = pd.read_parquet(CURVE)
    c.index = pd.DatetimeIndex(c.index).normalize()
    if "front_ok" in c.columns:
        c = c[c["front_ok"].astype(bool)]
    return c


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--preview", action="store_true")
    ap.add_argument("--test", action="store_true")
    a = ap.parse_args()

    cfg = smtp_config()
    app_url = os.environ.get("APP_URL", "").strip()
    st = cargar_estado()
    st["correo"] = "configurado" if cfg else "sin configurar"
    curva = leer_curva()

    if a.test:
        if not cfg:
            print("ERROR: faltan SMTP_USER / SMTP_PASSWORD / ALERT_TO")
            return 1
        ult = curva.dropna(subset=["ratio_m2m1", "ratio_vix3m"]).iloc[-1]
        ev = al.Evento(al.AVISO, fecha_dato=curva.index[-1].date(),
                       fecha_ejecucion=al.siguiente_sesion(curva.index[-1].date()),
                       r1=float(ult["ratio_m2m1"]), r2=float(ult["ratio_vix3m"]),
                       detalle="Correo de PRUEBA: la configuración funciona. No hay ningún evento real.",
                       extra={"invertida": "prueba"})
        enviar(cfg, "VIX Inverse · PRUEBA de alertas", al.cuerpo_texto(ev, app_url),
               al.cuerpo_html(ev, app_url))
        st["prueba_enviada"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
        guardar_estado(st)
        print("Correo de prueba enviado a", ", ".join(cfg["to"]))
        return 0

    ev = al.detectar_evento(curva)
    t_ult = curva.dropna(subset=["ratio_m2m1", "ratio_vix3m"]).index[-1].date()

    # Dato pendiente: en la pasada de la mañana (antes de la apertura) el último
    # cierre esperado debe estar ya. Si no, se avisa una sola vez.
    ahora_et = datetime.now(ET)
    hoy = ahora_et.date()
    esperada = al.sesion_anterior(hoy) if (al.es_sesion(hoy) and ahora_et.hour < 9) else None
    if esperada and t_ult < esperada:
        ev = al.evento_dato(t_ult, esperada)

    if a.preview:
        if ev is None:                         # sin evento hoy: la última SALIDA real
            cv = curva.dropna(subset=["ratio_m2m1", "ratio_vix3m"])
            for i in range(len(cv), 2, -1):
                cand = al.detectar_evento(cv.iloc[:i])
                if cand is not None and cand.tipo == al.SALIDA:
                    ev = cand
                    break
        out = Path("alert_preview.html")
        out.write_text(al.cuerpo_html(ev, app_url or "https://vix-controller.streamlit.app"),
                       encoding="utf-8")
        print("Asunto:", al.asunto(ev))
        print("Vista previa escrita en", out)
        return 0

    enviados = {e["clave"] for e in st["eventos"] if e.get("enviado")}
    registrados = {e["clave"]: e for e in st["eventos"]}
    st["evaluado"] = t_ult.isoformat()
    st["ultima_pasada"] = datetime.now(timezone.utc).isoformat(timespec="seconds")

    if ev is None:
        print(f"Sin evento · último cierre {t_ult} · correo {st['correo']}")
        guardar_estado(st)
        return 0
    if ev.clave in enviados:
        print(f"Evento {ev.clave} ya enviado — no se repite")
        guardar_estado(st)
        return 0

    reg = registrados.get(ev.clave) or {
        "clave": ev.clave, "tipo": ev.tipo, "fecha_dato": ev.fecha_dato.isoformat(),
        "fecha_ejecucion": ev.fecha_ejecucion.isoformat() if ev.fecha_ejecucion else None,
        "r1": ev.r1, "r2": ev.r2, "detalle": ev.detalle, "asunto": al.asunto(ev),
        "enviado": False, "intentos": 0}
    if ev.clave not in registrados:
        st["eventos"].append(reg)
    print(f"EVENTO {ev.tipo}: {al.asunto(ev)}")

    if not cfg:
        reg["motivo"] = "correo sin configurar (faltan secretos SMTP)"
        print("AVISO: correo sin configurar — el evento queda registrado en el panel")
        guardar_estado(st)
        return 0
    try:
        reg["intentos"] = int(reg.get("intentos", 0)) + 1
        enviar(cfg, al.asunto(ev), al.cuerpo_texto(ev, app_url), al.cuerpo_html(ev, app_url))
        reg["enviado"] = True
        reg["enviado_en"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
        reg.pop("motivo", None)
        print("Correo enviado a", ", ".join(cfg["to"]))
        guardar_estado(st)
        return 0
    except Exception as e:                           # noqa: BLE001
        reg["motivo"] = f"fallo SMTP: {type(e).__name__}: {e}"
        guardar_estado(st)
        print("ERROR enviando el correo:", reg["motivo"])
        return 1                                     # el workflow falla → GitHub avisa


if __name__ == "__main__":
    sys.exit(main())
