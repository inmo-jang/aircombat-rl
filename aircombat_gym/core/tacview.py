"""TacView export -- .acmi file and real-time telemetry.

JSBSim hands us true latitude, longitude, altitude and attitude, so nothing has
to be reconstructed: the existing autonomy_bt tacview_interface converts pixels
back to lat/lon, and we skip that entirely.

Two ways to watch:

  file      always available.  Writes a .acmi that any TacView (including the
            free one) can open afterwards.
  realtime  needs TacView *Advanced*.  We listen on a TCP port; in TacView pick
            Record -> Real-time Telemetry and give it host:port.

Handshake and record format follow https://www.tacview.net/documentation/acmi/
"""
from __future__ import annotations

import socket
from typing import Iterable

HANDSHAKE_MAGIC = "XtraLib.Stream.0\nTacview.RealTimeTelemetry.0\n"
DEFAULT_PORT = 42674
DEFAULT_PASSWORD = ""

def _header(reference_time: str) -> str:
    return ("FileType=text/acmi/tacview\n"
            "FileVersion=2.1\n"
            f"0,ReferenceTime={reference_time}\n")


def _obj_line(oid: str, lon: float, lat: float, alt_m: float,
              roll_deg: float, pitch_deg: float, yaw_deg: float,
              props: str = "") -> str:
    return (f"{oid},T={lon:.7f}|{lat:.7f}|{alt_m:.1f}"
            f"|{roll_deg:.1f}|{pitch_deg:.1f}|{yaw_deg:.1f}{props}\n")


class AcmiWriter:
    """Collects frames and emits ACMI text.  Subclasses decide where it goes."""

    def __init__(self, reference_time: str = "2026-01-01T00:00:00Z") -> None:
        self.reference_time = reference_time
        self._declared: set[str] = set()
        self._colour: dict[str, str] = {}
        self._locked: dict[str, str | None] = {}
        self._last_t = -1.0
        #: events whose objects are not declared yet
        self._pending: list[tuple[str, list[str]]] = []

    # -- to be provided by the sink ------------------------------------------
    def _emit(self, text: str) -> None:
        raise NotImplementedError

    def start(self) -> None:
        self._emit(_header(self.reference_time))

    def frame(self, t: float, objects: Iterable[dict]) -> None:
        """objects: dicts with id, lat, lon, alt_m, roll, pitch, yaw (deg).

        `name` / `type` are written once, when the id is first seen.  `color` is
        written again whenever it changes, which is how a hit shows up in the
        viewer: TacView cannot draw a weapon cone, so the target is recoloured
        while rounds are landing on it.

        `locked` (another object's id) writes `LockedTarget`, which TacView draws
        as a line to the thing being tracked.
        """
        chunk = []
        if t > self._last_t:
            chunk.append(f"#{t:.2f}\n")
            self._last_t = t
        for o in objects:
            oid = str(o["id"])
            color = o.get("color", "Blue")
            props = ""
            if oid not in self._declared:
                self._declared.add(oid)
                props += (f",Name={o.get('name', 'F-16C')},Color={color}"
                          f",Type={o.get('type', 'Air+FixedWing')}")
            elif color != self._colour.get(oid):
                props += f",Color={color}"
            self._colour[oid] = color
            locked = o.get("locked")
            if locked != self._locked.get(oid):
                props += f",LockedTarget={locked}" if locked else ",LockedTarget=0"
                self._locked[oid] = locked
            chunk.append(_obj_line(oid, o["lon"], o["lat"], o["alt_m"],
                                   o.get("roll", 0.0), o.get("pitch", 0.0),
                                   o.get("yaw", 0.0), props))
        self._emit("".join(chunk))
        if self._pending:
            held, self._pending = self._pending, []
            for text, ids in held:
                self.event(text, *ids)

    def event(self, text: str, *object_ids: str) -> None:
        """A timeline marker.  Shows in TacView's event log and on the seek bar.

        Held until every object it names has been declared.  TacView refuses a
        file whose first event references ids it has not seen -- and the first
        event of a recording normally lands before the first frame.
        """
        ids = [str(i) for i in object_ids]
        if any(i not in self._declared for i in ids):
            self._pending.append((text, ids))
            return
        parts = "|".join(ids)
        self._emit(f"0,Event=Message|{parts}|{text}\n" if parts
                   else f"0,Event=Message|{text}\n")

    def close(self) -> None:
        pass


class AcmiFile(AcmiWriter):

    def __init__(self, path: str, reference_time: str = "2026-01-01T00:00:00Z") -> None:
        super().__init__(reference_time)
        self.path = path
        # BOM 을 붙이면 TacView 가 파일을 열지 못한다
        self._fh = open(path, "w", encoding="utf-8", newline=chr(10))
        self.start()

    def _emit(self, text: str) -> None:
        self._fh.write(text)

    def close(self) -> None:
        if not self._fh.closed:
            self._fh.close()


class AcmiRealtime(AcmiWriter):
    """TCP server TacView Advanced connects to.

    You can attach at any time -- before the sim starts, ten minutes in, or
    again after closing the viewer.  Every `frame()` first tries a non-blocking
    accept, and a viewer that arrives late is brought up to date rather than
    dropped into the middle of the stream:

      1. it gets the ACMI header, which the stream cannot be parsed without
      2. `_declared` is cleared, so the next frame re-sends every object's
         Name / Color / Type -- those are normally written once, on first
         sighting, and a late viewer would otherwise track unnamed objects

    `wait()` is still there for when you want the recording complete from t=0.
    """

    def __init__(self, port: int = DEFAULT_PORT, host: str = "",
                 password: str = DEFAULT_PASSWORD, callsign: str = "aircombat",
                 reference_time: str = "2026-01-01T00:00:00Z") -> None:
        super().__init__(reference_time)
        self.port = port
        self.password = password
        self.callsign = callsign
        self._srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._srv.bind((host, port))
        # A backlog of 1 meant TacView's retries timed out behind the stale
        # connection it had already given up on: attempt 1 connected, 2-5 did
        # not.  Measured 2026-08-11.
        self._srv.listen(8)
        self._sock: socket.socket | None = None

    @property
    def address(self) -> str:
        """What to type into TacView.  It used to advertise whichever adapter
        Windows named first, and that address did not connect."""
        return f"localhost:{self.port}"

    def poll(self) -> None:
        """Accept a waiting viewer.  Call it every render frame: `frame()` also
        accepts, but only while the simulation is stepping."""
        self._try_accept()

    def wait(self, timeout: float | None = None) -> bool:
        """Block until a viewer attaches.  Optional -- `frame()` also accepts."""
        self._srv.settimeout(timeout)
        try:
            sock, _ = self._srv.accept()
        except (socket.timeout, BlockingIOError, OSError):
            return False
        self._attach(sock)
        return True

    def frame(self, t: float, objects) -> None:
        self._try_accept()
        super().frame(t, objects)

    def _try_accept(self) -> None:
        if self._sock is not None:
            return
        self._srv.settimeout(0.0)
        try:
            sock, _ = self._srv.accept()
        except (BlockingIOError, socket.timeout, OSError):
            return
        self._attach(sock)

    def _attach(self, sock: socket.socket) -> None:
        sock.settimeout(2.0)
        try:
            sock.sendall((HANDSHAKE_MAGIC + self.callsign + "\n"
                          + self.password + "\0").encode())
            sock.recv(1024)                # client handshake, contents unused
        except (socket.timeout, OSError):
            pass                           # TacView does not always reply
        self._sock = sock
        print("  TacView attached", flush=True)
        # header first, then let the next frame re-introduce every object
        self.start()
        self._declared.clear()
        self._colour.clear()
        self._locked.clear()

    def _emit(self, text: str) -> None:
        if self._sock is None:
            return
        try:
            self._sock.sendall(text.encode())
        except OSError:
            self._sock = None              # it may come back; we keep listening
            print("  TacView detached", flush=True)

    @property
    def connected(self) -> bool:
        return self._sock is not None

    def close(self) -> None:
        for s in (self._sock, self._srv):
            try:
                if s is not None:
                    s.close()
            except OSError:
                pass
        self._sock = None


def state_to_object(state, oid: str = "101", name: str = "F-16C",
                    color: str = "Blue") -> dict:
    """AircraftState -> the dict `frame()` wants."""
    import math
    return dict(id=oid, name=name, color=color,
                lat=state.lat, lon=state.lon, alt_m=state.h,
                roll=math.degrees(state.phi),
                pitch=math.degrees(state.theta),
                yaw=math.degrees(state.psi) % 360.0)
