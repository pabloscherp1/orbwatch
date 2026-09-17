# ORBWATCH

An open-source Python toolchain for designing a mission to inspect an object
already in orbit: from a public catalogue entry to a delta-v budget and a
passively safe approach trajectory.

Work in progress. What runs today:

- Fetch an object's two-line element set by NORAD ID and propagate it with SGP4
- Convert between inertial, Earth-fixed and geodetic coordinates
- Ground station look angles, lighting and pass prediction
- A live tracker in the browser

Manoeuvre detection, transfer design, proximity operations and budgeting come
next.

The astrodynamics is implemented here rather than imported, with one exception:
SGP4, which nobody should reimplement. Every function documents its reference
frame and its units, and internally the convention is km, km/s, radians and
seconds.

## Install

```
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
pytest -m "not network"
```

## Live tracker

```
python -m orbwatch.gui
```

An orthographic globe with the ground track, footprint, day and night, a ground
station's horizon, live look angles and predicted passes, with time warp. Every
value comes from the tested modules in this package; the browser only draws
them. It binds to 127.0.0.1 and has no authentication, so do not expose it on a
network.

Map data is Natural Earth via world-atlas, drawn with D3. Licences are in
`orbwatch/gui/static/vendor/` and `orbwatch/gui/static/data/`.

## Licence

MIT.
