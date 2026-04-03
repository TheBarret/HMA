"""
Terrain Shell

Grammar
-------
  SELECT <type> [WHERE <filters>] [ORDER BY <field> [DESC]] [LIMIT <n>] [OFFSET <n>]
  SHOW   <id> [BRIEF]
  VIS    <id>
  UP     <id>
  DOWN   <id>
  PING   <x> <y> <radius_m> [<type>]
  STAT
  HELP
  EXIT | QUIT

<type>    : peak | ridge | valley | saddle | flat
<filters> : field__op=value  [field__op=value ...]
<id>      : full UUID or unambiguous prefix

Operators : eq  neq  gt  gte  lt  lte  in  contains

Examples
--------
  SELECT peak WHERE prominence__gt=50 ORDER BY prominence DESC LIMIT 5
  SELECT valley WHERE avg_slope__lt=10
  SELECT flat ORDER BY area_pixels DESC LIMIT 3
  SELECT ridge WHERE avg_slope__gt=20 ORDER BY avg_slope DESC OFFSET 2 LIMIT 10
  SHOW   a5e23b48
  SHOW   a5e23b48 BRIEF
  VIS    a5e23b48
  UP     a5e23b48
  DOWN   a5e23b48
  PING   256 256 200
  PING   256 256 200 peak
  STAT
"""

import shlex
from typing import List, Optional

from core import (
    ClassifiedFeature,
    FlatZoneFeature,
    PeakFeature,
    RidgeFeature,
    SaddleFeature,
    ValleyFeature,
)
from context import AnalyzedTerrain, TerrainQuery


# ---------------------------------------------------------------------------
#  Constants
# ---------------------------------------------------------------------------

_TYPE_MAP = {
    "peak"   : PeakFeature,
    "ridge"  : RidgeFeature,
    "valley" : ValleyFeature,
    "saddle" : SaddleFeature,
    "flat"   : FlatZoneFeature,
}


# ---------------------------------------------------------------------------
#  Helpers
# ---------------------------------------------------------------------------

def _resolve_type(token: str) -> type:
    t = _TYPE_MAP.get(token.lower())
    if t is None:
        raise ValueError(f"Unknown type '{token}'. Valid: {list(_TYPE_MAP)}")
    return t


def _resolve_id(terrain: AnalyzedTerrain, partial: str) -> str:
    """Resolve a full UUID from an unambiguous prefix."""
    matches = [fid for fid in terrain._feature_by_id if fid.startswith(partial)]
    if not matches:
        raise KeyError(f"No feature matching '{partial}'")
    if len(matches) > 1:
        raise KeyError(
            f"Ambiguous prefix '{partial}' — {len(matches)} matches. "
            f"Use more characters."
        )
    return matches[0]


def _parse_filters(tokens: list[str]) -> dict:
    """
    Parse ['field__op=value', ...] into **kwargs for .where().
    Values are coerced: int > float > str.
    """
    kwargs = {}
    for token in tokens:
        if "=" not in token:
            raise ValueError(f"Filter must be 'field__op=value', got '{token}'")
        key, raw = token.split("=", 1)
        try:
            v: int | float | str = int(raw)
        except ValueError:
            try:
                v = float(raw)
            except ValueError:
                v = raw
        kwargs[key] = v
    return kwargs


# ---------------------------------------------------------------------------
#  Output formatting
# ---------------------------------------------------------------------------

def _short(fid: str) -> str:
    return fid[:8]


def _print_feature(terrain: AnalyzedTerrain, f: ClassifiedFeature,
                   index: Optional[int] = None) -> None:
    fid   = f.feature_id
    label = terrain.describe(fid, brief=True)
    desc  = terrain.describe(fid)
    idx   = str(index).rjust(3) if index is not None else "   "
    print(f"  {idx}  {_short(fid)}  {label}")
    print(f"       {desc}")


def _print_results(terrain: AnalyzedTerrain, results: List[ClassifiedFeature],
                   header: str) -> None:
    print(f"\n{header}  [{len(results)}]")
    if not results:
        print("  (no results)")
    else:
        for i, f in enumerate(results, 1):
            _print_feature(terrain, f, index=i)
    print()


def _print_stat(terrain: AnalyzedTerrain) -> None:
    hm   = terrain.source_heightmap
    h, w = hm.shape
    cs   = hm.config.horizontal_scale
    elev = hm.data

    def _edges(g: dict) -> int:
        return sum(len(v) for v in g.values())

    print("\n  Statistics")
    print(f"  {'Grid':<22} {w}x{h} px  ({w*cs:.0f}x{h*cs:.0f}m)")
    print(f"  {'Cell size':<22} {cs:.2f} m/px")
    print(f"  {'Elevation range':<22} {elev.min():.1f}m — {elev.max():.1f}m")
    print(f"  {'Relief':<22} {elev.max() - elev.min():.1f}m")
    print()
    print(f"  {'Peaks':<22} {len(terrain.peaks)}")
    print(f"  {'Ridges':<22} {len(terrain.ridges)}")
    print(f"  {'Valleys':<22} {len(terrain.valleys)}")
    print(f"  {'Saddles':<22} {len(terrain.saddles)}")
    print(f"  {'Flat zones':<22} {len(terrain.flat_zones)}")
    print()
    print(f"  {'Visibility pairs':<22} {_edges(terrain.visibility_graph) // 2}")
    print(f"  {'Flow edges':<22} {_edges(terrain.flow_network)}")
    print(f"  {'Watersheds':<22} {len(terrain.watersheds)}")
    print(f"  {'Connectivity edges':<22} {_edges(terrain.connectivity_graph) // 2}")
    print()


# ---------------------------------------------------------------------------
#  Command handlers
# ---------------------------------------------------------------------------

def _cmd_select(terrain: AnalyzedTerrain, tokens: list[str]) -> None:
    """
    SELECT <type>
           [WHERE  field__op=value ...]
           [ORDER BY <field> [DESC|ASC]]
           [LIMIT  <n>]
           [OFFSET <n>]
    """
    if not tokens:
        print("Usage: SELECT <type> [WHERE ...] [ORDER BY <field> [DESC]] [LIMIT n] [OFFSET n]")
        return

    ft = _resolve_type(tokens[0])
    q: TerrainQuery = terrain.query().select(ft)

    upper = [t.upper() for t in tokens[1:]]
    orig  = tokens[1:]

    def _idx(kw: str) -> Optional[int]:
        try:
            return upper.index(kw)
        except ValueError:
            return None

    where_i  = _idx("WHERE")
    order_i  = _idx("ORDER")
    limit_i  = _idx("LIMIT")
    offset_i = _idx("OFFSET")

    # WHERE
    filter_tokens: list[str] = []
    if where_i is not None:
        ends = [i for i in (order_i, limit_i, offset_i)
                if i is not None and i > where_i]
        end           = min(ends) if ends else len(orig)
        filter_tokens = orig[where_i + 1 : end]
        q = q.where(**_parse_filters(filter_tokens))

    # ORDER BY
    field    = None
    desc     = False
    if order_i is not None:
        by_i = _idx("BY")
        if by_i is None or by_i != order_i + 1:
            raise ValueError("ORDER must be followed by BY")
        field_i = order_i + 2
        if field_i >= len(orig):
            raise ValueError("ORDER BY requires a field name")
        field = orig[field_i]
        desc  = (field_i + 1 < len(upper) and upper[field_i + 1] == "DESC")
        q = q.order_by(field, descending=desc)

    # LIMIT
    if limit_i is not None:
        if limit_i + 1 >= len(orig):
            raise ValueError("LIMIT requires a value")
        q = q.limit(int(orig[limit_i + 1]))

    # OFFSET
    if offset_i is not None:
        if offset_i + 1 >= len(orig):
            raise ValueError("OFFSET requires a value")
        q = q.offset(int(orig[offset_i + 1]))

    results = q.execute()

    # Header
    parts = [f"SELECT {tokens[0].lower()}"]
    if filter_tokens:
        parts.append("WHERE " + " ".join(filter_tokens))
    if field:
        parts.append(f"ORDER BY {field} {'DESC' if desc else 'ASC'}")
    if limit_i is not None:
        parts.append(f"LIMIT {orig[limit_i + 1]}")
    if offset_i is not None:
        parts.append(f"OFFSET {orig[offset_i + 1]}")

    _print_results(terrain, results, "  " + "  ".join(parts))


def _cmd_show(terrain: AnalyzedTerrain, tokens: list[str]) -> None:
    if not tokens:
        print("Usage: SHOW <id> [BRIEF]")
        return
    fid   = _resolve_id(terrain, tokens[0])
    brief = len(tokens) > 1 and tokens[1].upper() == "BRIEF"
    desc  = terrain.describe(fid, brief=brief)
    print(f"\n  {_short(fid)}  {desc}\n")


def _cmd_vis(terrain: AnalyzedTerrain, tokens: list[str]) -> None:
    if not tokens:
        print("Usage: VIS <id>")
        return
    fid         = _resolve_id(terrain, tokens[0])
    visible_ids = terrain.visibility_graph.get(fid, set())
    results     = [terrain._feature_by_id[vid]
                   for vid in visible_ids if vid in terrain._feature_by_id]
    _print_results(terrain, results, f"  visible from {_short(fid)}")


def _cmd_up(terrain: AnalyzedTerrain, tokens: list[str]) -> None:
    if not tokens:
        print("Usage: UP <id>")
        return
    fid     = _resolve_id(terrain, tokens[0])
    results = terrain.query().select(ValleyFeature).upstream_of(fid).execute()
    _print_results(terrain, results, f"  upstream of {_short(fid)}")


def _cmd_down(terrain: AnalyzedTerrain, tokens: list[str]) -> None:
    if not tokens:
        print("Usage: DOWN <id>")
        return
    fid     = _resolve_id(terrain, tokens[0])
    results = terrain.query().select(ValleyFeature).downstream_of(fid).execute()
    _print_results(terrain, results, f"  downstream of {_short(fid)}")


def _cmd_ping(terrain: AnalyzedTerrain, tokens: list[str]) -> None:
    if len(tokens) < 3:
        print("Usage: PING <x> <y> <radius_m> [<type>]")
        return
    x, y, r = float(tokens[0]), float(tokens[1]), float(tokens[2])
    ft       = _resolve_type(tokens[3]) if len(tokens) > 3 else None
    results  = terrain.ping_at(x, y, r, feature_type=ft)
    header   = f"  ping ({x:.0f}, {y:.0f})  r={r:.0f}m"
    if ft:
        header += f"  [{tokens[3].lower()}]"
    _print_results(terrain, results, header)


# ---------------------------------------------------------------------------
#  Dispatch
# ---------------------------------------------------------------------------

_COMMANDS = {
    "select" : _cmd_select,
    "show"   : _cmd_show,
    "vis"    : _cmd_vis,
    "up"     : _cmd_up,
    "down"   : _cmd_down,
    "ping"   : _cmd_ping,
    "stat"   : lambda t, _: _print_stat(t),
    "help"   : lambda _, __: print(__doc__),
}


# ---------------------------------------------------------------------------
#  REPL
# ---------------------------------------------------------------------------

def launch(terrain: AnalyzedTerrain, map_name: str = "terrain") -> None:

    total = (len(terrain.peaks) + len(terrain.ridges) + len(terrain.valleys)
             + len(terrain.saddles) + len(terrain.flat_zones))

    print(f"\n  terrain shell  —  {map_name}")
    print(f"  {total} features in memory")
    print("  type 'help' for commands, 'exit' to quit\n")

    prompt = f"[{map_name}]: "

    while True:
        try:
            line = input(prompt).strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not line or line.startswith("#"):
            continue
        if line.lower() in ("exit", "quit"):
            break

        try:
            tokens  = shlex.split(line)
            cmd     = tokens[0].lower()
            handler = _COMMANDS.get(cmd)

            if handler is None:
                print(f"  unknown command '{cmd}'. type 'help'.")
                continue

            handler(terrain, tokens[1:])

        except (KeyError, ValueError, TypeError) as e:
            print(f"  error: {e}")
        except Exception as e:
            print(f"  error: {e}")