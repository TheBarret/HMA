"""
Commands
--------
  ping <x> <y> <radius_m> [type]       spatial lookup at pixel coordinate
  from <feature_id> <radius_m> [type]  spatial lookup from a known feature
  find <type> [field__op=val ...]       attribute filter query
  show <feature_id> [brief]            describe a feature
  top  <type> <field> [n]              top-n by field, descending
  vis  <feature_id>                    list features visible from a feature
  up   <feature_id>                    upstream features in flow network
  down <feature_id>                    downstream features in flow network
  stat                                 terrain summary
  help                                 show this message
  exit / quit                          leave the shell

Types:  peak  ridge  valley  saddle  flat
Ops:    eq  neq  gt  gte  lt  lte  in  contains

Examples
--------
  ping 100 150 200
  ping 100 150 200 peak
  from a5e23b48 300 valley
  find peak prominence__gt=50 prominence__lte=300
  find flat avg_slope__lt=3
  top  peak prominence 5
  show a5e23b48
  show a5e23b48 brief
  vis  a5e23b48
  up   a5e23b48
"""

import shlex
from typing import Optional

from core import ( ClassifiedFeature,PeakFeature, RidgeFeature, ValleyFeature, SaddleFeature, FlatZoneFeature )
from context import AnalyzedTerrain

# ---------------------------------------------------------------------------
#  Helpers
# ---------------------------------------------------------------------------

_TYPE_ALIASES = {
    "peak"   : PeakFeature,
    "ridge"  : RidgeFeature,
    "valley" : ValleyFeature,
    "saddle" : SaddleFeature,
    "flat"   : FlatZoneFeature,
}

_ANSI = {
    "reset"  : "\033[0m",
    "bold"   : "\033[1m",
    "dim"    : "\033[2m",
    "cyan"   : "\033[36m",
    "yellow" : "\033[33m",
    "green"  : "\033[32m",
    "red"    : "\033[31m",
    "grey"   : "\033[90m",
}

#def _c(text: str, *styles: str) -> str:
#    prefix = "".join(_ANSI.get(s, "") for s in styles)
#    return f"{prefix}{text}{_ANSI['reset']}"
def _c(text: str, *styles: str) -> str:
    return text

def _resolve_type(token: str) -> type:
    t = _TYPE_ALIASES.get(token.lower())
    if t is None:
        raise ValueError(
            f"Unknown type '{token}'. Valid: {list(_TYPE_ALIASES)}"
        )
    return t

def _resolve_id(terrain: AnalyzedTerrain, partial_id: str) -> str:
    """Accept full UUID or unambiguous prefix."""
    matches = [fid for fid in terrain._feature_by_id
               if fid.startswith(partial_id)]
    if not matches:
        raise KeyError(f"No feature found matching '{partial_id}'")
    if len(matches) > 1:
        raise KeyError(
            f"Ambiguous prefix '{partial_id}' matches {len(matches)} features. "
            f"Use more characters."
        )
    return matches[0]

def _parse_filters(tokens: list[str]) -> dict:
    """Parse ['field__op=value', ...] into kwargs for .where()."""
    kwargs = {}
    for token in tokens:
        if "=" not in token:
            raise ValueError(f"Filter must be 'field__op=value', got '{token}'")
        key, raw_val = token.split("=", 1)
        # coerce value to float if possible, else keep as string
        try:
            val = float(raw_val)
            if val == int(val):
                val = int(val)
        except ValueError:
            val = raw_val
        kwargs[key] = val
    return kwargs

def _print_feature(terrain: AnalyzedTerrain, feature: ClassifiedFeature,
                   index: Optional[int] = None):
    fid      = feature.feature_id
    short_id = _c(fid[:8], "grey")
    label    = _c(terrain.describe(fid, brief=True), "cyan")
    full     = terrain.describe(fid)
    prefix   = f"  {_c(str(index).rjust(3), 'dim')}  " if index is not None else "  "
    print(f"{prefix}{short_id}  {label}")
    print(f"       {_c(full, 'dim')}")

def _print_results(terrain: AnalyzedTerrain,
                   results: list[ClassifiedFeature],
                   header: str):
    count = len(results)
    print(_c(f"\n{header}  [{count}]", "bold"))
    if not results:
        print(_c("  (no results)", "dim"))
        return
    for i, f in enumerate(results, 1):
        _print_feature(terrain, f, index=i)
    print()

# ---------------------------------------------------------------------------
#  Command handlers
# ---------------------------------------------------------------------------

def _cmd_ping(terrain: AnalyzedTerrain, args: list[str]):
    if len(args) < 3:
        print("Usage: ping <x> <y> <radius_m> [type]")
        return
    x, y, r = float(args[0]), float(args[1]), float(args[2])
    ft       = _resolve_type(args[3]) if len(args) > 3 else None
    results  = terrain.ping_at(x, y, r, feature_type=ft)
    header   = f"ping ({x:.0f}, {y:.0f})  r={r:.0f}m"
    if ft:
        header += f"  type={args[3]}"
    _print_results(terrain, results, header)


def _cmd_from(terrain: AnalyzedTerrain, args: list[str]):
    if len(args) < 2:
        print("Usage: from <feature_id> <radius_m> [type]")
        return
    fid     = _resolve_id(terrain, args[0])
    r       = float(args[1])
    ft      = _resolve_type(args[2]) if len(args) > 2 else None
    results = terrain.ping_from(fid, r, feature_type=ft)
    header  = f"from {fid[:8]}  r={r:.0f}m"
    if ft:
        header += f"  type={args[2]}"
    _print_results(terrain, results, header)


def _cmd_find(terrain: AnalyzedTerrain, args: list[str]):
    if not args:
        print("Usage: find <type> [field__op=value ...]")
        return
    ft      = _resolve_type(args[0])
    kwargs  = _parse_filters(args[1:])
    q       = terrain.query().select(ft)
    if kwargs:
        q = q.where(**kwargs)
    results = q.execute()
    header  = f"find {args[0]}"
    if kwargs:
        header += "  " + "  ".join(f"{k}={v}" for k, v in kwargs.items())
    _print_results(terrain, results, header)


def _cmd_show(terrain: AnalyzedTerrain, args: list[str]):
    if not args:
        print("Usage: show <feature_id> [brief]")
        return
    fid   = _resolve_id(terrain, args[0])
    brief = len(args) > 1 and args[1].lower() == "brief"
    desc  = terrain.describe(fid, brief=brief)
    print(f"\n  {_c(fid[:8], 'grey')}  {_c(desc, 'cyan')}\n")


def _cmd_top(terrain: AnalyzedTerrain, args: list[str]):
    if len(args) < 2:
        print("Usage: top <type> <field> [n]")
        return
    ft      = _resolve_type(args[0])
    field   = args[1]
    n       = int(args[2]) if len(args) > 2 else 10
    results = (terrain.query()
               .select(ft)
               .order_by(field, descending=True)
               .limit(n)
               .execute())
    _print_results(terrain, results, f"top {n}  {args[0]}  by {field}")


def _cmd_vis(terrain: AnalyzedTerrain, args: list[str]):
    if not args:
        print("Usage: vis <feature_id>")
        return
    fid        = _resolve_id(terrain, args[0])
    visible_ids = terrain.visibility_graph.get(fid, set())
    results    = [terrain._feature_by_id[vid]
                  for vid in visible_ids
                  if vid in terrain._feature_by_id]
    _print_results(terrain, results, f"visible from {fid[:8]}")


def _cmd_up(terrain: AnalyzedTerrain, args: list[str]):
    if not args:
        print("Usage: up <feature_id>")
        return
    fid     = _resolve_id(terrain, args[0])
    results = (terrain.query()
               .select(ValleyFeature)   # flow network is valley-centric
               .upstream_of(fid)
               .execute())
    _print_results(terrain, results, f"upstream of {fid[:8]}")


def _cmd_down(terrain: AnalyzedTerrain, args: list[str]):
    if not args:
        print("Usage: down <feature_id>")
        return
    fid     = _resolve_id(terrain, args[0])
    results = (terrain.query()
               .select(ValleyFeature)
               .downstream_of(fid)
               .execute())
    _print_results(terrain, results, f"downstream of {fid[:8]}")


def _cmd_stat(terrain: AnalyzedTerrain, _args: list[str]):
    hm    = terrain.source_heightmap
    h, w  = hm.shape
    scale = hm.config.horizontal_scale
    elev  = hm.data

    print(_c("\n  terrain stat", "bold"))
    print(f"  {'grid':<22} {w}x{h} px  ({w*scale:.0f}x{h*scale:.0f}m)")
    print(f"  {'cell size':<22} {scale:.2f} m/px")
    print(f"  {'elevation range':<22} {elev.min():.1f}m — {elev.max():.1f}m")
    print(f"  {'relief':<22} {elev.max() - elev.min():.1f}m")
    print()
    print(f"  {'peaks':<22} {len(terrain.peaks)}")
    print(f"  {'ridges':<22} {len(terrain.ridges)}")
    print(f"  {'valleys':<22} {len(terrain.valleys)}")
    print(f"  {'saddles':<22} {len(terrain.saddles)}")
    print(f"  {'flat zones':<22} {len(terrain.flat_zones)}")
    print()
    print(f"  {'visibility pairs':<22} {sum(len(v) for v in terrain.visibility_graph.values()) // 2}")
    print(f"  {'flow edges':<22} {sum(len(v) for v in terrain.flow_network.values())}")
    print(f"  {'watersheds':<22} {len(terrain.watersheds)}")
    print(f"  {'connectivity edges':<22} {sum(len(v) for v in terrain.connectivity_graph.values()) // 2}")
    print()


# ---------------------------------------------------------------------------
#  Dispatch table
# ---------------------------------------------------------------------------

_COMMANDS = {
    "ping" : _cmd_ping,
    "from" : _cmd_from,
    "find" : _cmd_find,
    "show" : _cmd_show,
    "top"  : _cmd_top,
    "vis"  : _cmd_vis,
    "up"   : _cmd_up,
    "down" : _cmd_down,
    "stat" : _cmd_stat,
    "help" : lambda _t, _a: print(__doc__),
}

# ---------------------------------------------------------------------------
#  REPL
# ---------------------------------------------------------------------------

def launch(terrain: AnalyzedTerrain, map_name: str = "terrain"):

    prompt  = _c(f"[{map_name}]", "yellow") + _c(": ", "bold")
    banner  = (
        _c("\n  terrain shell", "bold") +
        _c(f"  —  {map_name}", "dim") +
        f"\n  {len(terrain._feature_by_id)} features in memory"
        "\n  type 'help' for commands, 'exit' to quit\n"
    )
    print(banner)

    while True:
        try:
            line = input(prompt).strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not line or line.startswith("#"):
            continue

        if line.lower() in ("exit", "quit"):
            print(_c("  bye", "dim"))
            break

        try:
            tokens  = shlex.split(line)
            cmd     = tokens[0].lower()
            args    = tokens[1:]
            handler = _COMMANDS.get(cmd)

            if handler is None:
                print(_c(f"  unknown command '{cmd}'. type 'help'.", "red"))
                continue

            handler(terrain, args)

        except (KeyError, ValueError, TypeError) as e:
            print(_c(f"  error: {e}", "red"))
        except Exception as e:
            print(_c(f"  unexpected error: {e}", "red"))
