"""Every view of the Brill board table, its pages, and the store of board results.

    elo/.venv/Scripts/python.exe ben/scripts/match/brill/brill_views.py DATA_DIR build
    elo/.venv/Scripts/python.exe ben/scripts/match/brill/brill_views.py DATA_DIR register KEY [--name N] [--conf C] [--engine E] [--default]
    elo/.venv/Scripts/python.exe ben/scripts/match/brill/brill_views.py DATA_DIR show

DATA_DIR is data/brill-replay. DATA_DIR/models.json lists the replayed models whose
saved results the page shows:
    {"default": KEY, "models": {KEY: {"name": ..., "conf": ..., "engine": "checkout"|"image", "added": date}}}
The default model is the one the daily run replays (daily_update.ps1 with no -Model) and
the one the main page shows before Morty AI entered. A model is replayed once on the
past events (runs/KEY.jsonl, autoplay/KEY.jsonl); its results stay saved and viewable
when the default moves on.

Views, one page each, linked from the top of every page:
  ''          as played: Morty AI's own tables in the events it entered, the default
              model's replay before (board-table/, published as the main page)
  real        Morty AI's own tables only (board-table/views/real/)
  replay-KEY  model KEY's replay on every event it has bid, also those Morty AI
              entered (board-table/views/replay-KEY/)

Store of board results: DATA_DIR/results/board-results.parquet and .csv, one row per
view (real and each replay), event and board, with the model behind the row and the
time a real table was played.
"""
import argparse
import json
import pathlib
import sys
from datetime import datetime

import polars as pl

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from brill_board_page import build_page  # noqa: E402
from brill_board_table import build  # noqa: E402

STORE_COLUMNS = ['view', 'model_key', 'model', 'played_at', 'date', 'event', 'tournament_id', 'scoring', 'board',
                 'auction', 'contract', 'declarer', 'tricks', 'result', 'score', 'pct', 'imp', 'result_source', 'robots']


def load_registry(data):
    path = data / 'models.json'
    return json.loads(path.read_text(encoding='utf-8')) if path.exists() else {'default': None, 'models': {}}


def save_registry(data, registry):
    path = data / 'models.json'
    temp = path.with_name(path.name + '.new')
    temp.write_text(json.dumps(registry, indent=1, ensure_ascii=False) + '\n', encoding='utf-8')
    temp.replace(path)


def register(data, key, name=None, conf=None, engine=None, default=False):
    registry = load_registry(data)
    entry = registry['models'].setdefault(key, {'added': f'{datetime.now():%Y-%m-%d}'})
    for field, value in (('name', name), ('conf', conf), ('engine', engine)):
        if value:
            entry[field] = value
    entry.setdefault('name', key)
    if default or not registry.get('default'):
        registry['default'] = key
    save_registry(data, registry)
    return registry


def views_of(data, registry):
    """[(id, view, model key, title, text)] of the pages to build, the main page first."""
    default = registry['default']
    models = registry['models']
    out = [('', 'as-played', default, 'As played',
            f"Morty AI's own tables in the events it entered; before that, the replay of {default}."),
           ('real', 'real', default, 'Real tables only', "Only the events Morty AI entered: its own tables, bid by prod's model of the day.")]
    for key in [default] + sorted(k for k in models if k != default):
        if (data / 'runs' / f'{key}.jsonl').exists():
            out.append((f'replay-{key}', 'replay', key, f'Replay: {key}', f"{models[key]['name']}, replayed on every event it has bid."))
        else:
            print(f'model {key}: no runs/{key}.jsonl, no replay page')
    return out


def build_all(data):
    registry = load_registry(data)
    if not registry.get('default'):
        raise SystemExit(f'{data / "models.json"} names no default model: run register first')
    table = data / 'board-table'
    dd, sd, field = table / 'dd.json', table / 'sd.json', data / 'field.jsonl'
    built, store = [], []
    for view_id, view, key, title, text in views_of(data, registry):
        out = table if not view_id else table / 'views' / view_id
        name = registry['models'][key]['name']
        replay = data / 'runs' / f'{key}.jsonl'
        autoplay = data / 'autoplay' / f'{key}.jsonl'
        try:
            build(field, None if view == 'real' else replay, None if view == 'real' else autoplay, sd, out,
                  view, key, name, data / 'prod-models.csv', dd)
        except SystemExit as error:
            if not view_id:
                raise
            print(f'view {view_id or "main"} not built: {error}')
            continue
        built.append({'id': view_id, 'title': title, 'text': text, 'dir': out, 'model': name})
        if view in ('real', 'replay'):
            store.append(pl.read_parquet(out / 'boards.parquet').with_columns(pl.lit(view_id).alias('view')).select(STORE_COLUMNS))
    nav = [{k: v for k, v in b.items() if k in ('id', 'title', 'text')} for b in built]
    (table / 'views.json').write_text(json.dumps(nav, indent=1, ensure_ascii=False), encoding='utf-8')
    for b in built:
        build_page(b['dir'], b['model'], dd, nav, b['id'])
    results = data / 'results'
    results.mkdir(exist_ok=True)
    every = pl.concat(store, how='vertical_relaxed').sort('view', 'date', 'event', 'board')
    every.write_parquet(results / 'board-results.parquet')
    every.write_csv(results / 'board-results.csv')
    print(f'{len(built)} pages: ' + ', '.join(b['id'] or 'main' for b in built))
    print(f'{every.height} rows in {results / "board-results.parquet"}')
    with pl.Config(tbl_rows=40, tbl_width_chars=200, fmt_str_lengths=60):
        print(every.group_by('view', 'model_key').agg(pl.len().alias('boards'), pl.col('tournament_id').n_unique().alias('events'),
                                                     pl.col('date').min().alias('first'), pl.col('date').max().alias('last'))
              .sort('view', 'first'))


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('data', type=pathlib.Path)
    sub = parser.add_subparsers(dest='command', required=True)
    sub.add_parser('build')
    sub.add_parser('show')
    r = sub.add_parser('register')
    r.add_argument('key')
    r.add_argument('--name')
    r.add_argument('--conf')
    r.add_argument('--engine', choices=('checkout', 'image'))
    r.add_argument('--default', action='store_true')
    a = parser.parse_args()
    if a.command == 'build':
        build_all(a.data)
    elif a.command == 'register':
        print(json.dumps(register(a.data, a.key, a.name, a.conf, a.engine, a.default), indent=1, ensure_ascii=False))
    else:
        print(json.dumps(load_registry(a.data), indent=1, ensure_ascii=False))


if __name__ == '__main__':
    main()
