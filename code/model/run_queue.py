#!/usr/bin/env python3
"""
Sequential experiment queue for the single local GPU.

queue.txt holds one `train.py` argument string per line (blank lines and # comments ignored).
A line of the form `!<output file> <shell command>` runs an arbitrary command (e.g. inference) and counts as
done when <output file> exists.
The runner executes the first line whose run is not finished, then RE-READS the file, so lines can be
appended or re-ordered while it is running. A run that fails is not retried in the same session.
Exits when nothing is left to do.

  python run_queue.py [queue.txt]
"""
import shlex
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
PY = sys.executable


def run_dir(args):
    name, seed, fold, runs = None, '0', None, str(ROOT / 'runs')
    it = iter(args)
    for a in it:
        if a == '--name':
            name = next(it)
        elif a == '--seed':
            seed = next(it)
        elif a == '--cv-fold':
            fold = next(it)
        elif a == '--runs-dir':
            runs = next(it)
    sub = f'fold{fold}' if fold is not None else f'seed{seed}'
    p = Path(runs)
    if not p.is_absolute():
        p = ROOT / p
    return p / name / sub


def main():
    qfile = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / 'queue.txt'
    log = open(ROOT / 'logs/queue.log', 'a')
    failed = set()

    def say(msg):
        line = f'[{time.strftime("%m-%d %H:%M:%S")}] {msg}'
        print(line, flush=True)
        log.write(line + '\n'); log.flush()

    while True:
        todo = None
        for raw in qfile.read_text().splitlines():
            line = raw.strip()
            if not line or line.startswith('#') or line in failed:
                continue
            if line.startswith('!'):
                target, _, cmd = line[1:].partition(' ')
                if not (ROOT / target).exists():
                    todo = (line, ('shell', cmd, ROOT / target))
                    break
                continue
            args = shlex.split(line)
            if not (run_dir(args) / 'result.json').exists():
                todo = (line, args)
                break
        if todo is None:
            say('queue empty - exiting')
            return
        line, args = todo
        say(f'START {line}')
        t0 = time.time()
        if isinstance(args, tuple):
            _, cmd, target = args
            rc = subprocess.call(cmd, shell=True, cwd=str(ROOT),
                                 stdout=open(ROOT / 'logs/queue_stdout.log', 'a'), stderr=subprocess.STDOUT)
            ok = rc == 0 and target.exists()
        else:
            rc = subprocess.call([PY, str(HERE / 'train.py')] + args, cwd=str(ROOT),
                                 stdout=open(ROOT / 'logs/queue_stdout.log', 'a'), stderr=subprocess.STDOUT)
            ok = rc == 0 and (run_dir(args) / 'result.json').exists()
        if not ok:
            failed.add(line)
            say(f'FAILED rc={rc} after {(time.time() - t0) / 60:.1f} min: {line}')
        else:
            say(f'DONE in {(time.time() - t0) / 60:.1f} min: {line}')


if __name__ == '__main__':
    main()
