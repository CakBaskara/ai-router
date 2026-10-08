import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime

from . import chat, config, dispatch, learn
from .config import log, say, usage_today
from .learn import TIERS, classify


def _parse(argv):
    p = argparse.ArgumentParser(prog="ai", description="Route a prompt to the cheapest model that can answer it. With no prompt, open a chat.")
    p.add_argument("prompt", nargs="*", help="prompt text; read from stdin when omitted")
    p.add_argument("-i", "--interactive", action="store_true", help="open an interactive session in the current folder")
    p.add_argument("-t", "--tier", choices=TIERS, help="force a tier")
    p.add_argument("-p", "--provider", choices=["claude", "codex"], help="force a provider")
    p.add_argument("-n", "--dry-run", action="store_true", help="show the route without running it")
    p.add_argument("--no-llm", action="store_true", help="never ask a model to classify")
    p.add_argument("--plain", action="store_true", help="chat as plain text lines instead of the full-screen app")
    p.add_argument("--ml", action="store_true", help="show what the local prompt classifier has learned")
    p.add_argument("--learn", "--ml-train", action="store_true", dest="ml_train",
                   help="let the teacher model label past prompts and improve the reply style, then report")
    p.add_argument("--ml-auto", action="store_true", help=argparse.SUPPRESS)
    p.add_argument("--style", action="store_true", help="show how long and padded the chat answers are, per provider")
    p.add_argument("--quota", action="store_true", help="show the Claude and Codex usage left in the 5-hour and weekly windows")
    return p.parse_args(argv)


def _clock(ts) -> str:
    if not ts:
        return "-"
    when = datetime.fromtimestamp(ts)
    return when.strftime("%H:%M" if when.date() == datetime.now().date() else "%d %b %H:%M")


def quota_report(data: dict, now: float | None = None) -> str:
    rows = []
    for provider in ("claude", "codex"):
        windows = data.get(provider) or {}
        for key, label in (("5h", "5 jam"), ("week", "minggu")):
            if key in windows:
                left = config.quota_left(windows[key], now)
                rows.append(f"{provider:<7} {label:<7} sisa {left:>3}%   reset {_clock(windows[key].get('resets'))}")
        if windows:
            rows.append(f"{'':<7} data dari {_clock(windows.get('at'))}")
        else:
            rows.append(f"{provider:<7} belum ada data" + (" (muncul setelah satu jawaban Claude di chat)"
                                                            if provider == "claude" else ""))
    return "\n".join(rows)


def style_report(data: dict) -> str:
    if not data["replies"]:
        return "belum ada jawaban chat yang tercatat"
    trial = data["trial"]
    state = (f"diuji melawan {trial['against']}: {trial['replies']}/{trial['need']} jawaban" if trial
             else "dipakai")
    rows = [f"gaya {data['active']} {state}", f"{data['replies']} jawaban terakhir", ""]
    rows.append(f"{'provider':<8} {'gaya':<8} {'n':>4} {'kata':>5} {'skor':>5} {'stop':>4}  tanda / reaksi")
    for g in data["groups"]:
        flags = " ".join(f"{k} {v:.0%}" for k, v in g["flags"].items()) or "-"
        said = " ".join(f"{k} {v}" for k, v in g["reactions"].items())
        rows.append(f"{g['provider']:<8} {g['style']:<8} {g['replies']:>4} {g['median_words']:>5} "
                    f"{g['mean_score']:>5} {g['stopped']:>4}  {flags}" + (f" / {said}" if said else ""))
    return "\n".join(rows)


def main(argv=None) -> int:
    for stream in (sys.stdout, sys.stderr):
        stream.reconfigure(encoding="utf-8")
    args = _parse(argv if argv is not None else sys.argv[1:])
    cfg = config.load()

    if args.quota:
        config.save_quota("codex", config.codex_windows(dispatch.read_limits()))
        print(quota_report(config.load_quota()))
        return 0

    if args.style:
        print(style_report(learn.style_report()))
        return 0

    if args.ml_auto:
        learn.teach_if_due(cfg)
        return 0

    if args.ml or args.ml_train:
        c = cfg["classifier"]
        if args.ml_train:
            learn.warm_encoder(lambda text: say(f"· {text}"))
            say(f"· {learn.teach(cfg, lambda text: say(f'· {text}'))} label baru dari guru")
        encoder = learn.get_encoder() if c.get("embed", True) else None
        result = learn.retrain(cfg, encoder) if args.ml_train else learn.Learner.load(encoder).report(cfg)
        print(json.dumps(result, indent=2))
        if args.ml_train:
            say(f"· {learn.learn_style_locked(cfg)}")
        return 0

    prompt = " ".join(args.prompt).strip()
    if not prompt and not args.interactive and not sys.stdin.isatty():
        prompt = sys.stdin.read().strip()
    elif not prompt and not args.interactive and not args.dry_run:
        if args.plain:
            return chat.run(cfg, provider=args.provider, tier=args.tier, use_llm=not args.no_llm)
        from . import tui
        return tui.run(cfg, provider=args.provider, tier=args.tier, use_llm=not args.no_llm)
    if not prompt and not args.interactive:
        _parse(["-h"])
        return 2

    llm_used = False
    if args.tier:
        tier, score, reasons = args.tier, None, ["forced"]
    elif not prompt:
        tier, score, reasons = "medium", None, ["no prompt"]
    else:
        v = classify(prompt, cfg.get("rules", {}), repo=args.interactive)
        tier, score, reasons = v.tier, v.score, v.reasons
        if v.ambiguous:
            guess, why, llm_used = learn.judge(prompt, cfg, lambda text: say(f"· {text}"), not args.no_llm)
            if guess:
                tier = guess
                reasons = reasons + [why]

    providers = [args.provider] if args.provider else chat.lineup(cfg, usage=usage_today())
    if args.interactive:
        providers = providers[:1]

    started = time.time()
    code, used = 1, None
    for i, provider in enumerate(providers):
        route = cfg["tiers"][tier][provider]
        say(f"→ {tier} · {provider} {route['model']} ({route['effort']}) · {'; '.join(reasons) or 'default'}")
        cmd, stdin_text = dispatch.build(provider, route["model"], route["effort"], prompt, args.interactive)
        if args.dry_run:
            return 0
        code, used = dispatch.run(cmd, stdin_text, quiet_stderr=provider != "claude"), (provider, route)
        if code == 0 or i == len(providers) - 1:
            break
        say(f"· {provider} gagal (exit {code}), pindah ke {providers[i + 1]}")

    log({
        "ts": datetime.now().isoformat(timespec="seconds"),
        "cwd": os.getcwd(),
        "mode": "interactive" if args.interactive else "oneshot",
        "tier": tier,
        "score": score,
        "reasons": reasons,
        "llm_classified": llm_used,
        "provider": used[0] if used else None,
        "model": used[1]["model"] if used else None,
        "effort": used[1]["effort"] if used else None,
        "exit": code,
        "seconds": round(time.time() - started, 1),
        "prompt": prompt[:500],
    })
    if learn.teach_due(cfg) or learn.style_due():
        _learn_detached()
    return code


def _learn_detached():
    flags = getattr(subprocess, "DETACHED_PROCESS", 0) | getattr(subprocess, "CREATE_NO_WINDOW", 0)
    try:
        subprocess.Popen([sys.executable, "-m", "airouter", "--ml-auto"], stdin=subprocess.DEVNULL,
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, creationflags=flags)
    except OSError:
        pass
