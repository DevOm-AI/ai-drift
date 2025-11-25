import time
import threading
import sys
import os
import pandas as pd

from .live_generator import LiveGenerator

HELP = """
Live Monitor Commands:
  add <pattern>      : activate a named pattern (see 'patterns')
  remove <pattern>   : deactivate a named pattern
  patterns           : list all available patterns
  active             : list active patterns
  show [n]           : show last n rows (default 5)
  speed <seconds>    : change generation interval (float)
  batch <size>       : set batch size per tick
  start              : start generator
  stop               : stop generator
  exit / quit        : stop and exit
  help               : show this help
"""

def repl(gen: LiveGenerator):
    print("Live Monitor — interactive mode. Type 'help' for commands.")
    while True:
        try:
            cmd = input("command: ").strip().split()
            if not cmd:
                continue
            action = cmd[0].lower()
            if action == 'help':
                print(HELP)
            elif action == 'patterns':
                print("Available patterns:", gen.list_patterns())
            elif action == 'add':
                if len(cmd) < 2:
                    print("Usage: add <pattern>")
                    continue
                try:
                    gen.add_pattern(cmd[1])
                    print("Added pattern:", cmd[1])
                except KeyError as e:
                    print("Error:", e)
            elif action == 'remove':
                if len(cmd) < 2:
                    print("Usage: remove <pattern>")
                    continue
                gen.remove_pattern(cmd[1])
                print("Removed pattern:", cmd[1])
            elif action == 'active':
                print("Active patterns:", gen.list_active())
            elif action == 'show':
                n = 5
                if len(cmd) > 1:
                    try:
                        n = int(cmd[1])
                    except:
                        pass
                path = os.path.join("artifacts","live_stream","live_batch.csv")
                if os.path.exists(path):
                    df = pd.read_csv(path)
                    print(df.tail(n).to_string(index=False))
                else:
                    print("No live data yet.")
            elif action == 'speed':
                if len(cmd) < 2:
                    print("Usage: speed <seconds>")
                    continue
                try:
                    new_speed = float(cmd[1])
                    gen.interval = new_speed
                    print("Interval set to", new_speed)
                except:
                    print("Bad number")
            elif action == 'batch':
                if len(cmd) < 2:
                    print("Usage: batch <size>")
                    continue
                try:
                    gen.batch_size = int(cmd[1])
                    print("Batch size set to", gen.batch_size)
                except:
                    print("Bad number")
            elif action == 'start':
                gen.start()
                print("Generator started (interval:", gen.interval, "s, batch:", gen.batch_size, ")")
            elif action == 'stop':
                gen.stop()
                print("Generator stopped.")
            elif action in ('exit','quit'):
                gen.stop()
                print("Exiting...")
                break
            else:
                print("Unknown command. Type 'help' for commands.")
        except (KeyboardInterrupt, EOFError):
            gen.stop()
            print("\nKeyboard exit. Stopping generator and exiting.")
            break

def run_monitor(interval=2.0, batch_size=1):
    gen = LiveGenerator(interval=interval, batch_size=batch_size)
    print("Available patterns:", gen.list_patterns())
    # start by default
    gen.start()
    try:
        repl(gen)
    finally:
        gen.stop()

if __name__ == "__main__":
    # default: normal mode (2 seconds)
    run_monitor(interval=2.0, batch_size=1)