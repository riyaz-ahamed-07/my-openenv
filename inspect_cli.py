"""Read push.py source fully and write to file."""
import os

import openenv.cli
cli_path = os.path.dirname(openenv.cli.__file__)

push_file = os.path.join(cli_path, 'commands', 'push.py')
with open(push_file, 'r', encoding='utf-8', errors='replace') as fh:
    content = fh.read()

safe = ''.join(c if ord(c) < 128 else '?' for c in content)
with open('push_source.txt', 'w', encoding='ascii', errors='replace') as out:
    out.write(safe)
print("Written push_source.txt:", len(safe), "chars")
