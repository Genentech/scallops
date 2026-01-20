from argparse import ArgumentParser

_group_order = {
    "positional arguments": 0,
    "required arguments": 1,
    "optional arguments": 2,
}


def _sort_groups(parser):
    if hasattr(parser, "_action_groups"):
        parser._action_groups.sort(
            key=lambda x: _group_order[x.title] if x.title in _group_order else 3
        )


class ScallopsArgumentParser(ArgumentParser):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if hasattr(self, "_action_groups"):
            for g in self._action_groups:
                if g.title == "options":
                    g.title = "optional arguments"
