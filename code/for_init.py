from collections import OrderedDict


def remove_module(state_dict):
    """Strips all DataParallel/torch.compile wrapper prefixes from state_dict keys.

    Handles any nesting order or repetition of 'module.' and '_orig_mod.',
    e.g. '_orig_mod.module.X', 'module._orig_mod.X', 'module.module.X'.
    """
    prefixes = ('module.', '_orig_mod.')
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k
        changed = True
        while changed:
            changed = False
            for p in prefixes:
                if name.startswith(p):
                    name = name[len(p):]
                    changed = True
        new_state_dict[name] = v
    return new_state_dict


def add_module(old_dict):

    new_dict = dict()
    for x,y in old_dict.items():

        t = 'module.' + x
        new_dict[t] = old_dict[x]
    

    return new_dict
